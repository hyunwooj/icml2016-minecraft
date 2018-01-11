if not dqn then
    require 'util.initenv'
end
require 'model.init'
require 'algorithm.Memory'
require 'algorithm.ReplayBuffer'


local agent = torch.class('dqn.SeparateQAgent')


function agent:__init(args)
    self.state_dim  = args.state_dim or 3*32*32 -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or 0.1
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.minibatch_size = args.minibatch_size or 1

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 4
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 50000
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r or 1
    self.max_reward     = args.max_reward or 1
    self.min_reward     = args.min_reward or -1
    self.clip_delta     = args.clip_delta or 1
    self.target_q       = args.target_q or true
    self.gpu            = args.gpu
    self.ncols          = args.ncols or 3  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 32, 32}
    self.image_dims     = args.image_dims or {self.ncols, 32, 32}
    self.bufferSize       = args.bufferSize or 512
    self.smooth_target_q  = args.smooth_target_q or true
    self.target_q_eps     = args.target_q_eps or 1e-3
    self.clip_grad        = args.clip_grad or 20
    self.transition_params = args.transition_params or {}
    self.name           = args.network

    self.mem_network = self:create_network(self.hist_len-1)
    self.beh_network = self:create_network(self.n_actions)

    if self:use_gpu() then
        self.mem_network:cuda()
        self.beh_network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.mem_network:float()
        self.beh_network:float()
        self.tensor_type = torch.FloatTensor
    end

    self.memory = self:create_memory()
    self.buffer = self:create_replay_buffer()
    self.last_step = nil

    self.numSteps = 0 -- Number of perceived states.

    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.
    self.stat_eps = 1e-3

    self.r_max = 1

    -- Memory
    self.mem_w, self.mem_dw = self.mem_network:getParameters()
    self.mem_dw:zero()

    self.mem_deltas = self.mem_dw:clone():fill(0)
    self.mem_tmp= self.mem_dw:clone():fill(0)
    self.mem_g  = self.mem_dw:clone():fill(0)
    self.mem_g2 = self.mem_dw:clone():fill(0)

    if self.target_q then
        self.mem_target_network = self.mem_network:clone()
        self.mem_target_w = self.mem_target_network:getParameters()
    end

    -- Behavior
    self.beh_w, self.beh_dw = self.beh_network:getParameters()
    self.beh_dw:zero()

    self.beh_deltas = self.beh_dw:clone():fill(0)
    self.beh_tmp= self.beh_dw:clone():fill(0)
    self.beh_g  = self.beh_dw:clone():fill(0)
    self.beh_g2 = self.beh_dw:clone():fill(0)

    if self.target_q then
        self.beh_target_network = self.beh_network:clone()
        self.beh_target_w = self.beh_target_network:getParameters()
    end
end

function agent:perceive(reward, rawframe, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local frame = self:preprocess(rawframe)
    local state = self.memory:concat(frame)

    reward = self:process_reward(reward)

    if self.last_step and not testing then
        local s = self.last_step.s
        local a = self.last_step.a
        local r = reward
        local s2 = state
        local t = terminal
        self.buffer:add(s, a, r, s2, t)
    end

    input = unsqueeze(state, 1):float():clone():div(255)
    if self:use_gpu() then
        input = input:cuda()
    end
    self.mem_network:evaluate()
    self.beh_network:evaluate()
    mem_action, beh_action = self:eGreedy(state, testing_ep)

    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:learn()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end
    local action = (mem_action*(self.n_actions+1)) + beh_action
    assert(action < 256, 'action overflow')
    self.last_step = {s=state, a=action}

    self.memory:replace(frame, mem_action)

    if not testing and self.target_q then
        self:update_to_mem_target()
        self:update_to_beh_target()
    end

    if terminal then
        self.memory:reset()
    end
    return beh_action
end

function agent:learn()
    self:update_lr()

    local s, a, r, s2, t = self.buffer:sample(self.minibatch_size)
    local beh_a = a % (self.n_actions+1)
    local mem_a = a / (self.n_actions+1)
    s = s:float():div(255)
    s2 = s2:float():div(255)
    if self:use_gpu() then
        s = s:cuda()
        s2 = s2:cuda()
    end
    if self.rescale_r then
        r:div(self.r_max)
    end

    -- y = (1-t)
    local y = t:float():mul(-1):add(1)

    self.mem_target_network:evaluate()
    self.beh_target_network:evaluate()
    local mem_q2_max = self.mem_target_network:forward(s2):float():clone():max(2)
    local beh_q2_max = self.beh_target_network:forward(s2):float():clone():max(2)

    -- y = r + (1-t) * gamma * maxQ'
    mem_y = torch.add(r, mem_q2_max:mul(self.discount):cmul(y))
    beh_y = torch.add(r, beh_q2_max:mul(self.discount):cmul(y))

    self.mem_network:training()
    self.beh_network:training()
    local mem_q_all = self.mem_network:forward(s):float():clone()
    local beh_q_all = self.beh_network:forward(s):float():clone()

    -- Memory
    local mem_q = torch.FloatTensor(mem_q_all:size(1))
    for i=1,mem_q_all:size(1) do
        mem_q[i] = mem_q_all[i][mem_a[i]]
    end

    -- dQ = r + (1-t) * gamma * Q' - Q
    local mem_delta = torch.add(mem_y, mem_q:mul(-1))

    -- self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
    --         self.stat_eps*mem_delta:clone():float():abs():mean()

    if self.clip_delta then
        mem_delta[mem_delta:ge(self.clip_delta)] = self.clip_delta
        mem_delta[mem_delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local mem_targets = torch.zeros(self.minibatch_size, self.hist_len-1):float()
    for i=1,math.min(self.minibatch_size,mem_a:size(1)) do
        mem_targets[i][mem_a[i]] = mem_delta[i]
    end
    self:update_mem_parameter(s, mem_targets)

    -- Behavior
    local beh_q = torch.FloatTensor(beh_q_all:size(1))
    for i=1,beh_q_all:size(1) do
        beh_q[i] = beh_q_all[i][beh_a[i]]
    end

    -- dQ = r + (1-t) * gamma * Q' - Q
    local beh_delta = torch.add(beh_y, beh_q:mul(-1))

    self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
            self.stat_eps*beh_delta:clone():float():abs():mean()

    if self.clip_delta then
        beh_delta[beh_delta:ge(self.clip_delta)] = self.clip_delta
        beh_delta[beh_delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local beh_targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,beh_a:size(1)) do
        beh_targets[i][beh_a[i]] = beh_delta[i]
    end
    self:update_beh_parameter(s, beh_targets)
end

function agent:update_to_mem_target()
    if self.smooth_target_q then
        self.mem_target_w:mul(1 - self.target_q_eps)
        self.mem_target_w:add(self.target_q_eps, self.mem_w)
    else
        if self.numSteps % self.target_q == 1 then
            self.mem_target_network = self.mem_network:clone()
        end
    end
end

function agent:update_to_beh_target()
    if self.smooth_target_q then
        self.beh_target_w:mul(1 - self.target_q_eps)
        self.beh_target_w:add(self.target_q_eps, self.beh_w)
    else
        if self.numSteps % self.target_q == 1 then
            self.beh_target_network = self.beh_network:clone()
        end
    end
end

function agent:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        mem_action = torch.random(1, self.hist_len-1)
        beh_action = torch.random(1, self.n_actions)
        return mem_action, beh_action
    else
        return self:greedy(state)
    end
end

function agent:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    pick_best = function(q)
        local maxq = q[1]
        local besta = {1}

        -- Evaluate all other actions (with random tie-breaking)
        for a = 2, #q do
            if q[a] > maxq then
                besta = { a }
                maxq = q[a]
            elseif q[a] == maxq then
                besta[#besta+1] = a
            end
        end

        local r = torch.random(1, #besta)
        return maxq, besta[r]
    end

    self.mem_network:evaluate()
    _, mem_action = pick_best(self.mem_network:forward(state):totable()[1])

    self.beh_network:evaluate()
    max_q, beh_action = pick_best(self.beh_network:forward(state):totable()[1])
    self.v_avg = (1-self.stat_eps)*self.v_avg + self.stat_eps*max_q

    return mem_action, beh_action
end

function agent:update_mem_parameter(s, targets)
    if self:use_gpu() then
        targets = targets:cuda()
    end

    -- zero gradients
    self.mem_dw:zero()

    -- compute gradients
    self.mem_network:backward(s, targets)

    -- gradient clipping
    local grad_norm = self.mem_dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.mem_dw:mul(self.clip_grad / grad_norm)
    end

    self.mem_g:mul(0.95):add(0.05, self.mem_dw)
    self.mem_tmp:cmul(self.mem_dw, self.mem_dw)
    self.mem_g2:mul(0.95):add(0.05, self.mem_tmp)
    self.mem_tmp:cmul(self.mem_g, self.mem_g)
    self.mem_tmp:mul(-1)
    self.mem_tmp:add(self.mem_g2)
    self.mem_tmp:add(0.01)
    self.mem_tmp:sqrt()

    -- accumulate update
    self.mem_deltas:mul(0):addcdiv(self.lr, self.mem_dw, self.mem_tmp)
    self.mem_w:add(self.mem_deltas)
end

function agent:update_beh_parameter(s, targets)
    if self:use_gpu() then
        targets = targets:cuda()
    end

    -- zero gradients
    self.beh_dw:zero()

    -- compute gradients
    self.beh_network:backward(s, targets)

    -- gradient clipping
    local grad_norm = self.beh_dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.beh_dw:mul(self.clip_grad / grad_norm)
    end

    self.beh_g:mul(0.95):add(0.05, self.beh_dw)
    self.beh_tmp:cmul(self.beh_dw, self.beh_dw)
    self.beh_g2:mul(0.95):add(0.05, self.beh_tmp)
    self.beh_tmp:cmul(self.beh_g, self.beh_g)
    self.beh_tmp:mul(-1)
    self.beh_tmp:add(self.beh_g2)
    self.beh_tmp:add(0.01)
    self.beh_tmp:sqrt()

    -- accumulate update
    self.beh_deltas:mul(0):addcdiv(self.lr, self.beh_dw, self.beh_tmp)
    self.beh_w:add(self.beh_deltas)
end

function agent:update_lr()
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)
end

function agent:process_reward(reward)
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end
    return reward
end

function agent:create_network(n_actions)
    local new_args = {}
    new_args.name               = self.name
    new_args.hist_len           = self.hist_len or 10
    -- new_args.n_actions          = self.n_actions or 6
    new_args.n_actions          = n_actions
    new_args.ncols              = self.ncols or 3
    new_args.image_dims         = self.image_dims or {3, 32, 32}
    new_args.input_dims         = self.input_dims or {new_args.hist_len * new_args.ncols, 32, 32}
    new_args.n_units            = self.n_units or {32, 64}
    new_args.filter_size        = self.filter_size or {4, 4}
    new_args.filter_stride      = self.filter_stride or {2, 2}
    new_args.pad                = self.pad or {1, 1}
    new_args.n_hid_enc          = self.n_hid_enc or 256
    new_args.edim               = self.edim or 256
    new_args.memsize            = self.memsize or (new_args.hist_len - 1)
    new_args.lindim             = self.lindim or new_args.edim / 2
    new_args.lstm_dim           = self.edim or 256
    new_args.gpu                = self.gpu or -1
    new_args.conv_dim           = new_args.n_units[#new_args.n_units] * 8 * 8
    new_args.Linear             = nn.LinearNB
    return g_create_network(new_args)
end

function agent:create_memory()
    local mem_args = {
        mem_size = self.hist_len-1
    }
    return Memory(mem_args)
end

function agent:create_replay_buffer()
    local buffer_args = {
        frame_dim = self.state_dim,
        mem_size = self.hist_len,
        max_size = self.replay_memory,
        batch_size = self.minibatch_size,
        nonTermProb = self.nonTermProb,
    }
    return ReplayBuffer(buffer_args)
end

function agent:preprocess(rawstate)
    if rawstate:size()[2] ~= self.image_dims[2] then
        if self.py == nil then
            -- resize image using antialias method
            self.msg, self.py = pcall(require, "fb.python")
            assert(self.msg, "fb.python (facebook package) is not installed!")
            self.py.exec([=[
import numpy as np
from PIL import Image
def resize(x, size):
    image = Image.fromarray(x)
    resized_x = image.resize((int(size), int(size)), Image.ANTIALIAS)
    return np.asarray(resized_x)
              ]=])
        end
        local resized = self.py.eval('resize(x, size)', {x=rawstate:permute(2, 3, 1), size=self.image_dims[2]})
        return resized:permute(3, 1, 2)
    end
    return rawstate
end

function agent:use_gpu()
    return self.gpu and self.gpu >= 0
end

function agent:report()
    print(get_weight_norms(self.network.net))
    print(get_grad_norms(self.network.net))
    print("Grad Norm: " .. tostring(self.dw:norm()))
end
