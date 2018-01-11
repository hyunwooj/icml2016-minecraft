if not dqn then
    require 'util.initenv'
end
require 'model.init'
require 'algorithm.Memory'
require 'algorithm.ReplayBuffer'


local agent = torch.class('dqn.TestAgent')


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

    self.network = self:create_network()

    if self:use_gpu() then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    self.memory = self:create_memory()
    self.buffer = self:create_replay_buffer()
    self.last_step = nil

    self.numSteps = 0 -- Number of perceived states.

    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.
    self.stat_eps = 1e-3

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)
    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
        self.target_w = self.target_network:getParameters()
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
    self.network:evaluate()
    action = self:eGreedy(state, testing_ep)

    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:learn()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end
    self.last_step = {s=state, a=action}

    -- require('fb.debugger').enter()
    self.memory:enqueue(frame)

    if not testing and self.target_q then
        self:update_to_target()
    end

    if terminal then
        self.memory:reset()
    end
    return action
end

function agent:learn()
    self:update_lr()

    local s, a, r, s2, t = self.buffer:sample(self.minibatch_size)
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

    self.target_network:evaluate()
    local q2_max = self.target_network:forward(s2):float():clone():max(2)

    -- y = r + (1-t) * gamma * maxQ'
    y = torch.add(r, q2_max:mul(self.discount):cmul(y))

    self.network:training()
    local q_all = self.network:forward(s):float():clone()
    local q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end

    -- dQ = r + (1-t) * gamma * Q' - Q
    local delta = torch.add(y, q:mul(-1))

    self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
            self.stat_eps*delta:clone():float():abs():mean()

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    self:update_parameter(s, targets)
end

function agent:update_to_target()
    if self.smooth_target_q then
        self.target_w:mul(1 - self.target_q_eps)
        self.target_w:add(self.target_q_eps, self.w)
    else
        if self.numSteps % self.target_q == 1 then
            self.target_network = self.network:clone()
        end
    end
end

function agent:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
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

    self.network:evaluate()
    local q = self.network:forward(state):float():squeeze()
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq
    self.v_avg = (1-self.stat_eps)*self.v_avg + self.stat_eps*maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end

function agent:update_parameter(s, targets)
    if self:use_gpu() then
        targets = targets:cuda()
    end

    -- zero gradients
    self.dw:zero()

    -- compute gradients
    self.network:backward(s, targets)

    -- gradient clipping
    local grad_norm = self.dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.dw:mul(self.clip_grad / grad_norm)
    end

    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
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

function agent:create_network()
    return g_create_network(self)
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
