if not dqn then
    require 'util.initenv'
end
require 'model.init'
require 'algorithm.Memory'
require 'algorithm.ReplayBuffer'


local agent = torch.class('dqn.Agent')


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
    self.target_eps     = args.target_q_eps or 1e-3
    self.clip_grad        = args.clip_grad or 20
    self.transition_params = args.transition_params or {}
    self.critic_name       = args.network
    self.actor_name       = args.network
    self.name = self.critic_name

    self.critic = self:create_critic()
    self.actor = self:create_actor()
    self.log_softmax = nn.LogSoftMax()
    self.softmax = nn.SoftMax()

    if self:use_gpu() then
        self.critic:cuda()
        self.actor:cuda()
        self.log_softmax:cuda()
        self.softmax:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.critic:float()
        self.actor:float()
        self.log_softmax:float()
        self.softmax:float()
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

    -- critic
    self.critic_w, self.critic_dw = self.critic:getParameters()
    self.critic_dw:zero()

    self.critic_deltas = self.critic_dw:clone():fill(0)
    self.critic_tmp    = self.critic_dw:clone():fill(0)
    self.critic_g      = self.critic_dw:clone():fill(0)
    self.critic_g2     = self.critic_dw:clone():fill(0)

    if self.target_q then
        self.target_critic = self.critic:clone()
        self.target_critic_w = self.target_critic:getParameters()
    end

    -- actor
    self.actor_w, self.actor_dw = self.actor:getParameters()
    self.actor_dw:zero()

    self.actor_deltas = self.actor_dw:clone():fill(0)
    self.actor_tmp    = self.actor_dw:clone():fill(0)
    self.actor_g      = self.actor_dw:clone():fill(0)
    self.actor_g2     = self.actor_dw:clone():fill(0)

    if self.target_q then
        self.target_actor = self.actor:clone()
        self.target_actor_w = self.target_actor:getParameters()
    end
end

function agent:perceive(reward, frame, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local frame = self:preprocess(frame)
    local state = self.memory:get(frame)

    reward = self:process_reward(reward)

    if self.last_step and not testing then
        local s = self.last_step.s
        local a = self.last_step.a
        local r = reward
        local s2 = state
        local t = terminal
        self.buffer:add(s, a, r, s2, t)
    end

    input = nn.utils.addSingletonDimension(state, 1):float():clone():div(255)
    if self:use_gpu() then
        input = input:cuda()
    end
    self.actor:evaluate()
    local action_logits = self.actor:forward(input, mem)
    local action_probs = self.softmax:forward(action_logits)
    local action, mem_idx, beh_idx = self:sample_action(action_probs)

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

    self.memory:replace(frame, mem_idx)

    if not testing and self.target_q then
        self:update_to_target()
    end

    if beh_idx < 1 or beh_idx > 6 then
        require('fb.debugger').enter()
    end
    return beh_idx
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

    self.target_critic:evaluate()
    local v2 = self.target_critic:forward(s2):float():clone()

    -- y = r + (1-t) * gamma * V'
    y = torch.add(r, v2:mul(self.discount):cmul(y))

    self.critic:training()
    local v = self.critic:forward(s):float()

    -- A = r + (1-t) * gamma * V' - V
    local delta = torch.add(y, v:mul(-1))
    local adv = delta:clone()

    self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
            self.stat_eps*delta:clone():float():abs():mean()

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    self:update_critic(s, delta)

    self.critic:training()
    local action_logits = self.actor:forward(s)
    delta = self.log_softmax:forward(action_logits):float():clone()
    adv = nn.utils.addSingletonDimension(adv, 2):expandAs(delta)
    delta:cmul(adv)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    self:update_actor(s, delta)
end

function agent:update_to_target()
    -- actor
    if self.smooth_target_q then
        self.target_actor_w:mul(1 - self.target_eps)
        self.target_actor_w:add(self.target_eps, self.actor_w)
    else
        if self.numSteps % self.target_q == 1 then
            self.target_actor = self.actor:clone()
        end
    end

    -- critic
    if self.smooth_target_q then
        self.target_critic_w:mul(1 - self.target_eps)
        self.target_critic_w:add(self.target_eps, self.critic_w)
    else
        if self.numSteps % self.target_q == 1 then
            self.target_critic = self.critic:clone()
        end
    end
end

function agent:sample_action(probs)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    local num_actions = probs:size():totable()[2]

    local action
    if torch.uniform() < self.ep then
        -- action = torch.random(1, num_actions):totable()[1]
         action = weighted_choice(probs:squeeze():totable())
    else
        _, action = torch.max(probs:squeeze(), 1)
        action = action:totable()[1]
    end

    mem_idx = (action-1) % (self.hist_len-1) + 1
    beh_idx = math.floor((action-1) / (self.hist_len-1)) + 1
    return action, mem_idx, beh_idx
end

function agent:update_critic(s, delta)
    if self:use_gpu() then
        delta = delta:cuda()
    end

    -- zero gradients
    self.critic_dw:zero()

    -- compute gradients
    delta = nn.utils.addSingletonDimension(delta, 2)
    self.critic:backward(s, delta)

    -- gradient clipping
    local grad_norm = self.critic_dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.critic_dw:mul(self.clip_grad / grad_norm)
    end

    self.critic_g:mul(0.95):add(0.05, self.critic_dw)
    self.critic_tmp:cmul(self.critic_dw, self.critic_dw)
    self.critic_g2:mul(0.95):add(0.05, self.critic_tmp)
    self.critic_tmp:cmul(self.critic_g, self.critic_g)
    self.critic_tmp:mul(-1)
    self.critic_tmp:add(self.critic_g2)
    self.critic_tmp:add(0.01)
    self.critic_tmp:sqrt()

    -- accumulate update
    self.critic_deltas:mul(0):addcdiv(self.lr, self.critic_dw, self.critic_tmp)
    self.critic_w:add(self.critic_deltas)
end

function agent:update_actor(s, delta)
    if self:use_gpu() then
        delta = delta:cuda()
    end

    -- zero gradients
    self.actor_dw:zero()

    -- compute gradients
    self.actor:backward(s, delta)

    -- gradient clipping
    local grad_norm = self.actor_dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.actor_dw:mul(self.clip_grad / grad_norm)
    end

    self.actor_g:mul(0.95):add(0.05, self.actor_dw)
    self.actor_tmp:cmul(self.actor_dw, self.actor_dw)
    self.actor_g2:mul(0.95):add(0.05, self.actor_tmp)
    self.actor_tmp:cmul(self.actor_g, self.actor_g)
    self.actor_tmp:mul(-1)
    self.actor_tmp:add(self.actor_g2)
    self.actor_tmp:add(0.01)
    self.actor_tmp:sqrt()

    -- accumulate update
    self.actor_deltas:mul(0):addcdiv(self.lr, self.actor_dw, self.actor_tmp)
    self.actor_w:add(self.actor_deltas)
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

function agent:create_critic()
    print('Creating Agent Network from ' .. self.critic_name)
    local net_args = {
        name           = self.critic_name,
        hist_len       = self.hist_len or 10,
        n_actions      = 1,
        ncols          = self.ncols or 3,
        image_dims     = self.image_dims or {3, 32, 32},
        n_units        = self.n_units or {32, 64},
        filter_size    = self.filter_size or {4, 4},
        filter_stride  = self.filter_stride or {2, 2},
        pad            = self.pad or {1, 1},
        n_hid_enc      = self.n_hid_enc or 256,
        edim           = self.edim or 256,
        lstm_dim       = self.edim or 256,
        gpu            = self.gpu or -1,
        Linear         = nn.LinearNB,
    }
    net_args.memsize    = self.memsize or (net_args.hist_len - 1)
    net_args.lindim     = self.lindim or net_args.edim / 2
    net_args.conv_dim   = net_args.n_units[#net_args.n_units] * 8 * 8
    net_args.input_dims = self.input_dims or
                          {net_args.hist_len * net_args.ncols, 32, 32}
    return g_create_network(net_args)
end

function agent:create_actor()
    print('Creating Agent Network from ' .. self.actor_name)
    local net_args = {
        name           = self.actor_name,
        hist_len       = self.hist_len or 10,
        n_actions      = self.n_actions * (self.hist_len - 1),
        ncols          = self.ncols or 3,
        image_dims     = self.image_dims or {3, 32, 32},
        n_units        = self.n_units or {32, 64},
        filter_size    = self.filter_size or {4, 4},
        filter_stride  = self.filter_stride or {2, 2},
        pad            = self.pad or {1, 1},
        n_hid_enc      = self.n_hid_enc or 256,
        edim           = self.edim or 256,
        lstm_dim       = self.edim or 256,
        gpu            = self.gpu or -1,
        Linear         = nn.LinearNB,
    }
    net_args.memsize    = self.memsize or (net_args.hist_len - 1)
    net_args.lindim     = self.lindim or net_args.edim / 2
    net_args.conv_dim   = net_args.n_units[#net_args.n_units] * 8 * 8
    net_args.input_dims = self.input_dims or
                          {net_args.hist_len * net_args.ncols, 32, 32}
    return g_create_network(net_args)
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

function nql:report()
    print('Actor')
    print(get_weight_norms(self.actor.net))
    print(get_grad_norms(self.actor.net))
    print("Grad Norm: " .. tostring(self.actor_dw:norm()))

    print('')

    print('Critic')
    print(get_weight_norms(self.critic.net))
    print(get_grad_norms(self.critic.net))
    print("Grad Norm: " .. tostring(self.critic_dw:norm()))
end
