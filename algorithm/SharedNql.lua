--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'util.initenv'
end
require 'model.init'
require 'algorithm.Memory'
require 'algorithm.ReplayBuffer'


local nql = torch.class('dqn.SharedNql')

function nql:__init(args)
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
    self.bestq          = 0
    self.gpu            = args.gpu
    self.ncols          = args.ncols or 3  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 32, 32}
    self.image_dims     = args.image_dims or {self.ncols, 32, 32}
    self.bufferSize       = args.bufferSize or 512
    self.smooth_target_q  = args.smooth_target_q or true
    self.target_q_eps     = args.target_q_eps or 1e-3
    self.clip_grad        = args.clip_grad or 20
    self.transition_params = args.transition_params or {}
    self.network          = args.network or self:createNetwork()

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    if args.network:sub(-3) == '.t7' then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file " .. self.network)
        end
        if self.best == 1 and exp.best_model then
            self.network = exp.best_model[1]
        elseif exp.model then
            self.network = exp.model
        else
            self.network = exp
        end
        self.hist_len = self.network.args.hist_len
        self.input_dims = self.network.args.input_dims
        print("Load Network from " .. args.network)
    else
        print('Creating Agent Network from ' .. self.network)
        self.name = self.network
        self.network = g_create_network(self)
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    -- self.transitions = dqn.TransitionTable(transition_args)
    self.memory = self:create_memory()
    self.buffer = self:create_replay_buffer()

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
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

function nql:preprocess(rawstate)
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


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    local beh_a = a % (self.n_actions+1)
    local mem_a = a / (self.n_actions+1)

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
    target_q_net:evaluate()
    local output2 = target_q_net:forward(s2)
    mem_q2_max = output2[1]:float():max(2)
    beh_q2_max = output2[2]:float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    mem_q2 = mem_q2_max:clone():mul(self.discount):cmul(term)
    beh_q2 = beh_q2_max:clone():mul(self.discount):cmul(term)

    mem_delta = r:clone():float()
    beh_delta = r:clone():float()

    if self.rescale_r then
        mem_delta:div(self.r_max)
        beh_delta:div(self.r_max)
    end
    mem_delta:add(mem_q2)
    beh_delta:add(beh_q2)

    -- q = Q(s,a)
    self.network:training()
    local output = self.network:forward(s)
    local mem_q_all = output[1]:float()
    local beh_q_all = output[2]:float()
    mem_q = torch.FloatTensor(mem_q_all:size(1))
    beh_q = torch.FloatTensor(beh_q_all:size(1))
    for i=1,mem_q_all:size(1) do
        mem_q[i] = mem_q_all[i][mem_a[i]]
    end
    for i=1,beh_q_all:size(1) do
        beh_q[i] = beh_q_all[i][beh_a[i]]
    end
    mem_delta:add(-1, mem_q)
    beh_delta:add(-1, beh_q)
    self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
            self.stat_eps*beh_delta:clone():float():abs():mean()

    if self.clip_delta then
        mem_delta[mem_delta:ge(self.clip_delta)] = self.clip_delta
        mem_delta[mem_delta:le(-self.clip_delta)] = -self.clip_delta
        beh_delta[beh_delta:ge(self.clip_delta)] = self.clip_delta
        beh_delta[beh_delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local mem_targets = torch.zeros(self.minibatch_size, self.hist_len-1):float()
    local beh_targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,mem_a:size(1)) do
        mem_targets[i][mem_a[i]] = mem_delta[i]
    end
    for i=1,math.min(self.minibatch_size,beh_a:size(1)) do
        beh_targets[i][beh_a[i]] = beh_delta[i]
    end

    if self.gpu >= 0 then mem_targets = mem_targets:cuda() end
    if self.gpu >= 0 then beh_targets = beh_targets:cuda() end

    local targets = {
        mem=mem_targets,
        beh=beh_targets,
    }
    local delta = {
        mem=mem_delta,
        beh=beh_delta,
    }
    local q2_max = {
        mem=mem_q2_max,
        beh=beh_q2_max,
    }
    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    -- assert(self.transitions:size() > self.minibatch_size)

    -- local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)
    local s, a, r, s2, term = self.buffer:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, {targets.mem, targets.beh})

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- clip gradients
    local grad_norm = self.dw:norm()
    if self.clip_grad > 0 and grad_norm > self.clip_grad then
        self.dw:mul(self.clip_grad / grad_norm)
    end

    -- use gradients
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

-- Store a new transition to the replay memory
-- Sample a mini-batch of transition and perform gradient descent
-- Choose one of the actions
function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate)
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    --self.transitions:add_recent_state(state, terminal)

    --local currentFullState = self.transitions:get_recent()

    ----Store transition s, a, r, s'
    --if self.lastState and not testing then
    --    self.transitions:add(self.lastState, self.lastAction, reward,
    --                         self.lastTerminal, priority)
    --end

    --curState= self.transitions:get_recent()
    --curState = curState:resize(1, unpack(self.input_dims))
    curState = self.memory:concat(state)
    curState = curState:float():div(255)
    if self:use_gpu() then
        curState = curState:cuda()
    end

    if self.last_step and not testing then
        local s = self.last_step.s
        local a = self.last_step.a
        local r = reward
        local s2 = curState
        local t = terminal
        self.buffer:add(s, a, r, s2, t)
    end

    -- Select action
    -- local actionIndex = 1
    local beh_action = 1
    local mem_action = 1
    if not terminal then
        -- actionIndex = self:eGreedy(curState, testing_ep, testing)
        mem_action, beh_action = self:eGreedy(curState, testing_ep, testing)
    end

    -- self.transitions:add_recent_action(actionIndex)
    -- self.memory:enqueue(state)
    self.memory:replace(state, mem_action)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    -- self.lastState = state:clone()
    -- self.lastAction = actionIndex
    -- self.lastTerminal = terminal
    local action = (mem_action*(self.n_actions+1)) + beh_action
    self.last_step = {s=curState, a=action}

    if not testing and self.target_q then
        if self.smooth_target_q then
            self.target_w:mul(1 - self.target_q_eps)
            self.target_w:add(self.target_q_eps, self.w)
        else
            if self.numSteps % self.target_q == 1 then
                self.target_network = self.network:clone()
            end
        end
    end

    if not terminal then
        return beh_action
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep, testing)
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


function nql:greedy(state)
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

    self.network:evaluate()
    local output = self.network:forward(state)
    _, mem_action = pick_best(output[1]:totable()[1])
    max_q, beh_action = pick_best(output[2]:totable()[1])
    self.v_avg = (1-self.stat_eps)*self.v_avg + self.stat_eps*max_q
    return mem_action, beh_action
end

function nql:report()
    print(get_weight_norms(self.network.net))
    print(get_grad_norms(self.network.net))
    print("Grad Norm: " .. tostring(self.dw:norm()))
end

function nql:create_memory()
    local mem_args = {
        mem_size = self.hist_len-1
    }
    return Memory(mem_args)
end

function nql:create_replay_buffer()
    local buffer_args = {
        frame_dim = self.state_dim,
        mem_size = self.hist_len,
        max_size = self.replay_memory,
        batch_size = self.minibatch_size,
        nonTermProb = self.nonTermProb,
        gpu = self.gpu,
    }
    return ReplayBuffer(buffer_args)
end

function nql:use_gpu()
    return self.gpu and self.gpu >= 0
end
