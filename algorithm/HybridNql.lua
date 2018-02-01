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
require 'util/lua_utils'


local nql = torch.class('dqn.HybridNql')

function nql:__init(args)
    self.state_dim  = args.state_dim
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose

    --- epsilon annealing
    self.ep_start   = 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = 0.1
    self.ep_endt    = 1000000

    self.lr             = args.lr
    self.minibatch_size = args.minibatch_size or 1

    --- Q-learning parameters
    self.discount       = 0.99 --Discount factor.
    self.update_freq    = 4
    -- Number of points to replay per learning step.
    self.n_replay       = 1
    -- Number of steps after which learning starts.
    self.learn_start    = 50000
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory
    self.st_mem_size    = args.st_mem_size
    self.lt_mem_size    = args.lt_mem_size
    self.mem_size       = self.st_mem_size + self.lt_mem_size
    self.rescale_r      = 1
    self.max_reward     = 1
    self.min_reward     = -1
    self.clip_delta     = 1
    self.target_q       = true
    self.bestq          = 0
    self.gpu            = args.gpu
    self.ncols          = 3  -- number of color channels in input
    self.input_dims     = {(self.mem_size+1)*self.ncols, 32, 32}
    self.image_dims     = {self.ncols, 32, 32}
    self.bufferSize       = 512
    self.smooth_target_q  = true
    self.target_q_eps     = 1e-3
    self.clip_grad        = 20
    self.network          = args.network

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
        self.mem_size = self.network.args.mem_size
        self.st_mem_size = self.network.args.st_mem_size
        self.lt_mem_size = self.network.args.lt_mem_size
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

    self.memory = self:create_memory()
    self.buffer = self:create_replay_buffer()
    self.reten_history = {}

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.mem_v_avg = 0 -- V running average.
    self.v_avg = 0 -- V running average.
    self.mem_tderr_avg = 0 -- TD error running average.
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
    local times, times2
    local noise, noise2
    local recall, recall2
    local q, q2, q2_max

    s = args.s.frames
    a = args.a
    mem_r = args.r.mem
    r = args.r.beh
    s2 = args.s2.frames
    term = args.term
    times = args.s.times
    times2 = args.s2.times
    recall = args.s.recall
    recall2 = args.s2.recall

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
    local output2 = target_q_net:forward(s2, times2, recall2)
    mem_q2_max = output2[1]:float():max(2)
    beh_q2_max = output2[2]:float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    mem_q2 = mem_q2_max:clone():mul(self.discount):cmul(term)
    beh_q2 = beh_q2_max:clone():mul(self.discount):cmul(term)

    mem_delta = mem_r:clone():float()
    beh_delta = r:clone():float()

    if self.rescale_r then
        mem_delta:div(self.r_max)
        beh_delta:div(self.r_max)
    end
    mem_delta:add(mem_q2)
    beh_delta:add(beh_q2)

    -- q = Q(s,a)
    self.network:training()
    local output = self.network:forward(s, times, recall)
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
    self.mem_tderr_avg = (1-self.stat_eps)*self.mem_tderr_avg +
            self.stat_eps*mem_delta:clone():float():abs():mean()
    self.tderr_avg = (1-self.stat_eps)*self.tderr_avg +
            self.stat_eps*beh_delta:clone():float():abs():mean()

    if self.clip_delta then
        mem_delta[mem_delta:ge(self.clip_delta)] = self.clip_delta
        mem_delta[mem_delta:le(-self.clip_delta)] = -self.clip_delta
        beh_delta[beh_delta:ge(self.clip_delta)] = self.clip_delta
        beh_delta[beh_delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local mem_targets = torch.zeros(self.minibatch_size, self.lt_mem_size):float()
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
    local zero_lt = targets.mem.new(self.minibatch_size, self.lt_mem_size):zero()
    local zero_st = targets.mem.new(self.minibatch_size, self.st_mem_size):zero()
    self.network:backward({s.frames, s.times},
                          {targets.mem, targets.beh,
                           zero_lt, zero_lt, zero_lt, zero_lt, zero_lt, zero_st})

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)

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
    local time_step = torch.FloatTensor(1):fill(rawstate.time)
    local rawstate = rawstate.screen
    -- Preprocess state (will be set to nil if terminal)
    local frame = self:preprocess(rawstate)
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

    --self.transitions:add_recent_state(frame, terminal)

    --local currentFullState = self.transitions:get_recent()

    ----Store transition s, a, r, s'
    --if self.lastState and not testing then
    --    self.transitions:add(self.lastState, self.lastAction, reward,
    --                         self.lastTerminal, priority)
    --end

    --curState= self.transitions:get_recent()
    --curState = curState:resize(1, unpack(self.input_dims))
    curState, times, recalls = self.memory:concat({frame=frame, time=time_step})
    curState = curState:float():div(255)
    curState = unsqueeze(curState, 1)
    times = unsqueeze(times, 1)
    recalls = unsqueeze(recalls, 1)
    if self:use_gpu() then
        curState = curState:cuda()
        times = times:cuda()
        recalls = recalls:cuda()
    end
    local state = {frames=curState, time=times, recall=recalls}

    if self.last_step and not testing then
        local s = self.last_step.s
        local a = self.last_step.a

        local reten_avg_reward
        if #self.reten_history == 0 then
            reten_avg_reward = 0
        else
            reten_avg_reward = 1 - (sum(unpack(self.reten_history)) / #self.reten_history)
        end

        local r = {beh=reward,
                   -- mem=reward + (0.1 * reten_avg_reward)}
                   -- mem=reward + (0.1 * self.last_step.r.mem)}
                   mem=reward}
        local s2 = state
        local t = terminal
        self.buffer:add(s, a, r, s2, t)
    end

    -- Select action
    -- local actionIndex = 1
    local beh_action = 1
    local mem_action = 1
    local atten, reten, stren
    local sigma, comps
    local mem_q, beh_q
    local output
    if not terminal then
        output = self:eGreedy(state, testing_ep, testing)
        mem_action, beh_action = output[1], output[2]
        atten, reten, stren = output[3], output[4], output[5]
        sigma, comps = output[6], output[7]
        mem_q, beh_q = output[8], output[9]
    end

    if atten ~= nil then
        local idxs = {}
        local recalled = atten:gt(0.2):totable()[1]
        for idx, val in pairs(recalled) do
            if val == 1 then
                table.insert(idxs, idx)
            end
        end
        self.memory:recall(idxs, time_step)
    end
    local reten_diff_reward = 0
    if reten ~= nil then
        table.insert(self.reten_history, reten[1][mem_action])
        reten_diff_reward = reten - torch.mean(reten)
        reten_diff_reward = reten_diff_reward[1][mem_action]
    end

    self.memory:memorize(mem_action, {frame=frame, time=time_step})

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

    local action = (mem_action*(self.n_actions+1)) + beh_action
    self.last_step = {s=state, a=action, r={mem=reten_diff_reward}}

    if terminal then
        self.memory:reset()
        self.reten_history = {}
    end

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

    local dbg = {}
    if mem_q ~= nil then
        dbg.mem_q = mem_q[1]:clone():float()
        dbg.beh_q = beh_q[1]:clone():float()
    end

    local mem_dbg = nil
    if atten ~= nil then
        mem_dbg = {
            atten = atten[1]:clone():float(),
            reten = reten[1]:clone():float(),
            times = times[1]:clone():float(),
        }
        if stren then
            mem_dbg.stren = stren[1]:clone():float()
        end
        if sigma then
            mem_dbg.sigma = sigma[1]:clone():float()
        end
        if comps then
            mem_dbg.comps = comps[1]:clone():float()
        end
    end

    if not terminal then
        return beh_action, dbg, mem_dbg
    else
        return 0, dbg, mem_dbg
    end
end


function nql:eGreedy(state, testing_ep, testing)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- self.learn_start = 100
    -- self.ep = 0
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        mem_action = torch.random(1, self.lt_mem_size)
        beh_action = torch.random(1, self.n_actions)
        return {mem_action, beh_action}
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    local times = state.time
    local recall = state.recall
    local state = state.frames
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
    local output = self.network:forward(state, times, recall)
    mem_max_q, mem_action = pick_best(output[1]:totable()[1])
    max_q, beh_action = pick_best(output[2]:totable()[1])
    self.mem_v_avg = (1-self.stat_eps)*self.mem_v_avg + self.stat_eps*mem_max_q
    self.v_avg = (1-self.stat_eps)*self.v_avg + self.stat_eps*max_q

    local mem_q, beh_q = output[1], output[2]
    local atten, reten, stren = output[3], output[4], output[5]
    local sigma, comps = output[6], output[7]
    local st_atten = output[8]

    return {mem_action, beh_action, atten, reten, stren, sigma, comps, mem_q, beh_q}
end

function nql:report()
    print(get_weight_norms(self.network.net))
    print(get_grad_norms(self.network.net))
    print("Grad Norm: " .. tostring(self.dw:norm()))
end

function nql:create_memory()
    local mem_args = {
        st_mem_size = self.st_mem_size,
        lt_mem_size = self.lt_mem_size,
        image_dims = self.image_dims,
    }
    return Memory(mem_args)
end

function nql:create_replay_buffer()
    local buffer_args = {
        frame_dim = self.state_dim,
        mem_size = self.mem_size,
        lt_mem_size = self.lt_mem_size,
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
