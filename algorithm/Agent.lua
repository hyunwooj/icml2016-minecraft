if not dqn then
    require 'util.initenv'
end
require 'model.init'
require 'algorithm.Memory'


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
    self.qu_name       = args.network
    self.mu_name       = args.network
    self.name = self.qu_name

    self.qu = self:create_qu()
    self.mu = self:create_mu()

    if self:use_gpu() then
        self.qu:cuda()
        self.mu:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.qu:float()
        self.mu:float()
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

    self.qu_w, self.qu_dw = self.qu:getParameters()
    self.qu_dw:zero()

    self.deltas = self.qu_dw:clone():fill(0)
    self.tmp= self.qu_dw:clone():fill(0)
    self.g  = self.qu_dw:clone():fill(0)
    self.g2 = self.qu_dw:clone():fill(0)

    if self.target_q then
        self.target_qu = self.qu:clone()
        self.target_qu_w = self.target_qu:getParameters()
    end
end

function agent:perceive(reward, frame, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local frame = self:preprocess(frame):float()
    local state = self.memory:get(frame)

    if self.last_step then
        local s = self.last_step.s
        local a = self.last_step.a
        local r = reward
        local s2 = state
        local t = terminal
        self.buffer:add(s, a, r, s2, t)
    end

    input = nn.utils.addSingletonDimension(state, 1)
    if self:use_gpu() then
        input = input:cuda()
    end
    self.mu:evaluate()
    local action_probs = self.mu:forward(input, mem)
    -- TODO: Add random noise to explore
    local _, action = torch.max(action_probs:squeeze(), 1)
    action = action:totable()[1]

    self.last_step = {s=state, a=action}

    mem = self.memory:replace(frame, action)
    return action
end

function agent:create_qu()
    print('Creating Agent Network from ' .. self.qu_name)
    local net_args = {
        name           = self.qu_name,
        hist_len       = self.hist_len or 10,
        n_actions      = self.n_actions or 6,
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

function agent:create_mu()
    print('Creating Agent Network from ' .. self.mu_name)
    local net_args = {
        name           = self.mu_name,
        hist_len       = self.hist_len or 10,
        n_actions      = self.n_actions or 6,
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
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
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
