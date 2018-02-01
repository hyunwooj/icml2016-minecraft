require 'model.net'

local MQN, parent = torch.class('HybridMQN', 'Net')

function MQN:build_init_states(args)
    local state = {}
    table.insert(state, torch.Tensor(1, args.lstm_dim)) -- c
    table.insert(state, torch.Tensor(1, args.lstm_dim)) -- h
    table.insert(state, torch.Tensor(1, args.edim)) -- o
    return state
end

function MQN:build_model(args)
    local input = {}
    local init_states = {}

    local x = nn.Identity()()
    table.insert(input, x)

    local t = nn.Identity()()
    table.insert(input, t)

    local recall = nn.Identity()()
    table.insert(input, recall)

    for i=1, #self.init_states do
        local state = nn.Identity()()
        table.insert(input, state)
        table.insert(init_states, state)
    end

    local T = args.lt_mem_size + args.st_mem_size
    local edim = args.n_hid_enc

    local cnn_features = self:build_cnn(args, x)

    local mem_cnn_features = nn.Narrow(2, 1, T)(cnn_features)
    local mem_flat = nn.View(-1):setNumInputDims(1)(mem_cnn_features)

    local mem_key = nn.Linear(args.conv_dim, edim)(mem_flat)
    local mem_val = nn.Linear(args.conv_dim, edim)(mem_flat)
    mem_key = nn.View(-1, T, edim):setNumInputDims(2)(mem_key)
    mem_val = nn.View(-1, T, edim):setNumInputDims(2)(mem_val)

    -- Short-term
    local ctx, st_out, st_dbg = self:build_short(args, cnn_features, mem_key, mem_val,
                                                 init_states)

    -- Long-term
    local lt_out, lt_dbg = self:build_long(args, cnn_features, ctx, mem_key, mem_val,
                                           t, recall)

    local st_atten = st_dbg.atten
    local lt_atten = lt_dbg.atten
    local reten = lt_dbg.reten
    local stren = lt_dbg.stren
    local sigma = lt_dbg.sigma
    local comps = lt_dbg.comps

    local mem_q = self:build_memory_q(args, ctx, lt_out)
    local beh_q = self:build_behavior_q(args, ctx, st_out, lt_out)
    return nn.gModule(input, {mem_q, beh_q,
                              lt_atten, reten, stren, sigma, comps,
                              st_atten})
end

function MQN:build_behavior_q(args, ctx, st_out, lt_out)
    local edim = args.n_hid_enc
    local out = nn.JoinTable(2)({st_out, lt_out})
    out = nn.Linear(2*edim, edim)(out)

    local C = args.Linear(edim, edim)(ctx)
    local D = nn.CAddTable()({C, out})
    local hid_out
    if args.lindim == args.edim then
        hid_out = D
    elseif args.lindim == 0 then
        hid_out = nn.ReLU()(D)
    else
        local F = nn.Narrow(2, 1, args.lindim)(D)
        local G = nn.Narrow(2, 1+args.lindim, edim-args.lindim)(D)
        local K = nn.ReLU()(G)
        hid_out = nn.JoinTable(2)({F,K})
    end
    local out = nn.View(-1):setNumInputDims(1)(hid_out)
    local beh_q = nn.Linear(edim, args.n_actions)(out)
    return beh_q
end

function MQN:build_memory_q(args, ctx, out)
    local edim = args.n_hid_enc

    local C = args.Linear(edim, edim)(ctx)
    local D = nn.CAddTable()({C, out})
    local hid_out
    if args.lindim == args.edim then
        hid_out = D
    elseif args.lindim == 0 then
        hid_out = nn.ReLU()(D)
    else
        local F = nn.Narrow(2, 1, args.lindim)(D)
        local G = nn.Narrow(2, 1+args.lindim, edim-args.lindim)(D)
        local K = nn.ReLU()(G)
        hid_out = nn.JoinTable(2)({F,K})
    end
    local out = nn.View(-1):setNumInputDims(1)(hid_out)
    local mem_q = nn.Linear(edim, args.lt_mem_size)(out)
    return mem_q
end

function MQN:build_short(args, cnn_features, mem_key, mem_val, init_states)
    local c0, h0, o0 = unpack(init_states)

    local T = args.lt_mem_size + args.st_mem_size
    local T_s = args.st_mem_size
    local T_l = args.lt_mem_size

    local st_cnn_features = nn.Narrow(2, T_l+1, T_s+1)(cnn_features)
    local st_mem_key = nn.Narrow(2, 1+T_l, T_s)(mem_key)
    local st_mem_val = nn.Narrow(2, 1+T_l, T_s)(mem_val)
    local ctx, st_out, atten = self:build_st_retrieval(args, st_mem_key, st_mem_val,
                                                       st_cnn_features,
                                                       c0, h0, o0)
    local dbg = {atten = atten}
    return ctx, st_out, dbg
end

function MQN:build_st_retrieval(args, st_mem_key, st_mem_val,
                                cnn_features, c0, h0, o0)
    local edim = args.edim
    local T = args.st_mem_size+1

    local x_flat = nn.View(-1):setNumInputDims(1)(cnn_features)
    local x_gate_flat = nn.Linear(args.conv_dim, 4*edim)(x_flat)
    local x_gate = nn.View(-1, T, 4*edim):setNumInputDims(2)(x_gate_flat)
    local x_gates = {nn.SplitTable(1, 2)(x_gate):split(T)}

    local atten
    local c = {c0}
    local h = {h0}
    local o = {o0}
    for t = 1, T do
        local input = nn.Reshape(4*edim)(x_gates[t])
        local prev_c = c[t]
        local prev_h = h[t]
        local prev_o = o[t]

        local prev_hr = nn.JoinTable(2)({prev_h, prev_o})
        local h2h = nn.Linear(2*edim, 4*edim)(prev_hr)
        local all_input_sums = nn.CAddTable()({input, h2h})
        local reshaped = nn.View(-1, 4, edim):setNumInputDims(2)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        n1 = nn.Contiguous()(n1)
        n2 = nn.Contiguous()(n2)
        n3 = nn.Contiguous()(n3)
        n4 = nn.Contiguous()(n4)

        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local in_transform = nn.Tanh()(n4)
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
          })
        local out_gate = nn.Sigmoid()(n3)
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

        c[t+1] = next_c
        h[t+1] = next_h

        -- Retrieve a memory block
        if t == 1 then
            o[t+1] = o0
        else
            local st_mem_key_t = nn.Narrow(2, 1, t-1)(st_mem_key)
            local st_mem_val_t = nn.Narrow(2, 1, t-1)(st_mem_val)
            local context = nn.View(1, -1):setNumInputDims(1)(next_h)
            local MM_key = nn.MM(false, true)
            local key_out = MM_key({context, st_mem_key_t})
            local key_out2dim = nn.View(-1):setNumInputDims(2)(key_out)
            atten = nn.SoftMax()(key_out2dim)
            local probs3dim = nn.View(1, -1):setNumInputDims(1)(atten)
            local MM_val = nn.MM(false, false)
            local val_out = MM_val({probs3dim, st_mem_val_t})
            local next_o = nn.View(-1):setNumInputDims(1)(val_out)
            if args.gpu and args.gpu > 0 then
                MM_key = MM_key:cuda()
                MM_val = MM_val:cuda()
            end
            o[t+1] = next_o
        end

        self:share_module("h2h", h2h)
    end

    return h[T+1], o[T+1], atten
end

function MQN:build_long(args, cnn_features, ctx, mem_key, mem_val, t, recall)
    local T_l = args.lt_mem_size
    local lt_mem_cnn_features = nn.Narrow(2, 1, T_l)(cnn_features)
    local lt_mem_flat = nn.View(-1):setNumInputDims(1)(lt_mem_cnn_features)

    local reten, stren, sigma, comps = self:build_retention(args, lt_mem_flat, t, recall)

    local lt_mem_key = nn.Narrow(2, 1, T_l)(mem_key)
    local lt_mem_val = nn.Narrow(2, 1, T_l)(mem_val)
    local _, lt_out, atten = self:build_lt_retrieval(args, lt_mem_key, lt_mem_val,
                                                     ctx, reten)
    lt_out = nn.View(-1):setNumInputDims(1)(lt_out)
    local lt_dbg = {
            atten = atten,
            reten = reten,
            stren = stren,
            sigma = sigma,
            comps = comps,
        }
    return lt_out, lt_dbg
end

function MQN:build_retention(args, history_flat, t, recall)
    local T = args.lt_mem_size

    local history_t = nn.Narrow(2, 1, T)(t)
    local current_t = nn.Narrow(2, T+1, 1)(t)
    current_t = nn.ExpandAs()({history_t, current_t})

    local t = nn.CSubTable()({history_t, current_t})

    local sigma = nn.Linear(args.conv_dim, 1)(history_flat)
    sigma = nn.View(-1, T, 1):setNumInputDims(2)(sigma)
    sigma = nn.Sigmoid()(sigma)
    sigma = nn.MulConstant(0.1)(sigma)

    -- SoftPlus
    -- local S = nn.Linear(args.conv_dim, 1)(history_flat)
    -- S = nn.View(-1, T, 1):setNumInputDims(2)(S)
    -- S = nn.AddConstant(1)(nn.SoftPlus()(S))

    -- local reten = nn.Exp()(nn.CDivTable()({t, S}))

    -- ReLU
    local S = nn.Linear(args.conv_dim, 1)(history_flat)
    S = nn.View(-1, T, 1):setNumInputDims(2)(S)
    S = nn.AddConstant(1)(nn.ReLU()(S))

    local compound = nn.Log()(nn.AddConstant(1)(sigma))
    compound = nn.CMulTable()({recall, compound})
    compound = nn.Exp()(compound)
    compound = nn.CMulTable()({S, compound})

    local reten = nn.Exp()(nn.CDivTable()({t, compound}))

    -- Sigmoid 1/S
    -- local S_inv = nn.Linear(args.conv_dim, 1)(history_flat)
    -- S_inv = nn.View(-1, T, 1):setNumInputDims(2)(S_inv)
    -- S_inv = nn.Sigmoid(S_inv)
    -- local S = nn.Power(-1)(S_inv)

    -- local reten = nn.Exp()(nn.CMulTable({t, S_inv}))

    return reten, S, sigma, compound
end

function MQN:build_lt_retrieval(args, mem_key, mem_val, context, reten)
    local T = args.lt_mem_size
    local edim = args.edim
    local context = nn.View(1, -1):setNumInputDims(1)(context)
    local MM_key = nn.MM(false, true)
    local key_out = MM_key({context, mem_key})
    local key_out2dim = nn.View(-1):setNumInputDims(2)(key_out)

    -- Original
    -- local P = nn.SparseMax()(key_out2dim)
    -- local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)

    -- Multiplication
    local atten = nn.SparseMax()(key_out2dim)
    atten = nn.CMulTable()({atten, reten})
    atten = nn.Normalize(1)(atten)

    -- -- B x T x edim
    -- local expanded_hid = nn.ExpandAs()({mem_key, context})
    -- -- B x T x 1
    -- local unsqueezed_reten = nn.View(-1, T, 1)(reten)

    -- local concated = nn.JoinTable(3)({expanded_hid, mem_key, unsqueezed_reten})
    -- local atten_input = nn.View(-1, edim + edim + 1)(concated)
    -- local atten_hidden = nn.ReLU()(nn.Linear(edim + edim + 1, edim)(atten_input))
    -- local atten_out = nn.Linear(edim, 1)(atten_hidden)
    -- local atten = nn.SparseMax()(nn.View(-1, T)(atten_out))

    local probs3dim = nn.View(1, -1):setNumInputDims(1)(atten)

    local MM_val = nn.MM(false, false)
    local o = MM_val({probs3dim, mem_val})
    if args.gpu and args.gpu > 0 then
        MM_key = MM_key:cuda()
        MM_val = MM_val:cuda()
    end
    -- return context, o, P
    return context, o, atten
end

function MQN:build_cnn(args, input)
    local reshape_input = nn.View(-1, unpack(args.image_dims))(input)
    local conv, conv_nl = {}, {}
    local prev_dim = args.ncols
    local prev_input = reshape_input
    for i=1,#args.n_units do
        conv[i] = nn.SpatialConvolution(prev_dim, args.n_units[i],
                            args.filter_size[i], args.filter_size[i],
                            args.filter_stride[i], args.filter_stride[i],
                            args.pad[i], args.pad[i])(prev_input)
        conv_nl[i] = nn.ReLU()(conv[i])
        prev_dim = args.n_units[i]
        prev_input = conv_nl[i]
    end

    local conv_flat = nn.View(-1):setNumInputDims(3)(conv_nl[#args.n_units])
    return nn.View(-1, args.mem_size+1, args.conv_dim):setNumInputDims(2)(conv_flat)
end
