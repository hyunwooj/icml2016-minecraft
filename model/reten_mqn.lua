require 'model.net'

local MQN, parent = torch.class('RetenMQN', 'Net')
function MQN:build_model(args)
    local input = {}
    local init_states = {}

    local x = nn.Identity()()
    table.insert(input, x)

    local t = nn.Identity()()
    table.insert(input, t)

    local recall = nn.Identity()()
    table.insert(input, recall)

    -- local hit = nn.Identity()()
    -- table.insert(input, hit)

    for i=1, #self.init_states do
        local state = nn.Identity()()
        table.insert(input, state)
        table.insert(init_states, state)
    end
    local T = args.hist_len
    local edim = args.n_hid_enc
    local cnn_features = self:build_cnn(args, x)
    local history = nn.Narrow(2, 1, T-1)(cnn_features)
    local history_flat = nn.View(-1):setNumInputDims(1)(history)

    local reten, stren, sigma, comps = self:build_retention(args, history_flat, t, recall)

    local key_blocks = nn.Linear(args.conv_dim, edim)(history_flat)
    local val_blocks = nn.Linear(args.conv_dim, edim)(history_flat)
    key_blocks = nn.View(-1, T-1, edim):setNumInputDims(2)(key_blocks)
    val_blocks = nn.View(-1, T-1, edim):setNumInputDims(2)(val_blocks)
    local c0, h0 = unpack(init_states)
    local hid, o, atten = self:build_retrieval(args, key_blocks, val_blocks,
                cnn_features, args.conv_dim, c0, h0, reten)
    local hid2dim = nn.View(-1):setNumInputDims(1)(hid)
    local C = args.Linear(edim, edim)(hid2dim)
    local D = nn.CAddTable()({C, o})
    local hid_out
    if args.lindim == args.edim then
        hid_out = D
    elseif args.lindim == 0 then
        hid_out = nn.ReLU()(D)
    else
        local F = nn.Narrow(2,1,args.lindim)(D)
        local G = nn.Narrow(2,1+args.lindim,edim-args.lindim)(D)
        local K = nn.ReLU()(G)
        hid_out = nn.JoinTable(2)({F,K})
    end
    local out = nn.View(-1):setNumInputDims(1)(hid_out)

    local mem_q = nn.Linear(args.n_hid_enc, args.hist_len-1)(out)
    local beh_q = nn.Linear(args.n_hid_enc, args.n_actions)(out)
    return nn.gModule(input, {mem_q, beh_q, atten, reten, stren, sigma, comps})
end

function MQN:build_retention(args, history_flat, t, recall)
    local T = args.hist_len

    local history_t = nn.Narrow(2, 1, T-1)(t)
    local current_t = nn.Narrow(2, T, 1)(t)
    current_t = nn.ExpandAs()({history_t, current_t})

    local t = nn.CSubTable()({history_t, current_t})

    local sigma = nn.Linear(args.conv_dim, 1)(history_flat)
    sigma = nn.View(-1, T-1, 1):setNumInputDims(2)(sigma)
    sigma = nn.Sigmoid()(sigma)
    sigma = nn.MulConstant(0.1)(sigma)

    -- SoftPlus
    -- local S = nn.Linear(args.conv_dim, 1)(history_flat)
    -- S = nn.View(-1, T-1, 1):setNumInputDims(2)(S)
    -- S = nn.AddConstant(1)(nn.SoftPlus()(S))

    -- local reten = nn.Exp()(nn.CDivTable()({t, S}))

    -- ReLU
    local S = nn.Linear(args.conv_dim, 1)(history_flat)
    S = nn.View(-1, T-1, 1):setNumInputDims(2)(S)
    S = nn.AddConstant(1)(nn.ReLU()(S))

    local compound = nn.Log()(nn.AddConstant(1)(sigma))
    compound = nn.CMulTable()({recall, compound})
    compound = nn.Exp()(compound)
    compound = nn.CMulTable()({S, compound})

    local reten = nn.Exp()(nn.CDivTable()({t, compound}))

    -- Sigmoid 1/S
    -- local S_inv = nn.Linear(args.conv_dim, 1)(history_flat)
    -- S_inv = nn.View(-1, T-1, 1):setNumInputDims(2)(S_inv)
    -- S_inv = nn.Sigmoid(S_inv)
    -- local S = nn.Power(-1)(S_inv)

    -- local reten = nn.Exp()(nn.CMulTable({t, S_inv}))

    return reten, S, sigma, compound
end

function MQN:build_retrieval(args, key_blocks, val_blocks, cnn_features, conv_dim, c0, h0, reten)
    local T = args.hist_len
    local edim = args.edim
    local memsize = math.min(T-1, args.memsize)
    local context = self:build_context(args, cnn_features, args.conv_dim, edim, c0, h0)
    local hid = nn.View(1, -1):setNumInputDims(1)(context)
    local key_blocks_t = nn.Narrow(2, T - memsize, memsize)(key_blocks)
    local val_blocks_t = nn.Narrow(2, T - memsize, memsize)(val_blocks)
    -- local MM_key = nn.MM(false, true)
    -- local key_out = MM_key({hid, key_blocks_t})
    -- local key_out2dim = nn.View(-1):setNumInputDims(2)(key_out)

    -- Original
    -- local P = nn.SparseMax()(key_out2dim)
    -- local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)

    -- Multiplication
    -- local atten = nn.SparseMax()(key_out2dim)
    -- atten = nn.CMulTable()({atten, reten})
    -- atten = nn.Normalize(1)(atten)

    -- B x T-1 x edim
    local expanded_hid = nn.ExpandAs()({key_blocks_t, hid})
    -- B x T-1 x 1
    local unsqueezed_reten = nn.View(-1, T-1, 1)(reten)

    local concated = nn.JoinTable(3)({expanded_hid, key_blocks_t, unsqueezed_reten})
    local atten_input = nn.View(-1, edim + edim + 1)(concated)
    local atten_hidden = nn.ReLU()(nn.Linear(edim + edim + 1, edim)(atten_input))
    local atten_out = nn.Linear(edim, 1)(atten_hidden)
    local atten = nn.SparseMax()(nn.View(-1, T-1)(atten_out))

    local probs3dim = nn.View(1, -1):setNumInputDims(1)(atten)

    local MM_val = nn.MM(false, false)
    local o = MM_val({probs3dim, val_blocks_t})
    if args.gpu and args.gpu > 0 then
        -- MM_key = MM_key:cuda()
        MM_val = MM_val:cuda()
    end
    -- return context, o, P
    return context, o, atten
end

function MQN:build_context(args, x, xdim, edim, c0, h0)
    local context = nn.Narrow(2, args.hist_len, 1)(x)
    local context_flat = nn.View(-1):setNumInputDims(1)(context)
    return nn.Linear(xdim, edim)(context_flat)
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
    return nn.View(-1, args.hist_len, args.conv_dim):setNumInputDims(2)(conv_flat)
end