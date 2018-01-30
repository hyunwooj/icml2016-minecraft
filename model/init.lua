require 'nn'
require 'nngraph'
require 'util.LinearNB'
require 'model.dqn'
require 'model.drqn'
require 'model.mqn'
require 'model.rmqn'
require 'model.frmqn'
require 'model.reten_mqn'
require 'model.reten_rmqn'
require 'model.shared_mqn'
require 'model.shared_rmqn'
require 'model.shared_frmqn'

function g_create_network(args)
    local new_args = {}
    new_args.name               = args.name
    new_args.hist_len           = args.hist_len or 10
    new_args.n_actions          = args.n_actions or 6
    new_args.ncols              = args.ncols or 3
    new_args.image_dims         = args.image_dims or {3, 32, 32}
    new_args.input_dims         = args.input_dims or {new_args.hist_len * new_args.ncols, 32, 32}
    new_args.n_units            = args.n_units or {32, 64}
    new_args.filter_size        = args.filter_size or {4, 4}
    new_args.filter_stride      = args.filter_stride or {2, 2}
    new_args.pad                = args.pad or {1, 1}
    new_args.n_hid_enc          = args.n_hid_enc or 256
    new_args.edim               = args.edim or 256
    new_args.memsize            = args.memsize or (new_args.hist_len - 1)
    new_args.lindim             = args.lindim or new_args.edim / 2
    new_args.lstm_dim           = args.edim or 256
    new_args.gpu                = args.gpu or -1
    new_args.conv_dim           = new_args.n_units[#new_args.n_units] * 8 * 8
    new_args.Linear             = nn.LinearNB

    if args.name == "dqn" then
        return DQN.new(new_args)
    elseif args.name == "drqn" then
        return DRQN.new(new_args)
    elseif args.name == "mqn" then
        return MQN.new(new_args)
    elseif args.name == "rmqn" then
        return RMQN.new(new_args)
    elseif args.name == "frmqn" then
        return FRMQN.new(new_args)
    elseif args.name == "shared_mqn" then
        return SharedMQN.new(new_args)
    elseif args.name == "shared_rmqn" then
        return SharedRMQN.new(new_args)
    elseif args.name == "shared_frmqn" then
        return SharedFRMQN.new(new_args)
    elseif args.name == "reten_mqn" then
        return RetenMQN.new(new_args)
    elseif args.name == "reten_rmqn" then
        return RetenRMQN.new(new_args)
    else
        error("Invalid model name:" .. args.name)
    end
end
