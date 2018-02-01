require 'nn'
require 'nngraph'
require 'util.LinearNB'
require 'model.hybrid.mqn'
require 'model.shared_frmqn'

function g_create_network(args)
    local new_args = {}
    new_args.name               = args.name
    new_args.st_mem_size        = args.st_mem_size
    new_args.lt_mem_size        = args.lt_mem_size
    new_args.mem_size           = args.mem_size
    new_args.n_actions          = args.n_actions
    new_args.ncols              = args.ncols
    new_args.image_dims         = args.image_dims
    new_args.input_dims         = args.input_dims
    new_args.n_units            = {32, 64}
    new_args.filter_size        = {4, 4}
    new_args.filter_stride      = {2, 2}
    new_args.pad                = {1, 1}
    new_args.n_hid_enc          = 256
    new_args.edim               = 256
    new_args.lindim             = new_args.edim / 2
    new_args.lstm_dim           = 256
    new_args.gpu                = args.gpu or -1
    new_args.conv_dim           = new_args.n_units[#new_args.n_units] * 8 * 8
    new_args.Linear             = nn.LinearNB

    if args.name == "hybrid_mqn" then
        return HybridMQN.new(new_args)
    else
        error("Invalid model name:" .. args.name)
    end
end
