--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "util.initenv"
end
require "xlua"
local color = require "trepl.colorize"
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', 'environment.mcwrap', 'name of training framework')
cmd:option('-env', '', 'task name for training')
cmd:option('-test_env', '', 'task names for testing (comma-separated)')
cmd:option('-test_hist_len', 30, 'history length for testing')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-save_name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'name of architecture or the filename of pretrained model')
-- cmd:option('-agent', 'NeuralQLearner', 'name of agent file to use')
-- cmd:option('-agent', 'Agent', 'name of agent file to use')
-- cmd:option('-agent', 'TestAgent', 'name of agent file to use')
-- cmd:option('-agent', 'SharedQAgent', 'name of agent file to use')
-- cmd:option('-agent', 'SeparateQAgent', 'name of agent file to use')
cmd:option('-agent', 'MemNql', 'name of agent file to use')
-- cmd:option('-agent', 'SharedMemNql', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'random seed')
cmd:option('-saveNetworkParams', true, 'saves the parameter in a separate file')
cmd:option('-save_freq', 1e5, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 1e5, 'frequency of greedy evaluation')
cmd:option('-eval_steps', 1e4, 'number of evaluation steps')
cmd:option('-steps', 15e6, 'number of training steps')
cmd:option('-verbose', 2, 'the level of debug prints')
cmd:option('-threads', 4, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu id')
cmd:option('-port', 0, 'port number for minecraft: search over [30000,30100] if 0')
cmd:option('-ipaddr', '0.0.0.0', 'ip address for mincraft')
cmd:text()
local opt = cmd:parse(arg)

-- evaluate agent
function eval_agent(env, agent, steps)
    local screen, reward, terminal
    local total_reward = 0
    local nepisodes = 0
    local episode_reward = 0
    screen, reward, terminal = env:newGame()
    local estep = 1
    while true do
        xlua.progress(math.min(estep, steps), steps)
        local action_index = agent:perceive(reward, screen, terminal, true, 0.0)
        screen, reward, terminal = env:step(agent.actions[action_index])
        if estep % 1000 == 0 then collectgarbage() end
        episode_reward = episode_reward + reward
        if terminal then
            total_reward = total_reward + episode_reward
            episode_reward = 0
            nepisodes = nepisodes + 1
            action_index = agent:perceive(reward, screen, terminal, true, 0.0)
            if estep >= steps then
                break
            end
            screen, reward, terminal = env:newGame()
        end
        estep = estep + 1
    end
    return nepisodes, total_reward
end

-- General setup
local game_env, game_actions, agent, opt = setup(opt)
local train_env = opt.env

-- Load testing environments
local test_env_names = {}
local test_env = {}
local test_agent
for s in string.gmatch(opt.test_env, '([^,]+)') do
    table.insert(test_env_names, s)
    opt.env = s
    test_env[#test_env + 1], game_actions = create_env(opt)
end
if #test_env > 0 then
    local agent_param = {}
    for k, v in pairs(opt.agent_params) do
        agent_param[k] = v
    end
    agent_param.actions = game_actions
    agent_param.hist_len = opt.test_hist_len
    agent_param.minibatch_size = 1
    agent_param.target_q = nil
    agent_param.replay_memory = 10000
    test_agent = create_agent(opt, agent_param)
    share_weights(agent.network.net, test_agent.network.net)
    -- share_weights(agent.actor.net, test_agent.actor.net)
    -- share_weights(agent.critic.net, test_agent.critic.net)
    -- share_weights(agent.mem_network.net, test_agent.mem_network.net)
    -- share_weights(agent.mem_network.net, test_agent.mem_network.net)
end

local learn_start = agent.learn_start
local episode_counts = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local test_reward_history = {}
local best_history = {}
local step = 0
for i=1,#test_env do
    test_reward_history[i] = {}
end

local total_reward, nepisodes
local screen, reward, terminal = game_env:getState()
local epoch_time = sys.clock()

local ep_reward = 0
local ep_step = 0
local ev_flag = false
os.execute("mkdir -p save")
local epoch = 0
while step < opt.steps do
    step = step + 1
    xlua.progress(math.min(step - epoch * opt.eval_freq, opt.eval_freq), opt.eval_freq)
    if step % opt.eval_freq == 0 and step > learn_start then ev_flag = true end
    local action_index = agent:perceive(reward, screen, terminal)
    screen, reward, terminal = game_env:step(game_actions[action_index], true)
    ep_reward = ep_reward + reward
    ep_step = ep_step + 1

    if terminal then
        step = step + 1
        local action_index = agent:perceive(reward, screen, terminal)
        if step % opt.eval_freq == 0 and step > learn_start then ev_flag = true end
        screen, reward, terminal = game_env:newGame()
        ep_reward = 0
        ep_step = 0
        if ev_flag then
            epoch = epoch + 1
            epoch_time = sys.clock() - epoch_time
            print("Epoch:", epoch, "Steps/Sec:",  math.floor(opt.eval_freq / epoch_time))
            if opt.verbose > 2 then
                agent:report()
            end
            epoch_time = sys.clock()
            print("Evaluating the agent on the training environment: "
                    .. color.green(train_env))
            ev_flag = false
            nepisodes, total_reward = eval_agent(game_env, agent, opt.eval_steps)
            local ind = #reward_history+1
            total_reward = total_reward/math.max(1, nepisodes)
            if agent.v_avg then
                v_history[ind] = agent.v_avg
                td_history[ind] = agent.tderr_avg
                qmax_history[ind] = agent.q_max
            end
            print("Reward:", total_reward, "num. ep.:", nepisodes)
            reward_history[ind] = total_reward
            episode_counts[ind] = nepisodes
            screen, reward, terminal = game_env:newGame()
            if #test_env > 0 then
                if not agent.best_test_network then
                    agent.best_test_network = {}
                end
                for test_id=1,#test_env do
                    local ind = #test_reward_history[test_id]+1
                    print("Evaluating the agent on the test environment: "
                            .. color.green(test_env_names[test_id]))
                    nepisodes, total_reward = eval_agent(test_env[test_id], test_agent, opt.eval_steps)
                    total_reward = total_reward/math.max(1, nepisodes)
                    if #test_reward_history[test_id] == 0 or
                            total_reward > torch.Tensor(test_reward_history[test_id]):max() then
                        agent.best_test_network[test_id] = test_agent.network:clone():float()
                        -- agent.best_test_network[test_id] = {
                        --     actor=test_agent.actor:clone():float(),
                        --     critic=test_agent.critic:clone():float(),
                        -- }
                        -- agent.best_test_network[test_id] = {
                        --     mem_network=test_agent.mem_network:clone():float(),
                        --     beh_network=test_agent.beh_network:clone():float(),
                        -- }
                    end
                    test_reward_history[test_id][ind] = total_reward
                    print("Reward:", total_reward, "num. ep.:", nepisodes)
                end

                -- Maintain and save only top K best models
                if opt.saveNetworkParams then
                    local filename = string.format('save/%s_%03d.params.t7', opt.save_name, epoch)
                    torch.save(filename, agent.w:clone():float())
                    -- torch.save(filename, {actor=agent.actor_w:clone():float(),
                    --                       critic=agent.critic_w:clone():float()})
                    -- torch.save(filename, {mem=agent.mem_w:clone():float(),
                    --                       beh=agent.beh_w:clone():float()})
                    print('Parameter saved to:', filename)
                end
                collectgarbage()
            end
        end
    end

    if step%1000 == 0 then collectgarbage() end
    if step % opt.save_freq == 0 or step == opt.steps then
        local filename = 'save/' .. opt.save_name .. ".t7"
        torch.save(filename, {model = agent.network,
        -- torch.save(filename, {model = {actor = agent.actor, critic = agent.critic},
                                best_model = agent.best_test_network,
                                test_reward_history = test_reward_history,
                                reward_history = reward_history,
                                episode_counts = episode_counts,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                test_history = test_history,
                                arguments=opt,
                                step=step})
        print('Saved:', filename)
        collectgarbage()
    end
end
