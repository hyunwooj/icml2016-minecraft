local memory = torch.class('Memory')


function memory:__init(args)
    self.mem = {}
    self.times = {}
    self.mem_size = args.mem_size
end

function memory:concat(state)
    local frame = state.frame
    local time = state.time
    -- Reset
    if #self.mem == 0 then
        for i = 1, self.mem_size do
            rand_frame = torch.rand(unpack(frame:size():totable())):mul(127):byte()
            table.insert(self.mem, rand_frame)
            -- zero_frame = frame:clone():zero()
            -- table.insert(self.mem, zero_frame)
            table.insert(self.times, time:clone():zero())
        end
    end
    assert(#self.mem == self.mem_size, 'memory size error')
    assert(#self.times == self.mem_size, 'memory size error')

    local mem = torch.cat(self.mem, 1)
    mem = torch.cat({mem, frame}, 1)

    local times = torch.cat(self.times, 1)
    times = torch.cat({times, time}, 1)
    return mem:clone(), times:clone()
end

function memory:reset()
    self.mem = {}
    self.times = {}
end

function memory:replace_frame(state, idx)
    self.mem[idx] = state.frame:clone()
    self.times[idx] = state.time:clone()
end

function memory:update_time(idxs, time_step)
    for _, idx in pairs(idxs) do
        self.times[idx] = time_step:clone()
    end
end

function memory:enqueue(state)
    table.remove(self.mem, 1)
    table.insert(self.mem, state.frame:clone())

    table.remove(self.times, 1)
    table.insert(self.times, state.time:clone())
end
