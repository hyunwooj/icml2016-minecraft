local memory = torch.class('Memory')


function memory:__init(args)
    self.mem = {}
    self.times = {}
    self.mem_size = args.mem_size
    self.image_dims = args.image_dims
    self:reset()
end

function memory:concat(state)
    local frame = state.frame
    local time = state.time

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
    for i = 1, self.mem_size do
        local m = torch.rand(unpack(self.image_dims)):mul(127):byte()
        -- local m = torch.ByteTensor(unpack(self.image_dims)):zero()
        table.insert(self.mem, m)

        local time = torch.ByteTensor(1):zero()
        table.insert(self.times, time)
    end
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
