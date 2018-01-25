local memory = torch.class('Memory')


function memory:__init(args)
    self.mem = {}
    self.times = {}
    self.recalls = {}
    self.mem_size = args.mem_size
    self.image_dims = args.image_dims
    self:reset()
end

function memory:concat(state)
    local frame = state.frame
    local time = state.time

    assert(#self.mem == self.mem_size, 'memory size error')
    assert(#self.times == self.mem_size, 'memory size error')
    assert(#self.recalls == self.mem_size, 'memory size error')

    local mem = torch.cat(self.mem, 1)
    mem = torch.cat({mem, frame}, 1)

    local times = torch.cat(self.times, 1)
    times = torch.cat({times, time}, 1)

    local recalls = torch.cat(self.recalls, 1)

    return mem:clone(), times:clone(), recalls:clone()
end

function memory:reset()
    self.mem = {}
    self.times = {}
    self.recalls = {}
    for i = 1, self.mem_size do
        local m = torch.rand(unpack(self.image_dims)):mul(127):byte()
        -- local m = torch.ByteTensor(unpack(self.image_dims)):zero()
        table.insert(self.mem, m)

        local time = torch.FloatTensor(1):fill(-100)
        table.insert(self.times, time)

        local recall = torch.ByteTensor(1):zero()
        table.insert(self.recalls, recall)
    end
end

function memory:replace_frame(state, idx)
    self.mem[idx] = state.frame:clone()
    self.times[idx] = state.time:clone()
    self.recalls[idx] = self.recalls[idx]:clone():zero()
end

function memory:recall(idxs, time_step)
    for _, idx in pairs(idxs) do
        self.times[idx] = time_step:clone()
        self.recalls[idx]:add(1)
    end
end

function memory:enqueue(state)
    table.remove(self.mem, 1)
    table.insert(self.mem, state.frame:clone())

    table.remove(self.times, 1)
    table.insert(self.times, state.time:clone())

    table.remove(self.recalls, 1)
    table.insert(self.recalls, self.recalls[1]:clone():zero())
end
