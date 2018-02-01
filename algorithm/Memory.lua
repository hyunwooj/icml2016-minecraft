local memory = torch.class('Memory')

local mem_base = torch.class('MemoryBase')
local st_mem = torch.class('StMemory', 'MemoryBase')
local lt_mem = torch.class('LtMemory', 'MemoryBase')

--[[
    Memory
--]]
function memory:__init(args)
    self.st = StMemory({
            mem_size = args.st_mem_size,
            image_dims = args.image_dims,
        })
    self.lt = LtMemory({
            mem_size = args.lt_mem_size,
            image_dims = args.image_dims,
        })
end

function memory:concat(state)
    local frame = state.frame
    local time = state.time

    assert(#self.st.mem == self.st.mem_size, 'memory size error')

    assert(#self.lt.mem == self.lt.mem_size, 'memory size error')
    assert(#self.lt.times == self.lt.mem_size, 'memory size error')
    assert(#self.lt.recalls == self.lt.mem_size, 'memory size error')

    local lt_mem = torch.cat(self.lt.mem, 1)
    local st_mem = torch.cat(self.st.mem, 1)
    mem = torch.cat({lt_mem, st_mem, frame}, 1)

    local times = torch.cat(self.lt.times, 1)
    times = torch.cat({times, time}, 1)

    local recalls = torch.cat(self.lt.recalls, 1)

    -- assert(mem:size()[1] == #self.st.mem + #self.lt.mem + 1, 'concat size error')
    -- assert(times:size()[1] == #self.lt.mem + 1, 'concat size error')
    -- assert(recalls:size()[1] == #self.lt.mem, 'concat size error')

    return mem:clone(), times:clone(), recalls:clone()
end

function memory:recall(idxs, time_step)
    self.lt:recall(idxs, time_step)
end

function memory:memorize(idx, state)
    local s = self.st:enqueue(state)
    self.lt:replace(idx, s)
end

function memory:reset()
    self.st:reset()
    self.lt:reset()
end

--[[
    Memory Base
--]]
function mem_base:__init(args)
    self.mem = {}
    self.times = {}
    self.mem_size = args.mem_size
    self.image_dims = args.image_dims
end

function mem_base:reset()
    self.mem = {}
    self.times = {}
    for i = 1, self.mem_size do
        -- local m = torch.rand(unpack(self.image_dims)):mul(127):byte()
        local m = torch.ByteTensor(unpack(self.image_dims)):zero()
        table.insert(self.mem, m)

        local time = torch.FloatTensor(1):fill(-100)
        table.insert(self.times, time)
    end
end

--[[
    Short-term Memory
--]]
function st_mem:__init(args)
    mem_base.__init(self, args)
    self:reset()
end

function st_mem:enqueue(state)
    local frame = self.mem[1]:clone()
    table.remove(self.mem, 1)
    table.insert(self.mem, state.frame:clone())

    local time = self.times[1]:clone()
    table.remove(self.times, 1)
    table.insert(self.times, state.time:clone())
    return {frame=frame, time=time}
end

--[[
    Long-term Memory
--]]
function lt_mem:__init(args)
    mem_base.__init(self, args)
    self.recalls = {}
    self:reset()
end

function lt_mem:reset()
    mem_base.reset(self)
    self.recalls = {}
    for i = 1, self.mem_size do
        local recall = torch.ByteTensor(1):zero()
        table.insert(self.recalls, recall)
    end
end

function lt_mem:replace(idx, state)
    self.mem[idx] = state.frame:clone()
    self.times[idx] = state.time:clone()
    self.recalls[idx] = self.recalls[idx]:clone():zero()
end

function lt_mem:recall(idxs, time_step)
    if #idxs == 0 then
        return
    end
    for _, idx in pairs(idxs) do
        self.times[idx] = time_step:clone()
        self.recalls[idx]:add(1)
    end
end
