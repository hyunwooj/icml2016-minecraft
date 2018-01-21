local memory = torch.class('Memory')


function memory:__init(args)
    self.mem = {}
    self.mem_size = args.mem_size
    self.image_dims = args.image_dims
    self:reset()
end

function memory:concat(frame)
    assert(#self.mem == self.mem_size, 'memory size error')

    local mem = torch.cat(self.mem, 1)
    mem = torch.cat({mem, frame}, 1)
    return mem
end

function memory:reset()
    self.mem = {}
    for i = 1, self.mem_size do
        local m = torch.ByteTensor(unpack(self.image_dims)):zero()
        table.insert(self.mem, m)
    end
end

function memory:replace(frame, idx)
    self.mem[idx] = frame:clone()
end

function memory:enqueue(frame)
    table.remove(self.mem, 1)
    table.insert(self.mem, frame:clone())
end
