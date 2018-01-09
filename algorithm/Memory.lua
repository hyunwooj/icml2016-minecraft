local memory = torch.class('Memory')


function memory:__init(args)
    self.mem = {}
    self.mem_size = args.mem_size
end

function memory:get(frame)
    -- Reset
    if #self.mem == 0 then
        for i = 1, self.mem_size do
            table.insert(self.mem, frame:clone():zero())
        end
    end
    assert(#self.mem == self.mem_size, 'memory size error')

    local mem = torch.cat(self.mem, 1)
    mem = torch.cat({mem, frame}, 1)
    return mem
end

function memory:reset()
    self.mem = {}
end

function memory:replace(frame, idx)
    self.mem[idx] = frame:clone()
end
