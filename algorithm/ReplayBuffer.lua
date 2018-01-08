local buffer = torch.class('ReplayBuffer')


function buffer:__init(args)
    self.exps = {}
    self.max_buffer_size = args.bufferSize
end

function buffer:add(s, a, r, s2, t)
    local exp = {
        s=s:clone():float(),
        s2=s2:clone():float(),
        a=a, r=r, t=t}

    table.insert(self.exps, exp)

    if #self.exps > self.max_buffer_size then
        table.remove(self.exps, 1)
    end
end
