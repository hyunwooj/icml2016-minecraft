local buffer = torch.class('ReplayBuffer')


function buffer:__init(args)
    self.max_size = args.max_size
    self.frame_dim = args.frame_dim
    self.mem_size = args.mem_size
    self.batch_size = args.batch_size
    self.gpu = args.gpu

    self.s = torch.ByteTensor(self.max_size, self.mem_size*self.frame_dim)
    self.a = torch.ByteTensor(self.max_size)
    self.mem_r = torch.FloatTensor(self.max_size)
    self.r = torch.FloatTensor(self.max_size)
    self.s2 = torch.ByteTensor(self.max_size, self.mem_size*self.frame_dim)
    self.t = torch.ByteTensor(self.max_size)
    self.time = torch.FloatTensor(self.max_size, self.mem_size)
    self.time2 = torch.FloatTensor(self.max_size, self.mem_size)
    self.recall = torch.ByteTensor(self.max_size, self.mem_size-1)
    self.recall2 = torch.ByteTensor(self.max_size, self.mem_size-1)

    self.ptr = 1
    self.full = false

    self.batch_s = torch.ByteTensor(self.batch_size, self.mem_size*self.frame_dim)
    self.batch_a = torch.ByteTensor(self.batch_size)
    self.batch_mem_r = torch.FloatTensor(self.batch_size)
    self.batch_r = torch.FloatTensor(self.batch_size)
    self.batch_s2 = torch.ByteTensor(self.batch_size, self.mem_size*self.frame_dim)
    self.batch_t = torch.ByteTensor(self.batch_size)
    self.batch_time = torch.FloatTensor(self.batch_size, self.mem_size)
    self.batch_time2 = torch.FloatTensor(self.batch_size, self.mem_size)
    self.batch_recall = torch.ByteTensor(self.batch_size, self.mem_size-1)
    self.batch_recall2 = torch.ByteTensor(self.batch_size, self.mem_size-1)

end

function buffer:add(s, a, r, s2, t)
    local time = s.time
    local recall = s.recall
    local s = s.frames

    local time2 = s2.time
    local recall2 = s2.recall
    local s2 = s2.frames

    self.s[self.ptr]:copy(s:clone():mul(255):byte())
    self.a[self.ptr] = a
    self.mem_r[self.ptr] = r.mem
    self.r[self.ptr] = r.beh
    self.s2[self.ptr]:copy(s2:clone():mul(255):byte())
    if t then
        self.t[self.ptr] = 1
    else
        self.t[self.ptr] = 0
    end
    self.time[self.ptr]:copy(time:clone():float())
    self.time2[self.ptr]:copy(time2:clone():float())

    self.recall[self.ptr]:copy(recall:clone():float())
    self.recall2[self.ptr]:copy(recall2:clone():float())

    self.ptr = self.ptr + 1
    if self.ptr > self.max_size then
        self.ptr = 1
        self.full = true
    end
end

function buffer:sample(batch_size)
    local n
    if self.full then
        n = self.max_size
    else
        n = self.ptr - 1
    end
    assert(n >= batch_size, 'not enough samples')
    local sample_idxs = torch.randperm(n)[{{1, batch_size}}]:totable()

    for batch_idx, sample_idx in pairs(sample_idxs) do
        self.batch_s[batch_idx]:copy(self.s[sample_idx])
        self.batch_a[batch_idx] = self.a[sample_idx]
        self.batch_mem_r[batch_idx] = self.mem_r[sample_idx]
        self.batch_r[batch_idx] = self.r[sample_idx]
        self.batch_s2[batch_idx]:copy(self.s2[sample_idx])
        self.batch_t[batch_idx] = self.t[sample_idx]

        self.batch_time[batch_idx]:copy(self.time[sample_idx])
        self.batch_time2[batch_idx]:copy(self.time2[sample_idx])

        self.batch_recall[batch_idx]:copy(self.recall[sample_idx])
        self.batch_recall2[batch_idx]:copy(self.recall2[sample_idx])
    end
    s = self.batch_s:clone():float():div(255)
    a = self.batch_a:clone()
    mem_r = self.batch_mem_r:clone()
    r = self.batch_r:clone()
    s2 = self.batch_s2:clone():float():div(255)
    t = self.batch_t:clone()

    time = self.batch_time:clone()
    time2 = self.batch_time2:clone()

    recall = self.batch_recall:clone()
    recall2 = self.batch_recall2:clone()

    if self:use_gpu() then
        s = s:cuda()
        s2 = s2:cuda()
        time = time:cuda()
        time2 = time2:cuda()
        recall = recall:cuda()
        recall2 = recall2:cuda()
    end
    s = {frames=s, times=time, recall=recall}
    s2 = {frames=s2, times=time2, recall=recall2}
    r = {mem=mem_r, beh=r}
    return s, a, r, s2, t
end

function buffer:use_gpu()
    return self.gpu >= 0
end
