local MemoryDrop, Parent = torch.class('nn.MemoryDrop', 'nn.Module')

function MemoryDrop:__init()
    Parent.__init(self)
    self.mask = torch.Tensor()
    self.gradInput = {}
end

function MemoryDrop:updateOutput(input)
    local memory = input[1]
    local reten = input[2]
    local noise = input[3]
    self.mask:resizeAs(memory)
    self.output:resizeAs(memory):copy(memory)

    local mask = noise:lt(reten)
    mask = nn.utils.addSingletonDimension(mask, 3)
    self.mask:copy(mask:expandAs(self.mask))

    self.output:cmul(self.mask)
    return self.output
end

function MemoryDrop:updateGradInput(input, gradOutput)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
    self.gradInput[1]:cmul(self.mask)

    self.gradInput[2] = input[2]:clone():fill(0)
    self.gradInput[3] = input[3]:clone():fill(0)

    return self.gradInput
end

function MemoryDrop:clearState()
    if self.mask then
        self.mask:set()
    end
    return Parent.clearState(self)
end
