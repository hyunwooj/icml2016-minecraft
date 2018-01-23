local CPowerTable, parent = torch.class('nn.CPowerTable','nn.Module')

function CPowerTable:__init()
    parent.__init(self)
    self.gradInput = {}
end

function CPowerTable:updateOutput(input)
    local base = input[1]
    local pow = input[2]
    self.output:resizeAs(base):copy(base)
    self.output:cpow(pow)
    return self.output
end

function CPowerTable:updateGradInput(input, gradOutput)
    local base = input[1]
    local pow = input[2]

    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(base):copy(base)
    self.gradInput[1]:cpow(pow - 1)
    self.gradInput[1]:cmul(gradOutput):cmul(pow)

    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(pow):copy(base)
    self.gradInput[2]:cpow(pow):cmul(torch.log(base))
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
