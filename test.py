import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
output = loss(input, target)
output.backward()

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(input)
print(target)
output = loss(input, target)
output.backward()

a = torch.tensor([0])
b = torch.tensor([1])
c=torch.cat((a,b),0)
print(c[1])
d = [c[0]]
f = c[1][None]
print(f)