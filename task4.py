# Task 4
import numpy as np

from rsdl import Tensor
from rsdl.losses import loss_functions
import sys

sys.setrecursionlimit(100000)
X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# TODO: define w and b (y = w x + b) with random initialization ( you can use np.random.randn )
w = Tensor(np.random.randn(3, 1), requires_grad=True) # in_chan, out_chan = 3, 1
b = Tensor(np.random.randn(), requires_grad=True) # out_chan = 1

print(w)
print(b)

learning_rate = 0.0015
batch_size = 10

for epoch in range(200):
    
    epoch_loss = 0.0
    print(f"epoch is: {epoch}")
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        inputs.zero_grad()
        w.zero_grad()
        b.zero_grad()

        # TODO: predicted
        predicted = inputs @ w + b

        actual = y[start:end]
        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(predicted, actual)
        
        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()

        epoch_loss += loss


        # TODO: update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)
        w = w - learning_rate * w.grad
        b = b - learning_rate * b.grad

print(w)
print(b)

