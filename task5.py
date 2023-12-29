# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.optim import Adam
from rsdl.optim import Momentum
from rsdl.optim import RMSprop
from rsdl.losses import loss_functions
import sys

sys.setrecursionlimit(100000)
X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
model = Linear(3, 1)

# TODO: define an optimizer using SGD() class 

# optimizer = SGD([model], 0.1) # 100 epoch
# optimizer = Adam([model], learning_rate=0.1) # 50 epoch
# optimizer = Momentum([model], learning_rate=0.1) # 100 epoch
optimizer = RMSprop([model], learning_rate=0.05) # 100 epoch

# TODO: print weight and bias of linear layer
print(f"layer weights: {model.weight}\n")
print(f"layer bias: {model.bias}\n")

learning_rate = optimizer.learning_rate
batch_size = 10

for epoch in range(100):
    
    epoch_loss = 0.0
    print(f"epoch is: {epoch}")

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]
        
        inputs.zero_grad()
        model.zero_grad()

        # TODO: predicted
        predicted = model(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(-1, 1)
        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(predicted, actual)
        
        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()

        # TODO: add loss to epoch_loss
        epoch_loss += loss

        # TODO: update w and b using optimizer.step()
        optimizer.step()
        

# TODO: print weight and bias of linear layer
print(f"layer weights: {model.weight}\n")
print(f"layer bias: {model.bias}\n")