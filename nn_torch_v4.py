# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch nn module.
Created on Wed Mar 27 15:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

References:
[1] https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

"""


# imports
import torch
import time


t0 = time.time()

# Set random number generator seed for reproducibility.
torch.manual_seed(0)

# Set device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 25, 100, 50, 10

# Create random input and output data.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

# Set learning rate.
learning_rate = 1e-6

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y by passing x to the model. Module
    # objects override the __call__ operator so you can call them like
    # functions. When doing so you pass a Tensor of input data to the Module
    # and it produces a Tensor of output data.
    y_pred = model(x)
    
    # Compute loss. We pass Tensors containing the predicted and true values of
    # y, and the loss function returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y)
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss.item():.4f}')
    
    # Zero the gradients before running the backward pass.
    model.zero_grad()
    
    # Backward pass: compute gradient of the loss with respect to all the
    # learnable parameters of the model. Internally, the parameters of each
    # Module are stored in Tensors with requires_grad=True, so this call will
    # compute gradients for all learnable parameters in the model.
    loss.backward()
    
    # Update weights.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
