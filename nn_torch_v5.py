# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch nn module with optim package.
Created on Wed Mar 27 16:00:00 2019
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

# Define model.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# Define loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

# Set learning rate.
learning_rate = 1e-6

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y.
    y_pred = model(x)
    
    # Compute loss.
    loss = loss_fn(y_pred, y)
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss.item():.4f}')
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers (i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the loss with respect to model parameters.
    loss.backward()
    
    # Calling the step function on an Optimizer makes an update to its parameters.
    optimizer.step()

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
