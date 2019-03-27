# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch custom nn module.
Created on Wed Mar 27 17:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

References:
[1] https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

"""


# imports
import torch
import time


# Define network.
class Network(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules
        and assign them as member variables.
        """
        super(Network, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        return
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


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

# Construct model.
model = Network(D_in, H, D_out)

# Set learning rate.
learning_rate = 1e-6

# Define criterion and optimizer.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y.
    y_pred = model(x)
    
    # Compute loss.
    loss = criterion(y_pred, y)
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss.item():.4f}')
    
    # Zero gradients, perform backprop and update weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
