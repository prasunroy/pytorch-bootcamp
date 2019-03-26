# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch.
Created on Tue Mar 26 14:00:00 2019
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
x = torch.randn(N, D_in, dtype=torch.float, device=device)
y = torch.randn(N, D_out, dtype=torch.float, device=device)

# Randomly initialize weights.
w1 = torch.randn(D_in, H, dtype=torch.float, device=device)
w2 = torch.randn(H, D_out, dtype=torch.float, device=device)

# Set learning rate.
learning_rate = 1e-6

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    # Compute loss.
    loss = (y_pred - y).pow(2).sum().item()
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss:.4f}')
    
    # Backprop to compute gradients of loss with respect to w1 and w2.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    # Update weights.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
