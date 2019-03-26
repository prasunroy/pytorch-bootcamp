# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch with Autograd.
Created on Tue Mar 26 17:00:00 2019
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
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, dtype=torch.float, device=device, requires_grad=False)
y = torch.randn(N, D_out, dtype=torch.float, device=device, requires_grad=False)

# Randomly initialize weights.
# Setting requires_grad=True indicates that we want to compute gradients
# with respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, dtype=torch.float, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=torch.float, device=device, requires_grad=True)

# Set learning rate.
learning_rate = 1e-6

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y using operations on Tensors; these are
    # exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values
    # since we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    # Compute loss.
    loss = (y_pred - y).pow(2).sum()
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss.item():.4f}')
    
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()
    
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        # Manually zero the gradients after updating weights.
        w1.grad.zero_()
        w2.grad.zero_()

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
