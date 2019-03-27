# -*- coding: utf-8 -*-
"""
Implementation of a neural network using PyTorch with custom Autograd function.
Created on Wed Mar 27 14:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

References:
[1] https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

"""


# imports
import torch
import time


# Define custom autograd function.
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(tensor)
        return tensor.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """
        tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[tensor < 0] = 0
        return grad_input


t0 = time.time()

# Set random number generator seed for reproducibility.
torch.manual_seed(0)

# Set device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 25, 100, 50, 10

# Create random input and output data.
x = torch.randn(N, D_in, dtype=torch.float, device=device, requires_grad=False)
y = torch.randn(N, D_out, dtype=torch.float, device=device, requires_grad=False)

# Randomly initialize weights.
w1 = torch.randn(D_in, H, dtype=torch.float, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=torch.float, device=device, requires_grad=True)

# Set learning rate.
learning_rate = 1e-6

# Set activation function.
relu = CustomReLU.apply

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y.
    y_pred = relu(x.mm(w1)).mm(w2)
    
    # Compute loss.
    loss = (y_pred - y).pow(2).sum()
    print(f'[PyTorch_{device.type.upper()}] Epoch {epoch+1:4d} | Loss {loss.item():.4f}')
    
    # Backward pass: compute gradient of the loss with respect to model parameters.
    loss.backward()
    
    # Update weights.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        # Manually zero the gradients after updating weights.
        w1.grad.zero_()
        w2.grad.zero_()

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
