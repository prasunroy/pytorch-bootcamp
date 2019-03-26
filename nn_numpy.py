# -*- coding: utf-8 -*-
"""
Implementation of a neural network using Numpy.
Created on Tue Mar 26 12:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

References:
[1] https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

"""


# imports
import numpy as np
import time


t0 = time.time()

# Set random number generator seed for reproducibility.
np.random.seed(0)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 25, 100, 50, 10

# Create random input and output data.
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Set learning rate.
learning_rate = 1e-6

# Train neural network.
for epoch in range(1000):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # Compute loss.
    loss = np.square(y_pred - y).sum()
    print(f'[Numpy_CPU] Epoch {epoch+1:4d} | Loss {loss:.4f}')
    
    # Backprop to compute gradients of loss with respect to w1 and w2.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    # Update weights.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

t1 = time.time()

print(f'\nTotal execution time: {t1-t0:.2f} seconds')
