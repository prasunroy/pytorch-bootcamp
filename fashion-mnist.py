# -*- coding: utf-8 -*-
"""
Classification on Fashion-MNIST dataset.
Created on Sun Mar 10 12:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

"""


# imports
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm


# ETL pipeline - Extract Transform Load
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(
    root='../Datasets/', train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=100, shuffle=True
)

test_set = torchvision.datasets.FashionMNIST(
    root='../Datasets/', train=False, download=True, transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=100, shuffle=True
)

idx_to_class = {idx: class_name \
                for class_name, idx in train_set.class_to_idx.items()}


# visualize few samples from the dataset
def imshow(image, fig_name=''):
    image = image * 0.5 + 0.5
    image_np = image.numpy()
    plt.figure()
    plt.title(fig_name)
    plt.imshow(np.transpose(image_np, axes=(1, 2, 0)))
    plt.show()
    return

images, labels = next(iter(train_loader))
imshow(torchvision.utils.make_grid(images[0:5], nrow=5), 'Train samples')
print(', '.join([idx_to_class[label.item()] for label in labels[0:5]]))

images, labels = next(iter(test_loader))
imshow(torchvision.utils.make_grid(images[0:5], nrow=5), 'Test samples')
print(', '.join([idx_to_class[label.item()] for label in labels[0:5]]))


# define network
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.out = torch.nn.Linear(in_features=60, out_features=10)
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.out(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# train network
network = Network()

print('\n\n', network, '\n\n')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

print('[INFO] Training started using device: {}'.format(device.type.upper()))

time.sleep(0.5)

for epoch in range(10):
    running_loss = 0.
    train_loader_tqdm = tqdm.tqdm(train_loader, desc='[INFO] Epoch {:2d}'\
                                  .format(epoch + 1))
    
    for index, batch in enumerate(train_loader_tqdm):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = network(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print('[INFO] Epoch {:2d}: Loss = {:.4f}'\
          .format(epoch + 1, running_loss / (index + 1)))
    
    time.sleep(0.5)

train_loader_tqdm.close()

print('[INFO] Training finished')


# test network
images, labels = next(iter(test_loader))
imshow(torchvision.utils.make_grid(images[0:5], nrow=5), 'Test samples')
print('y: ', ', '.join([idx_to_class[label.item()] for label in labels[0:5]]))

images, labels = images.to(device), labels.to(device)

pred = network(images)
pred = torch.max(pred, 1)[1]
print('p: ', ', '.join([idx_to_class[label.item()] for label in pred[0:5]]))


# evaluate network
match = 0
total = 0
class_match = [0] * len(idx_to_class)
class_total = [0] * len(idx_to_class)

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        p = network(x)
        p = torch.max(p, 1)[1]
        match += (p == y).sum().item()
        total += y.size(0)
        for i, j in zip(y.squeeze(), (p == y).squeeze()):
            class_match[i] += j.item()
            class_total[i] += 1

for i in range(len(idx_to_class)):
    print('[INFO] Test accuracy of {:12s}: {:.2f}%'\
          .format(idx_to_class[i], 100. * class_match[i] / class_total[i]))

print('[INFO] Accuracy of the network on {} test images: {:.2f}%'\
      .format(total, 100. * match / total))
