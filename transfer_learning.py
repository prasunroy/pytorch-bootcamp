# -*- coding: utf-8 -*-
"""
Example of transfer learning using PyTorch.
Created on Thu Mar 28 11:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/pytorch-bootcamp

References:
[1] https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

"""


# imports
import copy
import math
import matplotlib.pyplot
import numpy
import os
import requests
import time
import tqdm
import zipfile

import torch
import torchvision


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

# download dataset
def download_dataset(url, save_to_dir=None):
    dst = './' if save_to_dir is None else save_to_dir
    res = requests.get(url, stream=True)
    total_size = int(res.headers.get('content-length', 0))
    chunk_size = 1024
    total_chunks = math.ceil(total_size / chunk_size)
    total_dumped = 0
    filepath = os.path.join(dst, os.path.split(url)[-1])
    if not os.path.isdir(dst):
        os.makedirs(dst)
    elif os.path.isfile(filepath):
        return filepath
    with open(filepath, 'wb') as file:
        for data in tqdm.tqdm(res.iter_content(chunk_size), total=total_chunks,
                              unit=' KB'):
            total_dumped += len(data)
            file.write(data)
    if total_chunks == 0 or (total_chunks > 0 and total_dumped != total_size):
        filepath = None
    return filepath


# extract zipfile
def extract_zipfile(filepath, save_to_dir=None, delete_after_extract=False):
    loc = os.path.splitext(filepath)[0]
    if os.path.isdir(loc + '/train/') and os.path.isdir(loc + '/val/'):
        return
    dst = './' if save_to_dir is None else save_to_dir
    with zipfile.ZipFile(filepath) as zf:
        for member in tqdm.tqdm(zf.namelist(), unit=' item'):
            zf.extract(member, path=dst)
    if delete_after_extract:
        os.remove(filepath)
    return


# show image
def imshow(image_tensor, fig_name=''):
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    image_np = image_tensor.numpy().transpose((1, 2, 0))
    image_np = image_np * std + mean
    image_np = numpy.clip(image_np, 0, 1)
    matplotlib.pyplot.title(fig_name)
    matplotlib.pyplot.imshow(image_np)
    matplotlib.pyplot.pause(0.001)
    return

# -----------------------------------------------------------------------------


# download and extract dataset
url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
dst = '../Datasets/'
print('[INFO] Downloading dataset from {}'.format(url))
time.sleep(0.5)
filepath = download_dataset(url, dst)
if filepath is None:
    print('[INFO] Unable to download dataset')
else:
    print('[INFO] Extracting dataset...')
    time.sleep(0.5)
    extract_zipfile(filepath, dst)

# data loading and augmentation
data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
    ])
}

datasets = {
    x: torchvision.datasets.ImageFolder(
        f'../Datasets/hymenoptera_data/{x}/',
        data_transforms[x]
    ) for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True) \
    for x in ['train', 'val']
}

dataset_sizes = {
    x: len(datasets[x]) for x in ['train', 'val']
}

class_names = datasets['train'].classes


# visualize few samples from the dataset
images, labels = next(iter(dataloaders['train']))
img_grid = torchvision.utils.make_grid(images[:5], nrow=5)
fig_name = str([class_names[label] for label in labels[:5]])
imshow(img_grid, fig_name)


# -----------------------------------------------------------------------------
# general function for train a model
# -----------------------------------------------------------------------------

def train(model, criterion, optimizer, scheduler, epochs=10, device=None):
    # set device
    if not device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # initialize weights and accuracy of best model
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print('-'*80)
    
    # begin iteration
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        time.sleep(0.5)
        
        # each epoch has a training phase and a validation phase
        for phase in ['train', 'val']:
            # update learning rate scheduler and set mode
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            # initialize accumulators for loss and number of correct matches
            running_loss = 0.0
            running_matches = 0
            
            # iterate over data
            for inputs, labels in tqdm.tqdm(dataloaders[phase],
                                            unit=' batch', ncols=80):
                # copy tensors to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # optimize
                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # backward pass
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # record statistics
                running_loss += loss.item() * inputs.size(0)
                predictions = torch.max(outputs, 1)[1]
                running_matches += torch.sum(predictions == labels).item()
            
            # update statistics for current phase
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_matches / dataset_sizes[phase]
            
            print(f'{phase}_loss: {epoch_loss:.4f} {phase}_acc: {epoch_acc:.4f}')
            time.sleep(0.5)
            
            # update statistics for model
            if phase == 'val' and epoch_acc > best_acc:
                best_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
        
        print('-'*80)
    
    # return best model
    model.load_state_dict(best_wts)
    return model


# -----------------------------------------------------------------------------
# test model
# -----------------------------------------------------------------------------

def test(model, num_images=8, device=None):
    # set device
    if not device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    training_state = model.training
    model.eval()
    num_prediction = 0
    matplotlib.pyplot.figure()
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            preds = torch.max(preds, dim=1)[1]
            for index, image_tensor in enumerate(images):
                num_prediction += 1
                ax = matplotlib.pyplot.subplot(num_images // 2, 2,
                                               num_prediction)
                ax.axis('off')
                matplotlib.pyplot.tight_layout()
                imshow(image_tensor.cpu(),
                       f'Predicted: {class_names[labels[index]]}')
                if num_prediction >= num_images:
                    model.train(mode=training_state)
                    return
    model.train(mode=training_state)
    return

# -----------------------------------------------------------------------------

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load pretrained model and reset final fully connected layer
model_ckpt = torchvision.models.resnet18(pretrained=True)
model_ckpt.fc = torch.nn.Linear(model_ckpt.fc.in_features, 2)

# transfer model to device
model_ckpt = model_ckpt.to(device)

# create criterion, optimizer and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ckpt.parameters(), lr=1e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# finetune model
train(model_ckpt, criterion, optimizer, scheduler, epochs=10, device=device)

# test model
test(model_ckpt, device=device)

# show predictions
matplotlib.pyplot.show()
