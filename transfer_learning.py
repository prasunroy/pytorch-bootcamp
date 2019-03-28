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
import math
import matplotlib
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
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(fig_name)
    matplotlib.pyplot.imshow(image_np)
    matplotlib.pyplot.show(block=False)
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


# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
