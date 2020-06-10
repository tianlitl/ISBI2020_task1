#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

# The following functions have been taken from YijinHuang's github repository
# https://github.com/YijinHuang/pytorch-DR

MEAN = [0.5552286, 0.5338945, 0.5210911]

STD = [0.16139875, 0.14060262, 0.10415223]

U = torch.tensor([[-0.41687687, -0.67768381, 0.60577085],
                  [-0.59736268, -0.2980542, -0.74452772],
                  [-0.68510693, 0.67224129, 0.28057111]], dtype=torch.float32)

EV = torch.tensor([0.05275069, 0.0026322, 0.00128549], dtype=torch.float32)

BALANCE_WEIGHTS = torch.tensor([1600/714, 1600/186,
                                1600/326, 1600/282,
                                1600/92], dtype=torch.double)
FINAL_WEIGHTS = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)

def generate_stem_dataset(data_path, input_size, data_aug):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=data_aug['scale'],
            ratio=data_aug['stretch_ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['ratation'],
            translate=data_aug['translation_ratio'],
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
        KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD))
    ])

    def load_image(x):
        return Image.open(x)

    return generate_dataset(data_path, load_image, ('jpg', 'jpeg'), train_transform, test_transform)



def generate_dataset(data_path, loader, extensions, train_transform, test_transform):
    train_path = os.path.join(data_path, 'ISBI2020_prep_Mix_sigma10')
#    test_path = os.path.join(data_path, 'ISBI2020_prep_Test_sigma10')
#    val_path = os.path.join(data_path, 'val')

    train_dataset = datasets.DatasetFolder(train_path, loader, extensions, transform=train_transform)
#    test_dataset = datasets.DatasetFolder(test_path, loader, extensions, transform=test_transform)
#    val_dataset = datasets.DatasetFolder(val_path, loader, extensions, transform=test_transform)

    return train_dataset



class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, initial_weight=BALANCE_WEIGHTS,
                 final_weight=FINAL_WEIGHTS, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = initial_weight
        self.wf = final_weight
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = 0.975**(self.epoch - 1) # r=0.975 here is a hyperparameter
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = self.weights[_class]
    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
    
    
    
class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.unsqueeze(dim=1))
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)