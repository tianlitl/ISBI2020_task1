#!/usr/bin/env python3
# -*- coding: utf-8 -*-


CONFIG = {
    'DATA_PATH': './Data',
    'SAVE_PATH': './Results',
    'PRETRAINED_PATH': None,
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_14_16_04_24/best_kappa.pth',
    'LEARNING_RATE': 1e-4,
    'MILESTONES': [50, 100, 150, 200, 250],
    'GAMMA': 0.5,
    'OPTIMIZER': 'ADAM', #'SGD' or 'ADAM'
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY' : 0,
    'BETAS': (0.9,0.999),
    'EPS': 1e-08,
    'INPUT_SIZE': 512,
    'BATCH_SIZE': 24,
    'EPOCHS': 50,
    'NUM_WORKERS': 8,
    'NUM_GPU': 3,
    'LOSS_FUNC': 'MSELoss', # 'CrossEntropyLoss' or 'MSELoss'
    
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': ( 100 / 512, 100 / 512),  # 100 pixel 
        'sigma': 0.5
    }
}