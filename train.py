#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import generate_stem_dataset, ScheduledWeightedSampler
from model import GoogleNetDR, AlexNetDR
from model import load_pretrain_param_googlenet, load_pretrain_param_alexnet, Ensemble
from utils import print_msg
from metrics import accuracy, quadratic_weighted_kappa

def train(CONFIG):
    
    #creat result folder and save config as txt file
    t = time.strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = os.path.join(CONFIG['SAVE_PATH'],t)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir,'Settings.txt'), 'w') as file:
        file.write(json.dumps(CONFIG))
    
    # creat train dataset
    train_dataset = generate_stem_dataset(CONFIG['DATA_PATH'],
                                          CONFIG['INPUT_SIZE'],
                                          CONFIG['DATA_AUGMENTATION'])

    # split train dataset for 5-fold stratified cross validation 
    kf = model_selection.KFold(n_splits=5,shuffle=True)
        
    for fold_num, (train_index, test_index) in enumerate(kf.split(train_dataset)):
        
        #creat sub_train&sub_test dataset 
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        test_subset = torch.utils.data.Subset(train_dataset, test_index)
        
        #define dynamic weighted resampler
        train_targets = [item[1] for item in train_subset]
        weighted_sampler = ScheduledWeightedSampler(len(train_subset), train_targets, True)
        
        #creat dataloader
        train_loader = DataLoader(train_subset,
                              batch_size=CONFIG['BATCH_SIZE'],
                              sampler=weighted_sampler,
                              num_workers=CONFIG['NUM_WORKERS'],
                              drop_last=False)
        test_loader = DataLoader(test_subset, batch_size=CONFIG['BATCH_SIZE'],
                                 num_workers=CONFIG['NUM_WORKERS'],
                                 shuffle=False)
        
        # define model
        m1 = AlexNetDR()
        m2 = GoogleNetDR()
    
        m1 = load_pretrain_param_alexnet(m1)
        m2 = load_pretrain_param_googlenet(m2)
    
        model = Ensemble(m1, m2)
        model = model.cuda(CONFIG['NUM_GPU'])
        
        # load pretrained weights
        if CONFIG['PRETRAINED_PATH']:
            checkpoint = torch.load(CONFIG['PRETRAINED_PATH'])
            model.load_state_dict(checkpoint)
        
        # define loss and optimizer
        if CONFIG['LOSS_FUNC'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif CONFIG['LOSS_FUNC'] == 'MSELoss':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        
        if CONFIG['OPTIMIZER'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=CONFIG['LEARNING_RATE'],
                                        momentum=CONFIG['MOMENTUM'],
                                        nesterov=True,
                                        weight_decay=CONFIG['WEIGHT_DECAY'])
        elif CONFIG['OPTIMIZER'] == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=CONFIG['LEARNING_RATE'],
                                         betas=CONFIG['BETAS'],
                                         eps=CONFIG['EPS'],
                                         weight_decay=CONFIG['WEIGHT_DECAY'])
        else:
            raise NotImplementedError
            
        # learning rate decay
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=CONFIG['MILESTONES'],
                                                            gamma=CONFIG['GAMMA'])
        # train
        max_kappa = 0
        record_epochs, accs, losses, kappa_per_fold = [], [], [], []
        for epoch in range(1, CONFIG['EPOCHS']+1):
            
            # resampling weight update
            if weighted_sampler:
                weighted_sampler.step()
                
            # learning rate update
            if lr_scheduler:
                lr_scheduler.step()
                if epoch in lr_scheduler.milestones:
                    print_msg('Learning rate decayed to {}'.format(lr_scheduler.get_lr()[0]))
            
            epoch_loss = 0
            correct = 0
            total = 0
            progress = tqdm(enumerate(train_loader))
            for step, train_data in progress:
                X, y = train_data # X.dtype is torch.float32, y.dtype is torch.int64
                X, y = X.cuda(CONFIG['NUM_GPU']), y.float().cuda(CONFIG['NUM_GPU'])
    
                # forward
                y_pred = model(X)
                
                y_one_hot = torch.zeros(y.shape[0], 5).cuda(CONFIG['NUM_GPU'])
                y_one_hot[range(y_one_hot.shape[0]), y.to(dtype=torch.int64)]=1
                
                loss = criterion(y_pred, y_one_hot)
    
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # metrics
                epoch_loss += loss.item()
                total += y.size(0)
                correct += accuracy(torch.argmax(y_pred, dim=1), y, method='5_class_vec_output') * y.size(0)
                avg_loss = epoch_loss / (step + 1)
                avg_acc = correct / total
                
                progress.set_description('Fold {} Epoch: {}/{}, loss: {:.6f}, acc: {:.4f}'.format(fold_num+1, epoch, CONFIG['EPOCHS'], avg_loss, avg_acc))
            
            # save model and kappa score&confusion matrix 
            acc, c_matrix, kappa, all_pred = _eval(model, test_loader, CONFIG)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            if kappa > max_kappa:
                torch.save(model.state_dict(), results_dir+'/fold'+str(fold_num+1)+'_best_kappa.pth')
                max_kappa = kappa
                print_msg('Fold {} of 5. Best kappa model save at {}'.format(fold_num+1, results_dir))
                print_msg('Fold '+str(fold_num+1)+' of 5. Confusion matrix with best kappa is:\n', c_matrix)
    #            ks_dataframe = pd.DataFrame({'file_name':[sampler[0] for sampler in test_dataset.samples],
    #                                     'truth':[sampler[1] for sampler in test_dataset.samples],
    #                                     'prediction':list(all_pred),
    #                                     'kappa_score':''})
    #            ks_dataframe.at[0,'kappa_score'] = kappa
    #            ks_dataframe.to_csv(os.path.join(results_dir,'test_kappa_score.csv'),index=False,sep=',')
                np.savetxt(os.path.join(results_dir,'ford'+str(fold_num+1)+'_confusion_matrix.csv'), np.array(c_matrix), delimiter = ',')
                with open(os.path.join(results_dir, 'ford'+str(fold_num+1)+'_kappa_score.txt'), 'w') as f:
                    f.write('Best kappa: {}'.format(kappa))

            # record
            record_epochs.append(epoch)
            accs.append(acc)
            losses.append(avg_loss)
        kappa_per_fold.append(max_kappa)
    print('\nBest validation kappa score for fold 1 to 5:\n {}'.format(kappa_per_fold))
    return record_epochs, accs, losses

# func 'evaluate' does not finish yet 
def evaluate(model_path, test_dataset, CONFIG):
    c_matrix = np.zeros((5,5), dtype=int)
    trained_model = torch.load(model_path).cuda(CONFIG['NUM_GPU'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_acc, test_c_matrix, test_kappa = _eval(trained_model, test_loader)
    print('==============================')
    print('Finished! test acc: {}'.format(test_acc))
    print('Confusion Matrix:')
    print(c_matrix)
    print('quadratic kappa: {}'.format(quadratic_weighted_kappa(c_matrix)))
    print('==============================')
    

def _eval(model, dataloader, CONFIG):
    model.eval()
    torch.set_grad_enabled(False)
    
    correct = 0
    total = 0
    
    all_targ = torch.tensor([]).to(dtype=torch.int64).cuda(CONFIG['NUM_GPU'])
    all_pred = torch.tensor([]).to(dtype=torch.int64).cuda(CONFIG['NUM_GPU'])
    
    for test_data in dataloader:
        X, y = test_data
        X, y = X.cuda(CONFIG['NUM_GPU']), y.cuda(CONFIG['NUM_GPU'])
        
        y_pred = model(X)

        all_pred = torch.cat((all_pred, torch.argmax(y_pred,dim=1)))
        all_targ = torch.cat((all_targ, y.to(torch.int64)))
        total += y.size(0)
        correct += accuracy(torch.argmax(y_pred,dim=1), y, method='5_class_vec_output') * y.size(0)
    acc = round(correct / total, 4)
    c_matrix, kappa = quadratic_weighted_kappa(all_targ.cpu().numpy(), all_pred.cpu().numpy())
    model.train()
    torch.set_grad_enabled(True)
    return acc, c_matrix, kappa, all_pred.cpu().numpy()


