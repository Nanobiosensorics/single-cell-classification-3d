#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
from classifier.train import train, multi_acc
from classifier.video.cellnet import get_model
from data.dataset import VideoDataset
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def __run_train(result_path, model, train_dl, val_dl, epochs, patience, lr, model_path, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)   # optimize all cnn parameters

    history = train(model, train_dl, val_dl, optimizer, epochs, patience, model_path, device=device)
    train_losses, test_losses, train_accs, test_accs = history

    pd.DataFrame(list(zip(*history))).to_csv(os.path.join(result_path, 'history.csv'), header=None, index=None)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(train_losses, label='train')
    ax[0].plot(test_losses, label='val')
    ax[1].plot(train_accs, label='train')
    ax[1].plot(test_accs, label='val')
    plt.savefig(os.path.join(result_path, 'train-test-metrics.png'), dpi=300)

    # labels = []
    # preds = []
    # model.load_state_dict(torch.load(model_path), strict=False)

    # model.eval()
    # model.to(device)

    # acc_mean = 0
    # for b_x, b_y in test_dl:
    #     b_x, b_y = b_x.to(device), b_y.to(device)

    #     y_pred = model(b_x)

    #     y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    #     _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    #     # print(y_pred)

    #     correct_pred = (y_pred_tags == b_y).float()
    #     acc = correct_pred.sum() / len(correct_pred)

    #     acc = torch.round(acc * 100)
    #     acc_mean += acc.item()

    #     preds.extend(test_dl.dataset.encoder.inverse_transform(y_pred_tags.cpu().detach().numpy()))
    #     labels.extend(test_dl.dataset.encoder.inverse_transform(b_y.cpu().detach().numpy()))
    # print(acc_mean / len(test_dl))

    # pd.DataFrame(list(zip(labels, preds))).to_csv(os.path.join(result_path, 'y_preds.csv'), header=None, index=None)

def run(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.kfold <= 0:
        raise ValueError()
    
    file_name = '-'.join([*sorted(list(args.cell_types)), str(args.time), args.type])

    if args.kfold == 1:
        result_path = os.path.join(args.result_path, args.model, file_name)
        model_path = os.path.join(result_path, f'model.pt')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        LR = 1e-4              # learning rate
        train_ds = VideoDataset(args.path, os.path.join(args.path, f'{file_name}-dataset.pkl'), partition='train', normalize=args.normalize, tp=args.type)

        val_ds = VideoDataset(args.path, os.path.join(args.path, f'{file_name}-dataset.pkl'), partition='val', normalize=args.normalize, tp=args.type)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=(os.cpu_count() - 1), persistent_workers=True, prefetch_factor=256)
        val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=(os.cpu_count() - 1), persistent_workers=True, prefetch_factor=256)
        
        model = get_model(args.model, nn.CrossEntropyLoss(), len(train_ds.type_set), int(train_ds.time * 9 / 60))
        print(model)
        
        __run_train(result_path, model, train_dl, val_dl, args.epochs, args.patience, LR, model_path, device)

    else:
        for i in range(args.kfold):
            result_path = os.path.join(args.result_path, args.model, file_name, f'cv{i}')
            model_path = os.path.join(result_path, f'model.pt')
            if not os.path.exists(result_path):
                os.makedirs(result_path)


            LR = 1e-4              # learning rate
            train_ds = VideoDataset(args.path, os.path.join(args.path, f'{file_name}-cv-dataset.pkl'), partition='train', normalize=args.normalize, tp=args.type, cross_validation=f'cv{i}')

            val_ds = VideoDataset(args.path, os.path.join(args.path, f'{file_name}-cv-dataset.pkl'), partition='val', normalize=args.normalize, tp=args.type, cross_validation=f'cv{i}')

            train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=(os.cpu_count() - 1), persistent_workers=True, prefetch_factor=256)
            val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=(os.cpu_count() - 1), persistent_workers=True, prefetch_factor=256)
            # test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=(os.cpu_count() - 1), persistent_workers=True, prefetch_factor=256)
            
            model = get_model(args.model, nn.CrossEntropyLoss(), len(train_ds.type_set), int(train_ds.time * 9 / 60))
            print(model)
            
            __run_train(result_path, model, train_dl, val_dl, args.epochs, args.patience, LR, model_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Cell Classification')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-ct', '--cell_types', nargs='+', required=True)
    parser.add_argument('-tm', '--time', type=int, required=True)
    parser.add_argument('-m', '--model', type=str, choices=['cnn', 'resnet', 'cnn-lstm', 'densenet'], default='cnn')
    parser.add_argument('--normalize', type=str, choices=['norm', 'std'], default='std')
    parser.add_argument('-t', '--type', type=str, choices=['max', 'int', 'im_cardio', 'im_cover', 'im_watershed', 'im_pred'], default='im_cardio')
    parser.add_argument('-r', '--result_path', type=str, default='.')
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-pt', '--patience', type=int, default=15)
    parser.add_argument('-k', '--kfold', type=int, default=0)
    
    args = parser.parse_args()
    run(args)


