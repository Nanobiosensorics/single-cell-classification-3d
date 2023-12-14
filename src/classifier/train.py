import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

def cust(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{hour:02d}:{min:02d}:{sec:02d}".format(hour=int(h), min=int(m), sec=int(s)) 

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    # _, y_test_tags = torch.max(y_test, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train(model, train_loader, val_loader, optimizer, epochs=500, patience=50, save_path='./model.pt', device='cpu'):

    best_loss = 1000
    best_loss_epoch = 0
    model.train()
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = [] 
    tr_len = len(train_loader)
    tst_len = len(val_loader)
    start = datetime.now()
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for b_x, b_y in (pbar := tqdm(train_loader, 
                                      desc=f'Epoch {epoch + 1}/{epochs}', 
                                      postfix=None if len(train_losses) == 0 else f't: {cust((datetime.now()- start).total_seconds())}, bst_l: {round(best_loss, 2)}, tr_l: {round(train_losses[-1], 2)}, ts_l: {round(test_losses[-1], 2)}, tr_a: {round(train_accs[-1], 1)}, ts_a: {round(test_accs[-1], 1)}',
                                      colour='green',
                                      position=0,
                                      leave=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):   # gives batch data, normalize x when iterate train_loader
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model(b_x)               # cnn output
            loss = model.loss_fnc(output, b_y)   # cross entropy loss
            acc = multi_acc(output, b_y)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            train_loss += loss.cpu().detach().numpy()
            train_acc += acc.item()
            
        train_loss /= tr_len
        train_acc /= tr_len
            
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for t_b_x, t_b_y in val_loader:
                t_b_x, t_b_y = t_b_x.to(device), t_b_y.to(device)
                test_output = model(t_b_x)
                acc = multi_acc(test_output, t_b_y)
                loss = model.loss_fnc(test_output, t_b_y)
                test_loss += loss.cpu().detach().numpy()
                test_acc += acc.item()
               
            test_loss /= tst_len
            test_acc /= tst_len
        # print(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if best_loss >= test_loss:
            best_loss_epoch = epoch
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
                
        # print('train loss: %.4f' % train_loss, '| test loss: %.4f' % test_loss, 'train acc: %.2f' % train_acc, '| test acc: %.2f' % test_acc)
        # print('train loss: %.4f' % train_loss, '| test loss: %.4f' % test_loss)
        if epoch > best_loss_epoch + patience:
            break
    
    history = [train_losses, test_losses, train_accs, test_accs]

    return history