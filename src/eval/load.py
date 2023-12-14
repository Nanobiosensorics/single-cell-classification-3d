import os
import numpy as np
import pandas as pd

def get_max_acc_experiment(experiment):
    mx_idx = 0
    max_accs = 0
    for n, exp in enumerate(experiment['experiments']):
        exp_hist = pd.read_csv(os.path.join(exp, 'history.csv'))
        mx = max(exp_hist.val_accuracy)
        if max_accs < mx:
            mx_idx = n
            max_accs = mx
    return mx_idx

def get_predictions(experiment, labels):
    preds = pd.read_csv(os.path.join(experiment, 'true-pred-values.csv'))
    y_pred = np.array(preds.pred)
    pred_labels = [labels[y_pred[n]] for n in range(y_pred.shape[0])]
    return y_pred, pred_labels


def get_data(path, labels):
    x_train = np.load(os.path.join(path, 'X_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    x_test = np.load(os.path.join(path, 'X_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    x_data = np.vstack([x_train, x_test])
    y_data = np.concatenate([y_train, y_test])
    data_labels = [labels[y_data[n]] for n in range(y_data.shape[0])]
    return x_data, y_data, data_labels

def get_test(path, labels):
    x_test = np.load(os.path.join(path, 'X_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    test_labels = [labels[y_test[n]] for n in range(y_test.shape[0])]
    return x_test, y_test, test_labels

def get_dictionary(path):
    dictionary = pd.read_csv(os.path.join(path, '../dictionary.csv'))
#     labels = list(map(lambda x: x.upper(), list(dictionary.iloc[:, 0])))
    labels = list(dictionary.iloc[:, 0])
    label_ids = list(dictionary.iloc[:, 1])
    return labels, label_ids

def get_colors(path, labels):
    df = pd.read_csv(os.path.join(path, 'color_map.csv'))
    colors = []
    names = []
    for label in labels:
        for i in range(df.shape[0]):
            if df.iloc[i, 0] == label:
                colors.append(df.iloc[i, 1])
                names.append(df.iloc[i, 2])
                break
    return colors, names if len(colors) == len(labels) else None