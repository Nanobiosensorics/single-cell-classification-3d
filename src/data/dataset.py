import numpy as np
from torch.utils.data import Dataset
import json
import os
import pandas as pd
import csv
import tiffile as tif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import random
import matplotlib
from sklearn.model_selection import KFold
import pickle

# For a new value new_value, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sample_variance)
    
def calculate_ids_by_time(path, tp, data_type, time=90):
    time = int(time * 60 / 9)

    indices = []
    if data_type in ['max', 'int']:
        pth1 = os.path.join(path, tp, f'{tp}_{data_type}_signals.csv')

        if not os.path.exists(pth1):
            return None

        with open(pth1, 'r') as mp:
            reader1 = csv.reader(mp)
            for n, mp_line in enumerate(reader1):
                mp_line = np.array(list(map(float, mp_line)))
                if mp_line.shape[0] >= time:
                    line_max = 800 if mp_line.shape[0] >= 800 else mp_line.shape[0]
                    if np.max(mp_line[:line_max]) > 50:
                        indices.append(n)
    else:
        pth1 = os.path.join(path, tp, data_type)

        if not os.path.exists(pth1):
            return None

        files = list(sorted(os.listdir(pth1)))
        for file in files:
            idx = file.split('_')[-1][:-5]
            d = np.asarray(tif.imread(os.path.join(pth1, file)))
            if d.shape[0] >= time:
                line_max = 800 if d.shape[0] >= 800 else d.shape[0]
                if np.max(d[:line_max]) > 50:
                    indices.append(idx)

    if len(indices) > 0:
        return indices
    else:
        return None

    
def calculate_scaler_line(path, mask_filter, types, data, label, time):
    count, mean, M2, mn, mx = 0, np.zeros((time, )), np.zeros((time, )), np.full((time, ), 1000), np.zeros((time, ))
    for tp in types:
        data_filter = [idx for idx, l in zip(data, label) if l == tp]
        data_filter = sorted(data_filter)
        with open(os.path.join(path, tp, tp + mask_filter), 'r') as fp:
            reader = csv.reader(fp)
            for n, line in enumerate(reader):
                if n in data_filter:
                    mp_line = np.array(list(map(float, line)))
                    (count, mean, M2) = update((count, mean, M2), mp_line[:time])
                    mn = np.min(np.asarray([mp_line[:time], mn]), axis=0)
                    mx = np.max(np.asarray([mp_line[:time], mx]), axis=0)
    mean, variance, sample_variance = finalize((count, mean, M2))
    return {'mean': mean, 'std': np.sqrt(variance), 'min': mn, 'max': mx }

def calculate_scaler_block(path, mask_filter, size, types, data, label, time):
    count, mean, M2, mn, mx = 0, np.zeros((time, size, size)), np.zeros((time, size, size)), np.full((time, size, size), 1000), np.zeros((time, size, size))
    for tp in types:
        data_filter = [idx for idx, l in zip(data, label) if l == tp]
        data_filter = sorted(data_filter)
        for idx in data_filter:
            block = np.asarray(tif.imread(os.path.join(path, tp, mask_filter, f'image_{idx}.tiff'))).astype(np.float32)[:time]
            (count, mean, M2) = update((count, mean, M2), block[:time])
            mn = np.min(np.asarray([block[:time], mn]), axis=0)
            mx = np.max(np.asarray([block[:time], mx]), axis=0)
    mean, variance, sample_variance = finalize((count, mean, M2))
    return {'mean': mean, 'std': np.sqrt(variance), 'min': mn, 'max': mx }

def calculate_scaler_params(path, data_type, types, data, label, time=90):
    time = int(time * 60 / 9)
    stats = {}
    if data_type == 'max':
        stats['max'] = calculate_scaler_line(path, '_max_signals.csv', types, data, label, time)
    elif data_type == 'int':
        stats['int'] = calculate_scaler_line(path, '_int_signals.csv', types, data, label, time)
    else:
        stats[data_type] = calculate_scaler_block(path, data_type, 8, types, data, label, time)
    return stats
        
def repeat_list_to_length(lst, desired_length):
    repeated_list = []
    while len(repeated_list) < desired_length:
        repeated_list.extend(lst)
    return repeated_list[:desired_length]

def split_dataset(samples_dict, val_ratio = .2, upsample = True):
    X_train, y_train, X_test, y_test = [], [], [], []

    max_len = max([int(len(s) * (1 - val_ratio)) for s in samples_dict.values()])

    for t, s in samples_dict.items():
        shuffled_indices = np.random.permutation(len(s))
        train_indices, test_indices = train_test_split(shuffled_indices, test_size=val_ratio, random_state=42)
        train = list(np.array(s)[train_indices])
        train_extended = repeat_list_to_length(train, max_len) if upsample else train
        X_train.extend(train_extended)
        y_train.extend([t] * len(train_extended))
        X_test.extend(list(np.array(s)[test_indices]))
        y_test.extend([t] * len(test_indices))

    train = list(zip(X_train, y_train))
    random.shuffle(train)
    X_train, y_train = zip(*train)
    return X_train, y_train, X_test, y_test

def split_dataset_kfold(samples_dict, kfold = 5, upsample = True):
    X_train, y_train, X_test, y_test = [[] for _ in range(kfold)], [[] for _ in range(kfold)], [[] for _ in range(kfold)], [[] for _ in range(kfold)]

    max_len = max([int(len(s) * ((kfold - 1) / kfold)) for s in samples_dict.values()])

    for t, s in samples_dict.items():
        shuffled_indices = np.random.permutation(len(s))
        skf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        for i, (train_indices, test_indices) in enumerate(skf.split(shuffled_indices)):
            train = list(np.array(s)[train_indices])
            train_extended = repeat_list_to_length(train, max_len) if upsample else train
            X_train[i].extend(train_extended)
            y_train[i].extend([t] * len(train_extended))
            X_test[i].extend(list(np.array(s)[test_indices]))
            y_test[i].extend([t] * len(test_indices))

    for i in range(kfold):
        train = list(zip(X_train[i], y_train[i]))
        random.shuffle(train)
        X_train[i], y_train[i] = zip(*train)
        yield X_train[i], y_train[i], X_test[i], y_test[i]

class BaseDataset(Dataset):
    def __init__(self, pickle_path, tp, partition, normalize, cross_validation=None):
        super().__init__()
                  
        if not os.path.exists(pickle_path):
            raise ValueError('File not found!')
        
        self.pickle_path = pickle_path

        fp = open(self.pickle_path, 'rb')
        dataset = pickle.load(fp)

        if cross_validation != None:
            dataset = dataset[cross_validation]
        
        self.time = int(dataset['time'] * 60 / 9)
        self.normalize = normalize
        if self.normalize == 'norm':
            self.norm = (dataset[tp]['min'], dataset[tp]['max'])
        elif self.normalize == 'std':
            self.norm = (dataset[tp]['mean'], dataset[tp]['std'])
        self.types = list(dataset[partition][f'y_{partition}'])
        self.indices = list(map(int, dataset[partition][f'X_{partition}']))
        self.type_set = sorted(set(self.types))
        self.encoder = LabelEncoder()
        self.encoder.fit(self.type_set)
        self.types_encoded = self.encoder.transform(self.types)
        # self.types_encoded = np.squeeze(self.types_encoded)

        fp.close()
        
    def __getitem__(self, idx):
        return self.indices[idx], self.types_encoded[idx]

    def __len__(self):
        return len(self.indices)
    
    def transform(self, data):
        if self.normalize == 'norm':
            return np.divide((data - self.norm[0]), (self.norm[1] - self.norm[0]), 
                                out=np.zeros_like((data - self.norm[0])), where=(self.norm[1] - self.norm[0])!=0)
        elif self.normalize == 'std':
            return np.divide((data - self.norm[0]), (self.norm[1]), 
                                out=np.zeros_like((data - self.norm[0])), where=self.norm[1]!=0)
        else:
            return data
        
    def inverse_transform(self, data):
        if self.normalize == 'norm':
            return (data * (self.norm[1] - self.norm[0])) + self.norm[0]
        elif self.normalize == 'std':
            return (data * self.norm[1]) + self.norm[0]
        else:
            return data


class CsvDataset(BaseDataset):
    def __init__(self, folder_path, pickle_path, tp = 'max', partition = 'train', normalize = 'norm', cross_validation=None):
        super().__init__(pickle_path, tp, partition, normalize, cross_validation)

        self.data = np.zeros((len(self.indices), self.time))
        for t in self.type_set:
            with open(os.path.join(folder_path, t, f'{t}_{tp}_signals.csv')) as fp:
                reader = csv.reader(fp)
                for n, line in enumerate(reader):
                    for m, i in enumerate(self.indices):
                        if n == i and self.types[m] == t:
                            self.data[m] = np.array(list(map(float, line)))[:self.time]
        if normalize:
            self.data = self.transform(self.data, self.normalize)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.types_encoded[idx]

class VideoDataset(BaseDataset):
    def __init__(self, folder_path, pickle_path, tp = 'im_cardio', partition = 'train', normalize = 'norm', cross_validation=None):
        super().__init__(pickle_path, tp, partition, normalize, cross_validation)
        self.folder_path = folder_path
        self.image_type = tp

    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.types[idx],
                            self.image_type, f'image_{self.indices[idx]}.tiff')
        data = np.asarray(tif.imread(path))[:self.time]
        if self.normalize:
            data = self.transform(data)
        return np.expand_dims(data, 0).astype(np.float32), self.types_encoded[idx]