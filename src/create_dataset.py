import pandas as pd
from data.dataset import split_dataset, calculate_ids_by_time, calculate_scaler_params, split_dataset_kfold
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import json
import pickle

def __create_save_dataset(path, file_name, data_type, types_filter, time, samples, upsample):
    
    X_train, y_train, X_val, y_val = split_dataset(samples, upsample=upsample, val_ratio=0.33)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.5, random_state=42)
    obj = calculate_scaler_params(path, data_type, types_filter, X_train, y_train, time)
    obj['time'] = time
    obj['train'] = {
        'X_train': X_train,
        'y_train': y_train,
    }
    obj['val'] = {
        'X_val': X_val,
        'y_val': y_val
    }
    obj['test'] = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    # pd.DataFrame(list(zip(y_train, X_train))).to_csv(os.path.join(path, '-'.join([file_name, 'train.csv'])), header=None, index=None)
    # pd.DataFrame(list(zip(y_val, X_val))).to_csv(os.path.join(path, '-'.join([file_name, 'val.csv'])), header=None, index=None)
    # pd.DataFrame(list(zip(y_test, X_test))).to_csv(os.path.join(path, '-'.join([file_name, 'test.csv'])), header=None, index=None)
    
    # np.save(os.path.join(path, '-'.join([file_name, 'stats.npy'])), stats)
    with open(os.path.join(path, '-'.join([file_name, 'dataset.pkl'])), 'wb') as fp:
        pickle.dump(obj, fp)

def __create_save_kfold_dataset(path, file_name, data_type, types_filter, time, samples, kfold, upsample):
    obj = {}

    for i, (X_train, y_train, X_val, y_val) in enumerate(split_dataset_kfold(samples, kfold, upsample)):
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.5, random_state=42)
        cv = calculate_scaler_params(path, data_type, types_filter, X_train, y_train, time)
        cv['time'] = time
        cv['train'] = {
            'X_train': X_train,
            'y_train': y_train,
        }
        cv['val'] = {
            'X_val': X_val,
            'y_val': y_val
        }
        cv['test'] = {
            'X_test': X_test,
            'y_test': y_test
        }
        obj[f'cv{i}'] = cv
    
    # pd.DataFrame(list(zip(y_train, X_train))).to_csv(os.path.join(path, '-'.join([file_name, 'train.csv'])), header=None, index=None)
    # pd.DataFrame(list(zip(y_val, X_val))).to_csv(os.path.join(path, '-'.join([file_name, 'val.csv'])), header=None, index=None)
    # pd.DataFrame(list(zip(y_test, X_test))).to_csv(os.path.join(path, '-'.join([file_name, 'test.csv'])), header=None, index=None)
    
    # np.save(os.path.join(path, '-'.join([file_name, 'stats.npy'])), stats)
    with open(os.path.join(path, '-'.join([file_name, 'dataset.pkl'])), 'wb') as fp:
        pickle.dump(obj, fp)
    

def create_dataset(path, data_type, times, cell_types, upsample=True, kfold=1):
    save_samples = {}
    save_counts = 0
    export = None
    

    if kfold <= 0:
        raise ValueError(f'Kfold == {kfold}')

    for time in times:
        label_counts = 0
        label_samples = {}
        for t in cell_types:
            stats = calculate_ids_by_time(path, t, data_type, time)
            if stats != None:
                label_counts += len(stats)
                label_samples[t] = stats
            else:
                print(f'No samples found for {t}')
        
        print(time, label_counts, save_counts == label_counts)
        if save_counts != label_counts:
            save_counts = label_counts
            save_samples = label_samples

        if export == None:
            export = {key:[len(value)] for key, value in label_samples.items()}
        else:
            for key, value in label_samples.items():
                export[key].append(len(value))
            
        if kfold == 1:
            file_name = '-'.join([*sorted(list(save_samples.keys())), str(time), data_type])
            __create_save_dataset(path, file_name, data_type, list(save_samples.keys()), time, save_samples, upsample)
        else:
            file_name = '-'.join([*sorted(list(save_samples.keys())), str(time), data_type, f'cv'])
            __create_save_kfold_dataset(path, file_name, data_type, list(save_samples.keys()), time, save_samples, kfold, upsample)

    with open(os.path.join(path, '-'.join([*sorted(cell_types), data_type]) + '-cell_counts.json'), 'w') as fp:
        json.dump(export, 
                fp, 
                sort_keys=False,
                indent=4,
                separators=(',', ': ')
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group measurement data by cell type')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-d', '--data_type', type=str, required=True)
    parser.add_argument('-t', '--time', nargs='+', type=int, required=True)
    parser.add_argument('-c','--cell_types', nargs='+', required=True)
    parser.add_argument('-u', '--upsample', type=bool, default=True)
    parser.add_argument('-k', '--kfold', type=int, default=1)
    
    args = parser.parse_args()
    create_dataset(args.path, args.data_type, args.time, args.cell_types, args.upsample, args.kfold)