import argparse
import os
import pandas as pd
from .plots import *
from .load import get_colors
from torch import nn
import torch
from ..classifier.video.cellnet import get_model
from ..data.dataset import VideoDataset, CsvDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, mean_squared_error, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_tests, y_preds, outputs, labels, label_ids):
    accuracy = accuracy_score(y_tests, y_preds)
    precision = precision_score(y_tests, y_preds, average='macro')
    recall = recall_score(y_tests, y_preds, average='macro')
    f1 = f1_score(y_tests, y_preds, average='macro')
    mse = mean_squared_error(y_tests, y_preds)
    ll = log_loss(y_tests, outputs)
    cel = nn.functional.cross_entropy(torch.from_numpy(outputs), torch.from_numpy(y_tests)).detach().numpy()
    Y = label_binarize(y_tests, classes=label_ids)
    auc = roc_auc_score(Y, outputs, multi_class='ovo')
    aucpr = average_precision_score(y_tests, outputs)
    return [accuracy, precision, recall, f1, mse, cel, ll, auc, aucpr]

def __run_eval(result_path, model, test_dl, device):
    # trains = []
    x_tests = []
    y_tests = []
    y_preds = []
    y_loss = []
    y_test_labels = []
    y_pred_labels = []

    # model_path = result_path.replace('gt-results/', '')
    # print(result_path, model_path)
    model.load_state_dict(torch.load(os.path.join(result_path, f'model.pt')))

    model.eval()
    model.to(device)

    outputs = []
    # colors = []
    tags = []

    for b_x, b_y in test_dl:
        b_x, b_y = b_x.to(device), b_y.to(device)

        y_pred = model(b_x)
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        outputs.append(y_pred.cpu().detach().numpy())
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

        for t, y in zip(b_y, y_pred_tags):
            # colors.append('green' if t == y else 'red')
            # colors.append(cmap[t])
            tags.append(t)

        inv_b_x = test_dl.dataset.inverse_transform(b_x.cpu().detach().numpy())

        x_tests.extend(list(np.max(inv_b_x[:, 0], axis=(2,3))))
        y_tests.extend(b_y.cpu().detach().numpy())
        y_preds.extend(y_pred_tags.cpu().detach().numpy())

    x_tests = np.asarray(x_tests)
    y_tests = np.asarray(y_tests)
    y_preds = np.asarray(y_preds)
    outputs = np.vstack(outputs)
    y_test_labels = test_dl.dataset.encoder.inverse_transform(y_tests)
    y_pred_labels = test_dl.dataset.encoder.inverse_transform(y_preds)
    
    return x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags

def run_eval_single(path, result_path, model_type, cell_types, time, color_path, seg_type, normalize):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_name = '-'.join([*sorted(list(cell_types)), str(time), seg_type])
    result_path = os.path.join(result_path, model_type, file_name)
    labels = list(sorted(cell_types))
    label_ids = list(range(len(labels)))
    cmap, names = get_colors(color_path, labels)

    val_ds = VideoDataset(path, os.path.join(path, f'{file_name}-dataset.pkl'), partition='test', normalize=normalize, tp=seg_type)
    val_dl = DataLoader(val_ds, 8, shuffle=False)
    model = get_model(model_type, nn.CrossEntropyLoss(), len(val_ds.type_set), int(val_ds.time * 9 / 60))
    # print(model)
    
    x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags = __run_eval(result_path, model, val_dl, device)
    return  x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags

def run_eval_multi(path, params_list, cell_types, time, normalize, color_path):
    labels = list(sorted(cell_types))
    label_ids = list(range(len(labels)))
    cmap, names = get_colors(color_path, labels)
    results = []
    for args in params_list:
        results.append(run_eval_single(args['path'], args['result_path'], args['model_type'], cell_types, time, color_path, args['seg_type'], normalize))
    # tsne_params = [[outputs, label_ids, tags, cmap, names] for x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags in results]
    # generate_tsne_graph_multi(f'{os.path.join(path, "-".join([*labels, model_type]))}_tsne_multi.jpg', tsne_params)
    # rcc_params = [[y_tests, outputs, labels, label_ids, cmap, names] for x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags in results]
    # generate_rcc_graph_multi(f'{os.path.join(path, "-".join([*labels, model_type]))}_rcc_multi.jpg', rcc_params)
    # conf_params = [[y_pred_labels, y_test_labels, labels, names] for x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags in results]
    # generate_conf_matrix_multi(f'{os.path.join(path, "-".join([*labels, model_type]))}_conf_matrix_multi.jpg', conf_params)
    # generate_conf_graph(result_path, y_pred_labels, y_test_labels, labels, label_ids, cmap, names)
    combo_params = [[outputs, y_tests, y_preds, y_test_labels, y_pred_labels, labels, label_ids, tags, cmap, names] for x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags in results]
    generate_combined_plot(f'{os.path.join(path, "-".join(labels))}-combo.jpg', combo_params)

def run_eval(path, result_path, model_type, cell_types, time, color_path, seg_type, normalize, kfold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if kfold <= 0:
        raise ValueError()
    
    file_name = '-'.join([*sorted(list(cell_types)), str(time), seg_type])

    if kfold == 1:
        result_path = os.path.join(result_path, model_type, file_name)
        labels = list(sorted(cell_types))
        label_ids = list(range(len(labels)))
        cmap, names = get_colors(color_path, labels)

        val_ds = VideoDataset(path, os.path.join(path, f'{file_name}-dataset.pkl'), partition='test', normalize=normalize, tp=seg_type)
        val_dl = DataLoader(val_ds, 8, shuffle=False)
        model = get_model(model_type, nn.CrossEntropyLoss(), len(val_ds.type_set), int(val_ds.time * 9 / 60))
        # print(model)
        
        x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags = __run_eval(result_path, model, val_dl, device)
        stats = calculate_metrics(y_tests, y_preds, outputs, labels, label_ids)

        pd.DataFrame([stats]).to_csv(os.path.join(result_path, 'stats.csv'), header=['accuracy', 'precision', 'recall', 'f1-score', 'mse', 'cross-entropy', 'll', 'auc', 'aucpr'], index=None)

        generate_mean_plot(result_path, x_tests, y_tests, label_ids, cmap, names)
        generate_test_hist_plot(result_path, x_tests, y_tests, labels, label_ids, cmap, names)
        generate_test_type_hist_plot(result_path, x_tests, y_tests, y_preds, labels, label_ids, cmap, names, small=True, add_lines=True)
        generate_conf_matrix(result_path, y_pred_labels, y_test_labels, labels, names)
        generate_conf_graph(result_path, y_pred_labels, y_test_labels, labels, label_ids, cmap, names)
        generate_tsne_graph(result_path, outputs, label_ids, tags, cmap, names)
        generate_rcc_plot(result_path, y_tests, outputs, labels, label_ids, cmap, names)
        generate_roc_plot(result_path, y_tests, outputs, labels, label_ids, cmap, names)

    else:
        labels = list(sorted(cell_types))
        label_ids = list(range(len(labels)))
        cmap, names = get_colors(color_path, labels)
        kfold_stats = []
        kfold_tests = []
        kfold_outputs = []
        kfold_preds = []
        for i in range(kfold):
            result_path = os.path.join(result_path, model_type, file_name, f'cv{i}')
            
            val_ds = VideoDataset(path, os.path.join(path, f'{file_name}-cv-dataset.pkl'), partition='test', normalize=normalize, tp=seg_type, cross_validation=f'cv{i}')
            val_dl = DataLoader(val_ds, 8, shuffle=False)
            model = get_model(model_type, nn.CrossEntropyLoss(), len(val_ds.type_set), int(val_ds.time * 9 / 60))
            # print(model)
            
            x_tests, y_tests, y_preds, outputs, y_test_labels, y_pred_labels, tags = __run_eval(result_path, model, val_dl, device)
            stats = calculate_metrics(y_tests, y_preds, outputs, labels, label_ids)

            pd.DataFrame([stats]).to_csv(os.path.join(result_path, 'stats.csv'), header=['accuracy', 'precision', 'recall', 'f1-score', 'mse', 'cross-entropy', 'll', 'auc', 'aucpr'], index=None)

            generate_mean_plot(result_path, x_tests, y_tests, label_ids, cmap, names)
            generate_test_hist_plot(result_path, x_tests, y_tests, labels, label_ids, cmap, names)
            generate_test_type_hist_plot(result_path, x_tests, y_tests, y_preds, labels, label_ids, cmap, names, small=True, add_lines=True)
            generate_conf_matrix(result_path, y_pred_labels, y_test_labels, labels, names)
            generate_conf_graph(result_path, y_pred_labels, y_test_labels, labels, label_ids, cmap, names)
            generate_tsne_graph(result_path, outputs, label_ids, tags, cmap, names)
            generate_rcc_plot(result_path, y_tests, outputs, labels, label_ids, cmap, names)
            generate_roc_plot(result_path, y_tests, outputs, labels, label_ids, cmap, names)

            if len(kfold_tests) == 0:
                kfold_tests = y_tests
                kfold_preds = y_preds
                kfold_outputs = outputs
            else:
                kfold_tests = np.concatenate([kfold_tests, y_tests])
                kfold_preds = np.concatenate([kfold_preds, y_preds])
                kfold_outputs = np.vstack([kfold_outputs, outputs])
            kfold_stats.append(stats)
        
        snapshot = np.asarray(kfold_stats).copy()
        
        kfold_stats.append(np.mean(snapshot, axis=0))
        kfold_stats.append(np.std(snapshot, axis=0))

        kfold_stats.append(calculate_metrics(kfold_tests, kfold_preds, kfold_outputs, labels, label_ids))
        
        pd.DataFrame(kfold_stats).to_csv(os.path.join(result_path, '..', 'kfold_stats.csv'), header=['accuracy', 'precision', 'recall', 'f1-score', 'mse', 'cross-entropy', 'll', 'auc', 'aucpr'], index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Cell Prediction Evaluator')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-r', '--result_path', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, choices=['cnn', 'resnet', 'cnn-lstm', 'densenet'], default='cnn')
    parser.add_argument('-ct', '--cell_types', nargs='+', required=True)
    parser.add_argument('-tm', '--time', type=int, required=True)
    parser.add_argument('-c', '--color_path', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, choices=['max', 'int', 'im_cardio', 'im_cover', 'im_watershed', 'im_pred'], default='im_cardio')
    parser.add_argument('--normalize', type=str, choices=['norm', 'std'], default='std')
    parser.add_argument('-k', '--kfold', type=int, default=0)

    args = parser.parse_args()
    run_eval(args.path, args.result_path, args.model_type, args.cell_types, args.time, args.color_path, args.type, args.normalize, args.kfold)