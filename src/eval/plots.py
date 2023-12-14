import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import re
from sklearn.manifold import TSNE
from .log import log
from .load import get_max_acc_experiment, get_predictions
import warnings
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({'font.size': 14})

@log
def generate_mean_plot(path, x_data, y_data, label_ids, cmap, names):   
    fig, ax = plt.subplots(figsize=(5, 4))
    for l, c, n in zip(label_ids, cmap, names):
        avg = np.mean(np.array(x_data[np.where(y_data == l)]), axis=0)
        ax.plot(avg, label=n, c=c, linewidth=3)
    # ticks = [re.sub(u"\u2212", "-", i.get_text()) for i in ax.get_xticklabels()]
    # ax.set_xticklabels([item * 3 for item in ticks])
    plt.xlabel("Time(s)")
    plt.ylabel("WS(pm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'mean.png'), dpi=200)
    plt.close()

@log
def generate_loss_acc_plot(filename, experiment):
    fig, ax = plt.subplots(2,2, figsize=(15,8))
    plt.suptitle(experiment['name'])
    ax[0,0].set_title("Trn loss")
    ax[0,1].set_title("Val loss")
    ax[1,0].set_title("Trn Accuracy")
    ax[1,1].set_title("Val Accuracy")
    for n, exp in enumerate(experiment['experiments']):
        exp_hist = pd.read_csv(os.path.join(exp, 'history.csv'))
        ax[0,0].plot(exp_hist.loss, label=n)
        ax[0,1].plot(exp_hist.val_loss, label=n)
        ax[1,0].plot(exp_hist.accuracy, label=n)
        ax[1,1].plot(exp_hist.val_accuracy, label=n)
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,0].set_ylim((0,1))
    ax[1,1].set_ylim((0,1))
    plt.savefig(filename, dpi=200)
    plt.close()
    
@log
def generate_tr_tst_plot(filename, experiments):   
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].set_title("Trn Accuracy")
    ax[1].set_title("Val Accuracy")
    for exp in experiments:
    #     if exp['name'] not in ['resnet', 'fcn']: 
        mx_idx = get_max_acc_experiment(exp)
        exp_hist = pd.read_csv(os.path.join(exp['experiments'][mx_idx], 'history.csv'))
        ax[0].plot(exp_hist.accuracy, label=exp['name'], alpha=.5)
        ax[1].plot(exp_hist.val_accuracy, label=exp['name'], alpha=.5)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlim((0, 400))
    ax[1].set_xlim((0, 400))
    plt.savefig(filename, dpi=200)
    plt.close()

@log
def generate_tst_pred_plot(experiment, x_test, y_test, labels, cmap=None, names=None):
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(len(labels),1, figsize=(5,len(labels) * 4))
    # fig.suptitle("Test predictions")
    for i in range(label_count):
        for j in range(label_count):
            label = names[j] if names is not None else labels[j]
            ax[i].plot([], c=cmap[j], label=label)
        ax[i].legend()
        label = names[i] if names is not None else labels[i]
        ax[i].set_title(label)
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('WS(pm)')

    for n in range(x_test.shape[0]):
        ax[y_test[n]].plot(x_test[n,:], c=cmap[y_pred[n]])
        
    for i in range(label_count):
        ticks = [int(re.sub(u"\u2212", "-", i.get_text())) for i in ax[i].get_xticklabels()]
        ax[i].set_xticklabels([item * 3 for item in ticks])
    plt.tight_layout()
    plt.savefig(os.path.join(experiment, 'tst-predictions-types.png'), dpi=200)
    plt.close()

@log
def generate_preds_plot(path, x_test, y_test, y_pred, labels, names=None):
    label_count = len(labels)
    cmap = ['r', 'g']
    fig, ax = plt.subplots(len(labels),1, figsize=(5,len(labels) * 4))
    for i in range(label_count):
        for n, label in enumerate(['false', 'true']):
            ax[i].plot([], c=cmap[n], label=label)
        ax[i].legend()
        label = names[i] if names is not None else labels[i]
        ax[i].set_title(label)
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('WS(pm)')

    for m in range(x_test.shape[0]):
        ax[y_test[m]].plot(x_test[m, :], c=cmap[int(y_pred[m] == y_test[m])])
        
    for i in range(label_count):
        ticks = [int(re.sub(u"\u2212", "-", i.get_text())) for i in ax[i].get_xticklabels()]
        ax[i].set_xticklabels([item * 3 for item in ticks])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'tst-predictions.png'), dpi=200)
    plt.close()

@log
def generate_test_hist_plot(path, x_test, y_test, labels, label_ids, cmap=None, names=None):
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    x_test_lst = x_test[:, -1]
    separated_x_test = [ x_test[np.where(y_test == n)] for n in label_ids]
    bins_type = []
    for x_test_type in separated_x_test:
        bins, bin_edges = np.histogram(x_test_type[:, -1], bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
        bins_type.append(bins)
    bins, bin_edges = np.histogram(x_test_lst, bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
    fig, ax = plt.subplots(figsize=(5,4))
    for m in label_ids:
        label = names[m] if names is not None else labels[m]
        ax.bar(0, 0, width=0, color=cmap[m], label=label)
    for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        bottom = 0
        for m in label_ids:
            ax.bar(l_edge, bins_type[m][n], width=(r_edge - l_edge), color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
            bottom += bins_type[m][n]
    ax.set_xlabel("WS(pm)")
    ax.set_ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'test-hist.png'), dpi=200)
    plt.close()

@log
def generate_test_type_hist_plot(path, x_test, y_test, y_pred, labels, label_ids, cmap=None, names=None, small=False, add_lines=True):
    label_count = len(labels)
    sz = (5, label_count * 4)
    if small:
        sz = (4, label_count * 3)
    x_test_lst = x_test[:, -1]
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    fig, ax = plt.subplots(label_count, 1, figsize=sz)
    # fig.suptitle("Test predictions histogram")
    for i, (name, tag) in enumerate(zip(names, label_ids)):
        label_ids_l = label_ids[i:] + label_ids[:i]
        label_slice = np.where(y_test == tag)
        x_test_type = x_test[label_slice]
        x_test_type_lst = x_test_type[:, -1]
        # y_test_type = y_test[label_slice]
        y_pred_type = y_pred[label_slice]
        bins_type = []
        for n in label_ids:
            bins, bin_edges = np.histogram(x_test_type_lst[np.where(y_pred_type == n)], bins=50, range=(min(x_test_lst), max(x_test_lst)), density=False)
            bins_type.append(bins)
        bins, bin_edges = np.histogram(x_test_type_lst, bins=50, range=(min(x_test_lst), max(x_test_lst)), density=False)
        ax[tag].set_title(name, fontsize=12)
        for m in label_ids:
            label = names[m] if names is not None else labels[m]
            container = ax[tag].bar(0, 0, width=0, color=cmap[m], label=label)
        for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            width = r_edge - l_edge
            bottom = 0
            for m in label_ids_l:
                container = ax[tag].bar(l_edge + (width/2), bins_type[m][n], width=width, color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
                bottom += bins_type[m][n]
        if add_lines:
            mx = max(bins) * 0.12
            dff = mx / (len(label_ids) + 2)
            for m in label_ids:
                sample = x_test_type_lst[np.where(y_pred_type == m)]
                if sample.shape[0] != 0:
                    line = list(range(int(np.min(sample)), int(np.max(sample))))
                    if len(line) != 0:
                        ax[tag].plot(line, [-((m+1) * dff)] * len(line), color=cmap[m])
                        ax[tag].scatter([min(line), max(line)], [-((m+1) * dff), -((m+1) * dff)], color=cmap[m], s=10)
                    else:
                        ax[tag].scatter([int(np.min(sample))], [-((m+1) * dff)], color=cmap[m], s=10)
                        
            ax[tag].set_ylim(-(max(bins) * .12), max(bins) * 1.05)
        width = np.max(x_test_lst) - np.min(x_test_lst)
        ax[tag].set_xlim(np.min(x_test_lst) - .02 * width, np.max(x_test_lst) + .02 * width)
        ax[tag].legend()
        ax[tag].set_xlabel("WS(pm)", fontsize=12)
        ax[tag].set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    props = list(filter(lambda a: a != "", ['small' if small else '', 'ranges' if add_lines else '']))
    ext = f"({','.join(props)})" if len(props) > 0 else ""
    plt.savefig(os.path.join(path, f'tst-types-hist{ext}.png'), dpi=300)
    plt.close()
    
@log
def generate_conf_matrix(path, pred_labels, test_labels, labels, names=None):
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=names if names is not None else labels)
    pl = disp.plot(cmap=mpl.cm.Blues, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'conf-matrix.png'), pad_inches=5, dpi=200)
    plt.close()
    
@log
def generate_conf_graph(path, pred_labels, test_labels, labels, label_ids, cmap=None, names=None):
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    label_count = len(labels)
    if cmap is None:
        cmap = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    radius = 1
    G = nx.DiGraph(edge_layout='curved')
    
    text_pos = []
    
    for i in label_ids:
        theta = 2 * np.pi * i / len(label_ids)
        if len(label_ids) > 2:
            theta += np.pi / 2
        G.add_node(
            label_ids[i],
            pos=((radius * np.cos(theta), radius * np.sin(theta))),
            color=cmap[i],
            weight=round(cm[i,i] * 7000),
                  )
        if theta % (2*np.pi) >= 0 and theta % (2*np.pi) <= np.pi:
            text_pos.append((radius * np.cos(theta), radius * np.sin(theta) + .3))
        else:
            text_pos.append((radius * np.cos(theta), radius * np.sin(theta) - .3))

    for i in label_ids:
        for j in label_ids:
            if i == j:
                continue
            G.add_edge(i,j,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[i],
                        weight = cm[i, j] * 30,
                       )
            G.add_edge(j,i,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[j],
                        weight = cm[j, i] * 30,
                       )

    edges = G.edges()
    pos = list(nx.get_node_attributes(G, 'pos').values())
    node_colors = list(nx.get_node_attributes(G, 'color').values())
    node_weights = list(nx.get_node_attributes(G, 'weight').values())
    edge_colors = list(nx.get_edge_attributes(G, 'color').values())
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())


    # Draw nodes and edges
    plt.figure(figsize=(10,10))
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_weights, 
        node_color=node_colors,
    #     edgecolors='black'
    )
    edges = nx.draw_networkx_edges(
        G, pos,
        node_size=node_weights,
        edge_color=edge_colors,
        width=edge_weights,
        connectionstyle="arc3,rad=0.1",
        arrowstyle='-'
    )
    nx.draw_networkx_labels(
        G, text_pos, 
        labels={n: label for n, label in enumerate(names if names is not None else labels)},

    )
    plt.gca().set_frame_on(False)
    plt.xlim((-1.6,1.6))
    plt.ylim((-1.6,1.6))
    plt.savefig(os.path.join(path, 'conf-graph.png'), dpi=200)
    plt.tight_layout()
    plt.close()

@log
def generate_tsne_graph(path, outputs, label_ids, tags, cmap, names):
    tsne = manifold.TSNE(n_components=2, init='pca')
    outputs_tsne = tsne.fit_transform(outputs)

    fig = plt.figure()
    for i in label_ids:
        indices = [ n for n,t in enumerate(tags) if t == i ]
        plt.scatter(outputs_tsne[indices, 0], outputs_tsne[indices, 1], c=cmap[i], label=names[i], alpha=.8)
    plt.legend(ncol=4, bbox_to_anchor=(0.985, -0.075))
    plt.xlim((-60,60))
    plt.ylim((-60,60))
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'tsne-graph.png'), dpi=300)
    plt.close()

@log
def generate_rcc_plot(path, y_test, y_pred, labels, label_ids, cmap, names):
    Y = label_binarize(y_test, classes=label_ids)

    precision = dict()
    recall = dict()
    fig = plt.figure()
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_pred[:, i])
        plt.plot(recall[i], precision[i], lw=2, c=cmap[i], label=names[i])
            
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(os.path.join(path, 'rcc.png'), dpi=300)
    plt.close()

@log
def generate_roc_plot(path, y_test, y_pred, labels, label_ids, cmap, names):
    Y = label_binarize(y_test, classes=label_ids)

    fpr = dict()
    tpr = dict()
    fig = plt.figure()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_pred[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, c=cmap[i], label=names[i])
            
    plt.xlabel("true positive rate")
    plt.ylabel("false positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.savefig(os.path.join(path, 'roc.png'), dpi=300)
    plt.close()

def _plot_tsne(ax, outputs, label_ids, tags, cmap, names, add_legend=False):
    tsne = TSNE(n_components=2, init='pca')
    outputs_tsne = tsne.fit_transform(outputs)

    for i in label_ids:
        indices = [n for n, t in enumerate(tags) if t == i]
        s = [3.5 for _ in range(len(indices))]

        if add_legend:
            ax.scatter(
                outputs_tsne[indices, 0],
                outputs_tsne[indices, 1],
                c=cmap[i],
                alpha=0.8,
                s=s,
                label=names[i]
            )
        else:
            ax.scatter(
                outputs_tsne[indices, 0],
                outputs_tsne[indices, 1],
                c=cmap[i],
                alpha=0.8,
                s=s,
            )


def _plot_rcc(ax, y_test, y_pred, labels, label_ids, cmap, names, add_legend=False):
    Y = label_binarize(y_test, classes=label_ids)

    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_pred[:, i])

        if add_legend:
            ax.plot(
                recall[i],
                precision[i],
                lw=2,
                c=cmap[i],
                label=names[i]
            )
        else:
            ax.plot(
                recall[i],
                precision[i],
                lw=2,
                c=cmap[i]
            )

def _plot_conf(ax, pred_labels, test_labels, labels, names):
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    cm = np.round(cm, 2)
    disp = ConfusionMatrixDisplay(cm, display_labels=names if names is not None else labels)
    pl = disp.plot(ax=ax, cmap=mpl.cm.Blues, xticks_rotation=40, colorbar=False)

@log
def generate_tsne_graph_multi(path, params):
    fig, axs = plt.subplots(int(len(params) / 2), 2, figsize=(8, int(len(params) / 2) * 3), sharex=True, sharey=True)
    if len(params) / 2 == 1:
        axs = [axs]
    for n, (outputs, label_ids, tags, cmap, names) in enumerate(params):
        _plot_tsne(axs[int(n / 2)][n % 2], outputs, label_ids, tags, cmap, names)
    # plt.legend(ncol=4, bbox_to_anchor=(0.985, -0.075))
    # plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()

@log
def generate_rcc_graph_multi(path, params):
    fig, axs = plt.subplots(int(len(params) / 2), 2, figsize=(8, int(len(params) / 2) * 3), sharex=True, sharey=True)
    if len(params) / 2 == 1:
        axs = [axs]
    for n, (y_test, y_pred, labels, label_ids, cmap, names) in enumerate(params):
        _plot_rcc(axs[int(n / 2)][n % 2], y_test, y_pred, labels, label_ids, cmap, names)
        if n % 2 == 0:
            axs[int(n / 2)][n % 2].set_ylabel('Precision')
        if int(n / 2) == len(axs) - 1:
            axs[int(n / 2)][n % 2].set_xlabel('Recall')
        
    plt.savefig(path, dpi=500)
    plt.close()

@log
def generate_conf_matrix_multi(path, params):
    fig, axs = plt.subplots(int(len(params) / 2), 2, figsize=(10, int(len(params) / 2) * 5), sharex=True, sharey=True)
    if len(params) / 2 == 1:
        axs = [axs]
    for n, (pred_labels, test_labels, labels, names) in enumerate(params):
        _plot_conf(axs[int(n / 2)][n % 2], pred_labels, test_labels, labels, names)
    plt.tight_layout()
    plt.savefig(path, pad_inches=5, dpi=500)
    plt.close()

@log
def generate_combined_plot(path, params):
    fig, axs = plt.subplots(len(params), 3, figsize=(15, len(params) * 4.5), sharex='col', sharey='col')

    for n, (outputs, y_test, y_pred, test_labels, pred_labels, labels, label_ids, tags, cmap, names) in enumerate(params):
        _plot_conf(axs[n][0], pred_labels, test_labels, labels, names)
        _plot_tsne(axs[n][1], outputs, label_ids, tags, cmap, names)
        _plot_rcc(axs[n][2], y_test, outputs, labels, label_ids, cmap, names, add_legend=True if n == 0 else False)

        axs[n][1].set_ylabel('tSNE1')
        axs[n][2].set_ylabel('Precision')

        if n < len(params) - 1:
            axs[n][0].set_xlabel('')
        else:
            axs[n][1].set_xlabel('tSNE2')
            axs[n][2].set_xlabel('Recall')

    # Add a single legend at the bottom of the figure
    plt.figlegend(loc='lower center', bbox_to_anchor=(0.7, 0), ncol=4)

    plt.tight_layout(w_pad=0.5)
    plt.savefig(path, pad_inches=5, dpi=500)
    plt.close()