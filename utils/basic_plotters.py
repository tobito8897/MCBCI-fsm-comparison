#!/usr/bin/python3.7
from collections import defaultdict
import numpy as np
import scipy.signal
from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from .parameters import methods_color
from .signal_processors import jaccard_similarity

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica'] + plt.rcParams['font.sans-serif']


def get_grid_spec(num_plots: int, wspace: int = None, hspace: int = None):
    if num_plots == 1:
        return (GridSpec(1, 1, wspace=wspace, hspace=hspace)[0, 0], ), (True, )
    elif num_plots == 2:
        gs = GridSpec(1, 2, wspace=wspace, hspace=hspace)
        return (gs[0, 0], gs[0, 1]), (True, False)
    elif num_plots == 3:
        gs = GridSpec(2, 2, wspace=wspace, hspace=hspace)
        return (gs[0, 0], gs[0, 1], gs[1, :]), (True, False, True)
    elif num_plots == 4:
        gs = GridSpec(2, 2, wspace=wspace, hspace=hspace)
        return (gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]), (True, False, True, False)
    elif num_plots == 5:
        gs = GridSpec(2, 3, wspace=wspace, hspace=hspace)
        return (gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]), (True, False, False, True, False)
    elif num_plots == 8:
        gs = GridSpec(2, 4, wspace=wspace, hspace=hspace)
        return (gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3],
                gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]), (True, False, False, False, True, False, False, False)
    else:
        raise Exception()


def eeg_plot_plus_label(data: np.array, label: np.array, channels: list,
                        channel_names: list = None, title: str = "",
                        period: tuple = None, fs: int = 256, save: str = None):
    """
    Plot EEG signal in time domain + label (seizure/background)
    """
    start, stop = 0, -1
    space = np.max(np.max(data))/2
    if not channel_names:
        channel_names = channels
    time = np.arange(data.shape[1])/fs

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    if period:
        start = int(period[0])*fs
        stop = int(period[1])*fs

    for channel, name, count in zip(channels, channel_names,
                                    range(len(channels))):
        _data = data[int(channel), start: stop]
        _label = label[start: stop]
        _time = time[start: stop]
        _label = label[start: stop]
        factor = np.mean(np.absolute((_data)))*2
        ax.plot(_time, _data + factor*count, label=name)
        ax.plot(_time, factor*_label + factor*count)

    ax.legend()
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def eeg_psd_plot(data: np.array, channels: list, channel_names: list = None,
                 title: str = "", period: tuple = None, fs: int = 256,
                 save: str = None):
    """
    Calculate PSD of EEG signal using periodogram
    """
    start, stop = 0, -1
    title = title
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    if period:
        start = int(period[0]*data.shape[1]/fs)
        stop = int(period[1]*data.shape[1]/fs)

    for channel, name, count in zip(channels, channel_names,
                                    range(len(channels))):
        fxx, psd = scipy.signal.periodogram(data[int(channel)],
                                            fs=fs)
        factor = np.max(psd)*0.3
        ax.plot(fxx[start: stop], count*factor + psd[start: stop], label=name)

    ax.legend()
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, classes: list,
                          _id: str, normalize: bool = True,
                          cmap=plt.cm.YlOrRd, save: str = None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    title = "Confusion matrix, top=%s, acc=%s" % (_id,
                                                  int(accuracy*1000)/1000)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, label="Number of instances")
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    fig.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def get_top_plot(columns: list, values: list, save: str = None):
    """
    Plot most important features using bar plot
    """
    _, ax = plt.subplots(figsize=(7.16, 4))
    ax.barh(columns, values)
    ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.5,
            alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylabel("Feature Name", fontsize=14)
    ax.set_xlabel("Assigned importance", fontsize=14)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def get_feature_elimination_plot(used_features: list, y_real_list: list,
                                 y_predicted_list: list, title: str,
                                 save: str = None):
    """
    Plot accuracy for different number of used features
    """
    f1s, sen, acc = [], [], []

    for y_real, y_pred in zip(y_real_list, y_predicted_list):
        local_f1, local_sen, local_acc = [], [], []
        for rep_real, rep_pred in zip(y_real, y_pred):
            cm = confusion_matrix(rep_real, rep_pred)
            local_sen.append(float(cm[0][0]/np.sum(cm[0])))
            local_acc.append(accuracy_score(rep_real, rep_pred))
            rep_real = np.where((rep_real == 0) | (rep_real == 1), 1-rep_real,
                                rep_real)
            rep_pred = np.where((rep_pred == 0) | (rep_pred == 1), 1-rep_pred,
                                rep_pred)
            local_f1.append(f1_score(rep_real, rep_pred))

        f1s.append([np.mean(local_f1), np.std(local_f1)])
        sen.append([np.mean(local_sen), np.std(local_sen)])
        acc.append([np.mean(local_acc), np.std(local_acc)])

    used_features, f1s, sen, acc = zip(*sorted(zip(used_features, f1s,
                                                   sen, acc),
                                               reverse=True))

    _ = plt.figure(figsize=(12, 7))
    _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax0.plot(used_features, [x[0] for x in f1s], "r")
    ax0.invert_xaxis()
    ax0.set_ylim([0, 1])
    ax0.set_title(title, fontsize=15)
    ax0.set_ylabel("F1-Score", fontsize=14)
    ax0.set_xlabel("Number of features", fontsize=14)
    ax0.grid(True)

    norm = plt.Normalize(-0.3, 0.9)
    color_f1s = plt.cm.hot(norm([x[0] for x in f1s]))
    color_sen = plt.cm.hot(norm([x[0] for x in sen]))
    color_acc = plt.cm.hot(norm([x[0] for x in acc]))
    colors = [(x, y, z) for x, y, z in zip(color_f1s, color_sen, color_acc)]
    f1s = [(int(a[0]*1000)/1000, int(a[1]*1000)/1000) for a in f1s]
    sen = [(int(a[0]*1000)/1000, int(a[1]*1000)/1000) for a in sen]
    acc = [(int(a[0]*1000)/1000, int(a[1]*1000)/1000) for a in acc]

    ax1.axis("off")
    cell_text = [["{} +/- {}".format(x[0], x[1]),
                  "{} +/- {}".format(y[0], y[1]),
                  "{} +/- {}".format(z[0], z[1])]
                 for x, y, z in zip(f1s, sen, acc)]
    table = ax1.table(cellText=cell_text,
                      colLabels=["F1-Score", "Sensivity", "Accuracy"],
                      rowLabels=used_features,
                      loc="center", cellColours=colors)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def get_rfe_comparison_plot(data: list, save: str = None):
    """
    Plot accuracy for different number of used features and explainers
    """
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    grid_specs, y_flags = get_grid_spec(len(data), wspace=0.08, hspace=0.25)
    fig = plt.figure(figsize=(7.16, 4))
    axes, lines, labels = [], [], []

    for (classifier, _data), grid_spec, y_flag in zip(data.items(),
                                                      grid_specs,
                                                      y_flags):
        ax = fig.add_subplot(grid_spec)
        axes.append(ax)

        for explainer in _data:
            f1_scores = []
            for real, pred in zip(explainer["real"], explainer["predicted"]):
                real = np.where((real == 0) | (real == 1), 1-real,
                                real)
                pred = np.where((pred == 0) | (pred == 1), 1-pred, pred)
                f1_scores.append(f1_score(real, pred))
            used_features, f1_scores = zip(*sorted(zip(explainer["used_features"],
                                                       f1_scores),
                                                   reverse=True))

            line = ax.plot(used_features, f1_scores,
                           color=methods_color[explainer["explainer"]],
                           label=explainer["explainer"], linewidth=1)
        ax.invert_xaxis()
        ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.5,
                alpha=0.3)
        ax.set_ylim([0.50, 0.90])
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=5)

        if not y_flag:
            ax.set_yticklabels([])

        for ytick in ax.get_yticks():
            ax.hlines(ytick, 990, 960, linewidth=0.7, colors=["k"])

    line, label = axes[0].get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

    legend = fig.legend(lines, labels, loc=(0.75, 0.22), fontsize=6,
                        title=fill("Feature selection methods", 20))
    legend.get_frame().set_edgecolor("k")
    plt.setp(legend.get_title(), fontsize=6)
    print(used_features)
    for ax, letter in zip(axes, letters):
        ax.text(int(used_features[0]/2), 0.36, letter,
                fontsize=7, fontname="Times New Roman")
        ax.set_ylim([0.45, 0.92])
        ax.set_xlim(used_features[0], 0)
    fig.text(0.43, 0.015, "Number of features", fontsize=7)
    fig.text(0.05, 0.5, "F1-Score", rotation=90, va='center', fontsize=7)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def get_scatter_plot_acc(label_slices: dict, models: list, explainers: list,
                         imp_model_first: bool, save: str = None):
    """
    Plot accuracy for different combinations of classifiers and explainers
    """
    shapes = ["*", ",", "+", "v", "^", "<", ">"]
    letters = ["(a)", "(b)", "(c)", "(d)"]
    grid_specs, y_flags = get_grid_spec(len(label_slices),
                                        wspace=0.08,
                                        hspace=0.20)
    fig = plt.figure(figsize=(7.16, 5))
    y_min_lim, y_max_lim = 1, 0
    axes, lines, labels = [], [], []

    if imp_model_first:
        models, explainers = explainers, models

    for (_, labels), grid_spec, y_flag in zip(label_slices.items(),
                                              grid_specs, y_flags):
        count = 0
        for x in labels:
            x["name"] = (x["name"][1], x["name"][0])

        # ###Compute Accuracy and Plot
        ax = fig.add_subplot(grid_spec)
        axes.append(ax)
        for idx1, m in enumerate(models):
            for idx2, e in enumerate(explainers):
                x, y = [], []
                for label in labels:
                    if m in label["name"][0] and e in label["name"][1]:
                        x.append(count)
                        real = np.concatenate(label["real"])
                        pred = np.concatenate(label["predicted"])
                        real = np.where((real == 0) | (real == 1), 1-real,
                                        real)
                        pred = np.where((pred == 0) | (pred == 1), 1-pred,
                                        pred)
                        y.append(f1_score(real, pred))
                        count += 1
                if y:
                    y_min_lim = min(y) if min(y) < y_min_lim else y_min_lim
                    y_max_lim = max(y) if max(y) > y_max_lim else y_max_lim
                ax.scatter(x, y, s=50, c=methods_color[m], marker=shapes[idx2])

        f = lambda ax, m, c, l: ax.scatter([], [], marker=m, color=c,
                                           label=l)

        colors = [methods_color[m] for m in models]
        handles = [f(ax, "o", c, n) for c, n in zip(colors, models)]
        handles = [f(ax, "o", "w", "Dummy")] + handles + [f(ax, "o", "w", "Dummy")]
        handles += [f(ax, s, "k", n) for s, n in zip(shapes, explainers)]
        legend_labels = ([fill("Feature selection", 15)] + models +
                         ["Classification"] + explainers)

        ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.5,
                alpha=0.3)
        if not y_flag:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', labelsize=5)
        ax.set_xticks([])
    
    legend = fig.legend(handles, legend_labels, framealpha=0.5,
                        fontsize=8, bbox_to_anchor=(1.1, 0.75))
    legend.get_frame().set_edgecolor("k")

    for ax, letter in zip(axes, letters):
        ax.text(len(models)*len(explainers)*0.5, y_min_lim*0.50, letter,
                fontsize=8, fontname="Times New Roman")
        ax.set_ylim([y_min_lim*0.95, y_max_lim*1.05])
        ax.set_xlim([-1, count+1])
        for ytick in ax.get_yticks():
            ax.hlines(ytick, -1, 0, linewidth=0.7, colors=["k"])


    fig.text(0.05, 0.5, "F1-Score", rotation=90, va="center", fontsize=8)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def get_scatter_plot_friedman(labels_slices: dict, save: str = None):
    """
    Plot mean accuracy of each explainer across all classifiers
    """
    letters = ["(a)", "(b)", "(c)", "(d)"]
    grid_specs, y_flags = get_grid_spec(len(labels_slices),
                                        wspace=0.08,
                                        hspace=0.25)
    fig = plt.figure(figsize=(3.5, 4.2))
    x_max_lim, y_max_lim = 0, 0
    axes, lines, labels_legend = [], [], []

    for (_, labels), grid_spec, y_flag in zip(labels_slices.items(),
                                              grid_specs, y_flags):
        global_rank = defaultdict(list)
        num_classifiers = len(next(iter(labels.values())))

        # ####### Compute accuracy
        for f_selector, predictions in labels.items():
            for idx, prediction in enumerate(predictions):
                labels[f_selector][idx] = accuracy_score(np.concatenate(prediction["real"]),
                                                         np.concatenate(prediction["predicted"]))
        # ####### Compute ranking position
        for idx in range(num_classifiers):
            local_acc = []
            local_f_selectors = []
            local_rank = []
            for f_selector, acc in labels.items():
                local_acc.append(acc[idx])
                local_f_selectors.append(f_selector)
            local_rank = [sorted(local_acc, reverse=True).index(x) + 1
                          for x in local_acc]

            for f, r in zip(local_f_selectors, local_rank):
                global_rank[f].append(r)

        # ######Actual plotting
        ax = fig.add_subplot(grid_spec)
        axes.append(ax)
        for idx, (f_selector, rank) in zip(range(len(global_rank)),
                                           global_rank.items()):
            _mean, _std = np.mean(rank), np.std(rank)
            x_max_lim = _mean if _mean > x_max_lim else x_max_lim
            y_max_lim = _std if _std > y_max_lim else y_max_lim
            #print(_mean, _std, f_selector)
            ax.scatter(_mean, _std, s=30, c=methods_color[f_selector],
                       label=f_selector)

        ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.5,
                alpha=0.3)

        if not y_flag:
            ax.tick_params(axis="y", labelsize=5)
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis="y", labelsize=5)
        ax.tick_params(axis="x", labelsize=5)

    line, label = axes[0].get_legend_handles_labels()
    lines.extend(line)
    labels_legend.extend(label)
    legend = fig.legend(lines, labels, fontsize=7,
                        title=fill("Feature selection methods", 50),
                        loc="upper left", ncol=len(labels),
                        columnspacing=0.8, bbox_to_anchor=(0, 0))
    legend.get_frame().set_edgecolor("k")
    plt.setp(legend.get_title(), fontsize=7)

    for ax, letter in zip(axes, letters):
        ax.text(x_max_lim*0.5, -0.5, letter, fontsize=7,
                fontname="Times New Roman")
        ax.set_xlim([0, x_max_lim*1.1])
        ax.set_ylim([-0.1, y_max_lim*1.1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
        for ytick in ax.get_yticks():
            ax.hlines(ytick, 0, 0.13, linewidth=0.7, colors=["k"])

    fig.text(0.45, 0.02, "Mean", fontsize=8)
    fig.text(0.0, 0.5, "Standard deviation ", rotation=90, va="center",
             fontsize=8)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_jaccard_matrix(sets_slices: dict, save: str):
    """
    Jaccard Index per each combination of classifier and explainer
    """
    letters = ["(a)", "(b)", "(c)", "(d)"]
    grid_specs, _ = get_grid_spec(len(sets_slices), wspace=0.2, hspace=0.35)
    fig = plt.figure(figsize=(3.5, 4.7))
    cmap = plt.cm.YlOrRd
    cmap.set_bad("w")
    axes = []

    for (_, ranking_sets), grid_spec in zip(sets_slices.items(), grid_specs):
        len_sets = len(ranking_sets)
        jaccard_matrix = np.zeros((len_sets, len_sets))
        # ###Compute Jaccar Index for each pair of methods
        x = 0
        for fs_model_1, set_1 in ranking_sets.items():
            y = 0
            for fs_model_2, set_2 in ranking_sets.items():
                jaccard_matrix[x][y] = jaccard_similarity(set_1, set_2)
                y += 1
            x += 1
        # Masking lower part of matrix
        mask = np.triu(np.ones(jaccard_matrix.shape), k=1)
        jaccard_matrix = np.ma.array(jaccard_matrix, mask=mask)
        # ###Actual plot
        ax = fig.add_subplot(grid_spec)
        axes.append(ax)
        im = ax.imshow(jaccard_matrix, interpolation="nearest",
                       cmap=cmap, vmin=0, vmax=1)
        ax.set(xticks=np.arange(len(jaccard_matrix)),
               yticks=np.arange(len(jaccard_matrix)))
        ax.set_xticklabels(ranking_sets.keys(), fontsize=5)
        ax.set_yticklabels(ranking_sets.keys(), fontsize=5)

        for i in range(len_sets):
            for j in range(len_sets):
                if isinstance(jaccard_matrix[i, j], np.ma.core.MaskedConstant):
                    continue
                ax.text(j, i, format(jaccard_matrix[i, j], ".2f"),
                        ha="center", va="center",
                        color="white" if jaccard_matrix[i, j] > 0.5 else "black",
                        fontsize=5)

        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, location="bottom", shrink=0.95, aspect=30,
                        pad=0.1)
    cbar.ax.set_xlabel('Jaccard Index', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for ax, letter in zip(axes, letters):
        ax.text(len_sets/3, len_sets*1.15, letter, fontsize=7,
                fontname="Times New Roman")

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_jaccard_matrix_chb_vs_siena(chb_slices: dict, siena_slices: dict, save: str):
    """
    Jaccard Index per each combination of classifier and explainer
    """
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    grid_specs, ax_flags = get_grid_spec(len(chb_slices) + len(siena_slices),
                                         wspace=0.33, hspace=0)
    fig = plt.figure(figsize=(7, 5))
    cmap = plt.cm.YlOrRd
    cmap.set_bad("w")
    axes = []
    slices = list(chb_slices.items()) + list(siena_slices.items())

    for (_, ranking_sets), grid_spec, ax_flag in zip(slices, grid_specs,
                                                     ax_flags):
        len_sets = len(ranking_sets)
        jaccard_matrix = np.zeros((len_sets, len_sets))
        # ###Compute Jaccar Index for each pair of methods
        x = 0
        for fs_model_1, set_1 in ranking_sets.items():
            y = 0
            for fs_model_2, set_2 in ranking_sets.items():
                jaccard_matrix[x][y] = jaccard_similarity(set_1, set_2)
                y += 1
            x += 1
        # Masking lower part of matrix
        mask = np.triu(np.ones(jaccard_matrix.shape), k=1)
        jaccard_matrix = np.ma.array(jaccard_matrix, mask=mask)
        # ###Actual plot
        ax = fig.add_subplot(grid_spec)
        axes.append(ax)
        im = ax.imshow(jaccard_matrix, interpolation="nearest",
                       cmap=cmap, vmin=0, vmax=1)
        ax.set(xticks=np.arange(len(jaccard_matrix)),
               yticks=np.arange(len(jaccard_matrix)))
        ax.set_xticklabels(ranking_sets.keys(), fontsize=5)
        if ax_flag:
            ax.set_yticklabels(ranking_sets.keys(), fontsize=5)
        else:
            ax.set_yticklabels([])

        for i in range(len_sets):
            for j in range(len_sets):
                if isinstance(jaccard_matrix[i, j], np.ma.core.MaskedConstant):
                    continue
                ax.text(j, i, format(jaccard_matrix[i, j], ".2f"),
                        ha="center", va="center",
                        color="white" if jaccard_matrix[i, j] > 0.5 else "black",
                        fontsize=5)

        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, location="bottom", shrink=1, aspect=30,
                        pad=0.1)
    cbar.ax.set_xlabel('Jaccard Index', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for ax, letter in zip(axes, letters):
        ax.text(len_sets/3, len_sets*1.20, letter, fontsize=7,
                fontname="Times New Roman")

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()
