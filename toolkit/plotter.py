from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def palette_gen(n_colors):
    palette = plt.get_cmap("tab10")
    curr_idx = 0
    while curr_idx < 10:
        yield palette.colors[curr_idx]
        curr_idx += 1


def plot_losses(
    train_list: List[list],
    val_list: List[List],
    plot_path=None,
) -> None:
    """Plots training loss curves."""

    train_loss = np.array(train_list)
    train_loss = train_loss[:, 1]
    val_loss = np.array(val_list)
    val_loss = val_loss[:, 3]
    train_val_loss = np.vstack([train_loss, val_loss])
    train_val_loss = train_val_loss.astype(float)

    n_epochs = train_val_loss.shape[1]
    x_epochs = np.arange(n_epochs)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    name = ['tran_loss', 'val_loss']
    gen = palette_gen(len(name))

    for loss, name in zip(train_val_loss, name):
        ax1.plot(x_epochs, loss, label=name, lw=1.5, c=next(gen))

    plt.xlabel("Epochs")
    ax1.set_ylabel("Loss value")
    ax1.set_yscale("log")
    plt.grid(which="minor", axis="y")

    plt.savefig(plot_path + "/loss_plot.png")


def plot_roc_curve(y_test, y_score, save_path):

    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.subplot(1, 1, 1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path + "/roc_plot.png")


def plot_pr_curve(label, pred, save_path):
    plt.figure()
    precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
    auprc = metrics.auc(recall, precision)
    plt.plot(recall, precision, color='blue', label='ROC curve (area = %0.4f)' % auprc)
    plt.plot([1, 0.5], '--', color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.5, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision Recall Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.savefig(save_path + "/pr_plot.png")

    

