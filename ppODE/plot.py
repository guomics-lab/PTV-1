# plot.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

def configure_plotting():
    smaller_size = 13
    medium_size = 14
    bigger_size = 16

    plt.rc('font', size=bigger_size)          # controls default text sizes
    plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=smaller_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=smaller_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=smaller_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

def fl_line(x, fl_n):
    return [sum(x[i:i + fl_n]) / fl_n for i in range(0, len(x), fl_n)]

def plot_loss(train_loss, test_loss, fl_n, dir_save):
    configure_plotting()
    train_loss = fl_line(train_loss, fl_n)
    test_loss = fl_line(test_loss, fl_n)
    steps = len(train_loss)
    x_step = fl_line(np.arange(steps), fl_n)

    plt.figure(figsize=(9, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(x_step, train_loss)
    plt.title('Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(x_step, test_loss)
    plt.title('Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f"{dir_save}/fig1.pdf")

def plot_pheno(train_loss, train_accuracy, test_loss, test_accuracy, fl_n, dir_save):
    configure_plotting()
    train_loss = fl_line(train_loss, fl_n)
    train_accuracy = fl_line(train_accuracy, fl_n)
    test_loss = fl_line(test_loss, fl_n)
    test_accuracy = fl_line(test_accuracy, fl_n)
    steps = len(train_loss)
    x_step = fl_line(np.arange(steps), fl_n)

    plt.figure(figsize=(9, 8), dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(x_step, train_loss)
    plt.title('Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(x_step, test_loss)
    plt.title('Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(x_step, train_accuracy)
    plt.title('Train Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(x_step, test_accuracy)
    plt.title('Test Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f"{dir_save}/fig2.pdf")

def plot_auprc_auroc(train_auprc, train_auroc, test_auprc, test_auroc, fl_n, dir_save):
    configure_plotting()
    train_auprc = fl_line(train_auprc, fl_n)
    train_auroc = fl_line(train_auroc, fl_n)
    test_auprc = fl_line(test_auprc, fl_n)
    test_auroc = fl_line(test_auroc, fl_n)
    steps = len(train_auprc)
    x_step = fl_line(np.arange(steps), fl_n)

    plt.figure(figsize=(9, 8), dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(x_step, train_auprc)
    plt.title('Train auprc')
    plt.xlabel('Steps')
    plt.ylabel('auprc')

    plt.subplot(2, 2, 2)
    plt.plot(x_step, test_auprc)
    plt.title('Test auprc')
    plt.xlabel('Steps')
    plt.ylabel('auprc')

    plt.subplot(2, 2, 3)
    plt.plot(x_step, train_auroc)
    plt.title('Train auroc')
    plt.xlabel('Steps')
    plt.ylabel('auroc')

    plt.subplot(2, 2, 4)
    plt.plot(x_step, test_auroc)
    plt.title('Test auroc')
    plt.xlabel('Steps')
    plt.ylabel('auroc')

    plt.tight_layout()
    plt.savefig(f"{dir_save}/fig3.pdf")

def plot_proteomics_predict(train_loss, train_accuracy, test_loss, test_accuracy, fl_n, dir_save):
    configure_plotting()
    train_loss = fl_line(train_loss, fl_n)
    train_accuracy = fl_line(train_accuracy, fl_n)
    test_loss = fl_line(test_loss, fl_n)
    test_accuracy = fl_line(test_accuracy, fl_n)
    steps = len(train_loss)
    x_step = fl_line(np.arange(steps), fl_n)

    plt.figure(figsize=(9, 8), dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(x_step, train_loss)
    plt.title('Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(x_step, test_loss)
    plt.title('Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(x_step, train_accuracy)
    plt.title('Train Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(x_step, test_accuracy)
    plt.title('Test Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f"{dir_save}/fig4.pdf")


def binary_auroc_plt(output, target):
    return auc(*roc_curve(target, output)[:2])

def plot_auroc(pheno_predict_np, ph_all_np, dir_save, save_tailfix):
    auroc = binary_auroc_plt(pheno_predict_np, ph_all_np).item()
    fpr, tpr, _ = roc_curve(ph_all_np, pheno_predict_np)

    plt.figure(figsize=(3, 3), dpi=300)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="best")
    plt.savefig(f"{dir_save}/fig_auroc_{save_tailfix}.pdf")

def plot_auprc(pheno_predict_np, ph_all_np, dir_save, save_tailfix):
    average_precision = average_precision_score(ph_all_np, pheno_predict_np)
    precision, recall, _ = precision_recall_curve(ph_all_np, pheno_predict_np)

    plt.figure(figsize=(3, 3), dpi=200)
    lw = 2
    plt.plot(recall, precision, color='limegreen', lw=lw, label='PR curve (area = %0.2f)' % average_precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"{dir_save}/fig_auprc_{save_tailfix}.pdf")