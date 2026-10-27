"""Validation accuracy, per-class report and confusion-matrix figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else float("nan")


def report(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    return classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0, digits=3)


def plot_confusion(y_true, y_pred, classes, path, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row-normalised: recall per true brand
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
