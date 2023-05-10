import numpy
from torchmetrics import Accuracy, Recall, Precision
import torch
import numpy as np


def confusion_matrix(y, y_hat):
    """
    :return: TP, FP, TN, FN
    """
    if torch.is_tensor(y):
        if torch.max(y) == 255. and torch.min(y) == 0.:
            y = y / 255.
        y = y.numpy()
    if torch.is_tensor(y_hat):
        if torch.max(y_hat) == 255. and torch.min(y_hat) == 0.:
            y_hat = y_hat / 255.
        y_hat = y_hat.numpy()
    if isinstance(y, list) or isinstance(y_hat, list):
        y = np.array(y)
        if np.max(y) == 255. and np.min(y) == 0.:
            y = y / 255.
        y_hat = np.array(y_hat)
        if np.max(y_hat) == 255. and np.min(y_hat) == 0.:
            y_hat = y_hat / 255.
    if isinstance(y, numpy.ndarray) and isinstance(y_hat, numpy.ndarray):
        TP = np.sum(np.logical_and(y == 1., y_hat == 1.)).astype('float32')
        FP = np.sum(np.logical_and(y == 0., y_hat == 1.)).astype('float32')
        TN = np.sum(np.logical_and(y == 0., y_hat == 0.)).astype('float32')
        FN = np.sum(np.logical_and(y == 1., y_hat == 0.)).astype('float32')
        return TP, FP, TN, FN


def accuracy(y, y_hat, eps=1e-10):
    """
    :return: (TP + TN) / (TP + TN + FP + FN)
    """
    TP, FP, TN, FN = confusion_matrix(y, y_hat)
    return (TP + TN) / (TP + TN + FP + FN + eps)


def recall(y, y_hat, eps=1e-10):
    """
    :return: TP / (TP + FN)
    """
    TP, FP, TN, FN = confusion_matrix(y, y_hat)
    return TP / (TP + FN + eps)


def precision(y, y_hat, eps=1e-10):
    """
    :return: TP / (TP + FP)
    """
    TP, FP, TN, FN = confusion_matrix(y, y_hat)
    return TP / (TP + FP + eps)


def f_measure(y, y_hat, alpha=1, eps=1e-10):
    """
    :return: 2 * precision * recall / (precision + recall)
    """
    TP, FP, TN, FN = confusion_matrix(y, y_hat)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    pr = precision + recall + eps
    f1 = ((alpha * alpha + 1.) * precision * recall) / (alpha * alpha * pr)
    return f1


if __name__ == '__main__':
    y = np.array(
        [[[[1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
          [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]]]])
    y_hat = np.array(
        [[[[1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
          [1, 0, 1, 1, 1, 0, 0, 1, 1, 0]]]])
    TP, FP, TN, FN = confusion_matrix(y=y, y_hat=y_hat)
    print(TP, FP, TN, FN)
    accuracy = accuracy(y, y_hat)
    recall = recall(y, y_hat)
    precision = precision(y, y_hat)
    f1_score = f_measure(y, y_hat, alpha=1)
    print(accuracy)
    print(recall)
    print(precision)
    print(f1_score)
