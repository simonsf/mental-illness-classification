import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def cal_performance(label, prediction):
    label = label.cpu().numpy()
    prediction = prediction.cpu().numpy()
    label_ignore = np.where(label == -1)
    label = np.delete(label, label_ignore)
    prediction = np.delete(prediction, label_ignore)
    accuracy = accuracy_score(y_true=label, y_pred=prediction)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=label, y_pred=prediction, average='macro')
    return accuracy, f1.mean(), precision.mean(), recall.mean()