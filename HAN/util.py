import time
import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from random import shuffle


def read_dataset():
    with open('trainingless_data', 'rb') as f:
        data_x, data_y = pickle.load(f)
        xy = list(zip(data_x, data_y))
        shuffle(xy)
        data_x, data_y = zip(*xy)
        length = len(data_x)
        train_x, dev_x = data_x[:int(length * 0.95)], data_x[int(length * 0.95) + 1:]
        train_y, dev_y = data_y[:int(length * 0.95)], data_y[int(length * 0.95) + 1:]
        return train_x, train_y, dev_x, dev_y

def read_test():
    with open('testing_data2', 'rb') as f:
        data_x, data_y = pickle.load(f)
        xy = list(zip(data_x, data_y))
        shuffle(xy)
        data_x, data_y = zip(*xy)

        return data_x, data_y

def recall(predict, label):
    pre = np.asarray(predict)
    lab = np.asarray(label)

    recall = np.sum(lab[pre == 1]) / np.sum(lab)
    precision = np.sum(lab[pre == 1]) / np.sum(pre)

    return recall, precision

def AUC(prediction, labels):
    '''
    :param prediction: [[prob_neg, prob_postive],...]
    :param labels:[[negative_index, positive_index],...]
    :return:
    '''
    score = roc_auc_score(labels, prediction[:,1])

    return score

def weighted_softmaxloss():
    '''
    negativve over positive ratio 27:1
    weighted the false negative with weights 27
    '''

def testdata():
    with open('./training_data', 'rb') as f:
        data_x, data_y = pickle.load(f)
        print(np.asarray(data_x).shape,np.asarray(data_y))

