#!/usr/bin/env python
# -*- coding: utf-8 -*-

from orl_faces import OrlFaces
import numpy as np

import scipy
import matplotlib.pyplot as plt



NET_NAME = 'net_conv2_bigger_pooling_tanh_no_reg'

# load data from saved files
orl_faces = OrlFaces()
orl_faces.load_orl_predictions('pred_' + NET_NAME + '.csv')
orl_faces.load_orl_predictions 
orl_faces.load_orl_keypoints("C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv")

# add avg from train set to plot avg of keypoint from train data
from load_images import load, load2d
kaggle_X, kaggle_y = load2d() 
kaggle_y = kaggle_y * 48 + 48

kaggle_y_reshaped = np.ndarray(shape=(kaggle_y.shape[1],kaggle_y.shape[0]))
for person_num in range(kaggle_y.shape[0]):
    for keypoint_num in range(kaggle_y.shape[1]):
        kaggle_y_reshaped[keypoint_num][person_num]=kaggle_y[person_num][keypoint_num]



kaggle_y = kaggle_y_reshaped
print((kaggle_y[0]))
orl_variances = np.zeros(30)
kaggle_variances = np.zeros(30)

kaggle_means = np.zeros(30)
orl_means= np.zeros(30)

i = 0





for keypoint in list(orl_faces.orl_keypoints.columns):

    orl_variances[i] = np.var(orl_faces.orl_keypoints[keypoint].values)
    kaggle_variances[i] = np.var(kaggle_y[i])

    kaggle_means[i] = np.mean(kaggle_y[i]) 
    orl_means[i] = np.mean(orl_faces.orl_keypoints[keypoint].values)
    i += 1

def plot_variances():
    ind = np.arange(len(orl_faces.orl_keypoints.columns))
    width = 0.35
    idx = kaggle_variances.argsort()
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, kaggle_variances[idx]  , width, color='r')
    rects2 = ax.bar(ind + width, orl_variances[idx] , width, color='y')

    ax.set_ylabel('Wariancja')
    ax.set_title('Wariancja wspolrzednych w obu zbiorach')

    ax.set_xticks(ind)
    ax.set_xticklabels(orl_faces.orl_keypoints.columns[idx],  rotation='vertical')

    ax.legend((rects1[0], rects2[0]), ('Zbior Kaggle', 'Zbior ORL_FACES'),loc='top left')
    plt.show()



def plot_means():
    ind = np.arange(len(orl_faces.orl_keypoints.columns))
    width = 0.35
    idx = kaggle_means.argsort()
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, kaggle_means[idx], width, color='r')
    rects2 = ax.bar(ind + width, orl_means[idx] , width, color='y')

    ax.set_ylabel('Srednia')
    ax.set_title('Srednia wartosc wspolrzednych w obu zbiorach')

    ax.set_xticks(ind)
    ax.set_xticklabels(orl_faces.orl_keypoints.columns[idx],  rotation='vertical')

    ax.legend((rects1[0], rects2[0]), ('Zbior Kaggle', 'Zbior ORL_FACES'),loc='top left')

    plt.show()

plot_variances()
plot_means()



