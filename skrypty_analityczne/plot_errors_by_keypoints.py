from orl_faces import OrlFaces
import numpy as np

import scipy
import matplotlib.pyplot as plt



NET_NAME = 'net_conv2_even_bigger_pooling_tanh_no_reg_bigger_conv_less_dropout'

# load data from saved files
orl_faces = OrlFaces()
orl_faces.load_orl_predictions('pred_' + NET_NAME + '.csv')
orl_faces.load_orl_predictions 
orl_faces.load_orl_keypoints("C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv")

# add avg from train set to plot avg of keypoint from train data
from load_images import load, load2d
X, y = load2d() 

i = 0
for keypoint in list(orl_faces.orl_keypoints.columns):

    # find sort order for real data
    sort_order = orl_faces.orl_keypoints[keypoint].values.argsort()

    # apply sort order to predictions and real data and unnormalize
    orl_pred_sorted = orl_faces.orl_predictions [keypoint].values[sort_order] * 48 + 48
    orl_real_sorted = orl_faces.orl_keypoints[keypoint].values[sort_order]

    # plot everything
    plt.plot(orl_pred_sorted, label="pred " + keypoint)
    plt.plot(orl_real_sorted, label="target " + keypoint)
    plt.plot(np.ones(400) * np.average(orl_faces.orl_keypoints[keypoint].values), label="test_avg",  linestyle="--")


    plt.plot(np.ones(400) * np.average(y[:,i] * 48 + 48) , label= "train/valid avg", linestyle="--", color="black")


    plt.legend(loc='lower left')
    plt.show()
    i += 1
