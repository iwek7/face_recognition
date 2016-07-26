from orl_faces import OrlFaces
import numpy as np

import scipy
import matplotlib.pyplot as plt



NET_NAME = 'net_conv2_even_bigger_pooling_tanh_no_reg_bigger_conv_less_dropout'

# load data from saved files
orl_faces = OrlFaces()
orl_faces.load_orl_predictions('pred_' + NET_NAME + '.csv')
orl_faces.load_orl_keypoints("C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv")

# add avg from train set to plot avg of keypoint from train data
from load_images import load, load2d
X, y = load2d() 

def plot_predictions_agains_obs():
    i = 0
    for keypoint in list(orl_faces.orl_keypoints.columns):
        if keypoint in ('LEFT_EYE_MIDDLE_X', 'MOUTH_RIGHT_Y', 'LEFT_EYE_MIDDLE_Y', 'LEFT_BROW_LEFT_Y'):
            # find sort order for real data
            sort_order = orl_faces.orl_keypoints[keypoint].values.argsort()

            # apply sort order to predictions and real data and unnormalize
            orl_pred_sorted = orl_faces.orl_predictions [keypoint].values[sort_order] * 48 + 48
            orl_real_sorted = orl_faces.orl_keypoints[keypoint].values[sort_order]

            # plot everything
            plt.plot(orl_pred_sorted, label="predykcja " + keypoint)
            plt.plot(orl_real_sorted, label="target " + keypoint)
            plt.plot(np.ones(400) * np.average(orl_faces.orl_keypoints[keypoint].values), label="srednia ORL_FACES",  linestyle="--")
            plt.plot(np.ones(400) * np.average(y[:,i] * 48 + 48) , label= "srednia Kaggle", linestyle="--", color="black")

            plt.xlabel("obserwacja")
            plt.ylabel("wspolrzedna piksela")
            plt.legend(loc='lower right')
            plt.show()
        i += 1

def plot_error_against_keypoints():

    all_errors = orl_faces.calculate_error_by_keypoints()
    ind = np.arange(len(all_errors))
    idx = all_errors.argsort()

    fig, ax = plt.subplots()
    bar = ax.bar(ind, all_errors[idx], color='b')
    ax.set_ylabel('Blad')
    ax.set_xticks(ind)
    ax.set_xticklabels(orl_faces.orl_keypoints.columns[idx],  rotation='vertical')


    orl_variances = np.zeros(30)
    i = 0
    for keypoint in list(orl_faces.orl_keypoints.columns):
        orl_variances[i] = np.var(orl_faces.orl_keypoints[keypoint].values)
        i += 1
    ax2 = ax.twinx()
    line = ax2.plot(orl_variances[idx], color='r', lw=3)
    ax2.set_ylabel('Wariancja punktu charakterystycznego')
    plt.legend((bar, line[0]),('Blad sredniokwadratowy', 'Wariancja'),loc='upper left')
    plt.show()



plot_predictions_agains_obs()    
