from lasagne import layers
from nolearn.lasagne import NeuralNet

# tylko uzyte raz do robienia tabelki
#from nolearn.lasagne import PrintLayerInfo
from lasagne import nonlinearities
import pickle 
import numpy as np
import sys
import math
from scipy import ndimage
sys.setrecursionlimit(10000)

import theano

import os
import numpy as np

from matplotlib import pyplot
from orl_faces import OrlFaces

from load_images import load, load2d
from cust_batch_iterator import CustBatchIterator

# from adjust_variable import AdjustVariable

def float32(k):
    return np.cast['float32'](k)

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer), 
        
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=64, conv1_filter_size=(3, 3),
    pool1_pool_size=(4, 4),
    dropout1_p=0.3, 
    conv2_num_filters=128, conv2_filter_size=(2, 2),
    pool2_pool_size=(2, 2),
    
    dropout2_p=0.4, 
    hidden4_num_units=1000,  hidden4_nonlinearity=nonlinearities.tanh,
    hidden5_num_units=1000,  hidden5_nonlinearity=nonlinearities.tanh,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=CustBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    objective_l2=0.0002,
    max_epochs=5000,
    verbose=1,
    )

#X, y = load2d()  
#net.fit(X, y)
NET_NAME = 'net_conv2_bigger_pooling_tanh_no_reg'
#with open(NET_NAME + '.pickle', 'wb') as f:
#    pickle.dump(net, f, -1)
 


def load_neural_network(path):
    with open(path, 'rb') as f:
        nnet = pickle.load(f)
    return nnet
nnet = load_neural_network(NET_NAME + '.pickle')




# liczenie bledu

orl_faces = OrlFaces()
orl_faces.laod_orl_faces_2d_np_arr()

orl_faces.make_orl_predictions(nnet)
orl_faces.save_orl_predictions('pred_' + NET_NAME + '.csv')

orl_faces.load_orl_keypoints("C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv")
###orl_faces.load_orl_predictions('pred_' + NET_NAME + '.csv')
print(orl_faces.calculate_total_error())
print(nnet.train_history_[-1])
orl_faces.plot_orl_predictions()
##orl_faces.save_rearranged_keypoints_and_predictions(net_name=NET_NAME)




# plotowanie historii uczenia
#from neural_net_visualizations import plot_training_history
#plot_training_history(nnet)


# plotowanie feature map
#nnet.save_params_to(NET_NAME + '_weights.pickle')
#from neural_net_visualizations import plot_feature_maps
#plot_feature_maps(NET_NAME + '_weights.pickle', 'conv2', (12,11))


# przepuszczanie twarzy przez siec

#from neural_net_visualizations import plot_conv_layer_output
#from neural_net_visualizations import get_layer_output
#from neural_net_visualizations import plot_pool_layer_output

#orl_faces = OrlFaces()
#orl_faces.laod_orl_faces_2d_np_arr()
#input =  orl_faces.orl_faces_reshaped[0:1,:,:,:].astype('float64')
##plot_conv_layer_output(nnet, 1, input) # conv1



## conv1
#output_conv1 = get_layer_output(nnet, 1, input)
#plot_conv_layer_output(nnet, 1, input) 


## pool1
#plot_pool_layer_output(nnet, 2, output_conv1[0], (8,8))

#output_pool1 = get_layer_output(nnet, 2, output_conv1[0])
#output_pool1_reshaped = np.zeros(shape=(1,64,23,23))
#for feature_map in range(output_pool1.shape[0]):
#    output_pool1_reshaped[0][feature_map] = output_pool1[feature_map]

## conv2
#output_conv2 = get_layer_output(nnet, 4, output_pool1_reshaped)
#plot_conv_layer_output(nnet, 4, output_pool1_reshaped, (12,11))

## pool2
#output_pool2 = get_layer_output(nnet, 5, output_conv2[0])
#plot_pool_layer_output(nnet, 5, output_conv2[0], (12,11))