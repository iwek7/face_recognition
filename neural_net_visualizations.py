from matplotlib import pyplot as plt
import numpy as np
import pickle 
from itertools import product

def load_weights(weights_path):
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

NET_NAME = 'net_conv2_bigger_pooling_tanh_no_reg'
def load_neural_network(path):
    with open(path, 'rb') as f:
        nnet = pickle.load(f)
    return nnet
nnet = load_neural_network(NET_NAME + '.pickle')

def plot_training_history(nnet):
    train_loss =  np.sqrt(np.array([i["train_loss"] for i in nnet.train_history_])) * 48
    valid_loss = np.sqrt(np.array([i["valid_loss"] for i in nnet.train_history_])) * 48
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1, 6)
    plt.show()


def plot_feature_maps(weights_path, layer_name, figsize = (6, 6)):
    weights = load_weights(weights_path)
    nrows  = weights[layer_name][0].shape[2]
    ncols = weights[layer_name][0].shape[3]
    figs, axes = plt.subplots(figsize[0], figsize[1], figsize=figsize, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for (feature_map, ax) in zip(weights[layer_name][0], axes.flatten()):
        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
                if i >= weights[layer_name][0].shape[0]:
                    break
                ax.imshow(feature_map[0],  cmap='gray', interpolation='none')
    plt.show()

def get_layer_output(net, layer_index, input):
    return net.layers_[layer_index].get_output_for(input).eval()

def get_layer_output2(net, layer, input):
    return layer.get_output_for(input).eval()


def plot_pool_layer_output(net, layer_index, input, figsize = (8, 8),):
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    layer_output = get_layer_output(net, layer_index, input)
    print(layer_output.shape)
    for image_id in range(layer_output.shape[0]):
        ax = fig.add_subplot(figsize[0], figsize[1], image_id + 1,  xticks=[], yticks=[])
        ax.imshow(layer_output[image_id], cmap='gray')
    plt.show()

def plot_conv_layer_output(net, layer_index, input, figsize = (8, 8)):
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    layer_output = get_layer_output(net, layer_index, input)
    print(layer_output.shape)
    for image_id in range(layer_output.shape[1]):
        ax = fig.add_subplot(figsize[0], figsize[1], image_id + 1,  xticks=[], yticks=[])
        ax.imshow(layer_output[0][image_id], cmap='gray')
    plt.show()

####
###
##
# to to po kolei wyci?ganie danych z sieci na podstawie orl.faces
##
###
####

#orl_faces = OrlFaces()
#orl_faces.laod_orl_faces_2d_np_arr()


#input =  orl_faces.orl_faces_reshaped[0:1,:,:,:].astype('float64')
#output_conv1 = get_layer_output(nnet, 1, input)


#output_pool1 = get_layer_output(nnet, 2, output_conv1[0])

##layer = layers.InputLayer(shape=(1, 32, 94, 94),
##                    input_var=output_conv1
##                    )
##layer = layers.MaxPool2DLayer(layer, pool_size = (4, 4))


#output_pool1_reshaped = np.zeros(shape=(1,32,47,47))
#for feature_map in range(output_pool1.shape[0]):
#    output_pool1_reshaped[0][feature_map] = output_pool1[feature_map]


#output_conv2 = get_layer_output(nnet, 4, output_pool1_reshaped)
##plot_conv_layer_output(nnet, 4, output_pool1_reshaped, (8,8))

#output_pool2 = get_layer_output(nnet, 5, output_conv2[0])
#plot_pool_layer_output(nnet, 5, output_conv2[0], (8,8))

#output_output_pool2 = np.zeros(shape=(1,64,23,23))
#for feature_map in range(output_pool2.shape[0]):
#    output_output_pool2[0][feature_map] = output_pool2[feature_map]




#output_conv3 = get_layer_output(nnet, 7, output_output_pool2)
##plot_conv_layer_output(nnet, 7, output_output_pool2, (12,12))


#output_pool4 = get_layer_output(nnet, 8, output_conv3[0])
##plot_pool_layer_output(nnet, 8, output_conv3[0], (12,12))
