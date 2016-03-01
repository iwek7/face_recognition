
from load_images import load
import pickle as pickle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet




net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    
    
    # parametry odwoluja sie do warst prefixami
    
    input_shape= (None, 96 * 96),  # obrazki maja format 96 * 96
    hidden_num_units = 100,  # ilosc neuronow na ukrytej warstwie
    output_nonlinearity = None,  # output layer uses identity function
    output_num_units = 30,  # 15 charakterystycznych punktow, kazdy ma dwie koordynaty

    
    # parametry zaczynajace sie od update odnosza sie do update function
    # albo metody optymalizacji
    
    # inne to np: adagrad(?), rmsprop(?)
    # nesterov_momentum to jakis gradiend descend z czyms ekstra (doczytac)
    update=nesterov_momentum, 
    update_learning_rate = 0.01,
    update_momentum = 0.9,
    
    # jesli false to znaczy ze mamy klasyfikacje
    regression = True,  
    
    # An epoch is a measure of the number of times all of the training
    # vectors are used once to update the weights
    max_epochs = 400,  
    # to do printowania infa w czasie trenowania
    verbose = 1,
    )
    
# siec neuronowa sama robi training i validation set
X, y = load()
net1.fit(X, y)


with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)
#     
#==============================================================================



