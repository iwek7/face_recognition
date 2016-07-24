from lasagne import layers
from nolearn.lasagne import NeuralNet

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import PrintLayerInfo
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
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from matplotlib import pyplot
from orl_faces import OrlFaces

from load_images import load, load2d


def float32(k):
    return np.cast['float32'](k)


class CustBatchIterator(BatchIterator):
    
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
    
    # pusta przestrzen ktora pojawila sie po obrocie lub przesunieciu obrazka
    # jest zapelniana ta wartoscia
    EMPTY_SPACE_FILL = 0
      
    # zwraca lustrzane odbicie tablicy oraz punktow na niej
    def get_mirror_image(self,Xb, yb, changed_indices):

        Xb[changed_indices] = Xb[changed_indices, :, :, ::-1]
        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[changed_indices, ::2] = yb[changed_indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[changed_indices, a], yb[changed_indices, b] = (
                    yb[changed_indices, b], yb[changed_indices, a])
        return Xb, yb
    
    # rotation_range to zakres losowej rotacji
    # nie uzywam 
    def apply_rotation(self, Xb, yb, changed_indices, rotation_max_angle = 30):
   
        Xb_cols = Xb.shape[2]
        Xb_rows = Xb.shape[3]
        
        # jesli jeden bulk jest mniejszy to nalezy uciac wszystkie indeksy
        changed_indices = changed_indices[changed_indices < yb.shape[0]]
        
        rotation_angle = np.random.randint(-rotation_max_angle, rotation_max_angle)

        # dla kazdego zestawu punktow wspolrzednych, ktore zostaja zmieniane,
        # wykonujemy jego transformacje
        # potrzebujemy znac indeks poniewaz w przypadku, w ktorym rotacja sie nie udaje
        # nie wykonujemy jej i idziemy dalej
        for yb_value, index in zip(yb[changed_indices],changed_indices): 
            
            # tworzymy tablice zer o rozdzielczosci obrazka w celu
            # (malo) sprytnej rotacji indeksow
            temp_yb_frame = np.zeros((Xb_cols, Xb_rows)).astype(int)
            
            # zapelniamy jedynkami miejsca odpowiadajace wspolrzednym punktow charakterystycznych
            # wspolrzedne ulozone sa obok siebie wektorze stad taki loop
            # wspolrzedne sa floatami wiec je konwertuje do inta
            for x, y in zip(yb_value[1 :: 2] * 48 + 48, yb_value[0 :: 2] * 48 + 48): 
                temp_yb_frame[int(x) - 1, int(y) - 1] = 1
           
            # rotacja macierzy metoda interpolacji splainowej, kat losowy
            temp_yb_frame_trans = ndimage.interpolation.rotate(temp_yb_frame, rotation_angle, 
                                                   reshape = False, mode = 'constant',
                                                   cval = 0)

            new_yb = list()
            # odzyskanie indeksow punktow
            for x in range(Xb_cols):
                for y in range(Xb_rows):
                    if temp_yb_frame_trans[y,x] != 0:
                        new_yb.append((x - 48) / 48)
                        new_yb.append((y - 48) / 48)
            
            # tutaj dodac sprawdzenie czy wszystkie punkty sa zawarte
            # jesli nie to usuwamy indeks z listy do zmiany i idziemy dalej
            # jesli tak to podmieniamy wartosc w tablicy na nowa
            # UWAGA ZMIENIAAM STRUKTURY PO KTORYCH LOOPUJE TO MOZE RODZIC PROBLEMY POTENCJALNIE
            if len(new_yb) != 30:
                changed_indices = changed_indices[changed_indices != index]
            else:
                yb[index] = np.array(new_yb)

        # teraz dla wszystkich obrazow, dla ktorych transformacja puntow odbyla sie pomyslnie
                      
        for xb_value, index in zip( Xb[changed_indices], changed_indices):
            Xb[index][0] = ndimage.interpolation.rotate(xb_value[0], rotation_angle,
                                                  reshape=False, mode='constant',
                                                  cval=1)
            
         
        return Xb, yb

    # przesuwa obrazek ku dolowi
    # nie uzywam - slabe efekty
    def offset_image(self, Xb, yb, changed_indices, max_offset = 15):
        offset = np.random.randint(max_offset)
        Xb_cols = Xb.shape[2]
        Xb_rows = Xb.shape[3]
        changed_indices = changed_indices[changed_indices < yb.shape[0]]
        for yb_value, index in zip(yb[changed_indices], changed_indices): 
            yb_trans = yb_value * 48 + 48
            yb_trans[1::2] += offset 
            # sprawdzenie czy znacznik keypointa nie przesunal sie poza obrazek
            # jesli tak sie stalo to rezygnujemy z przesuniecia
            if yb_trans[yb_trans > Xb_rows].size > 0:
                changed_indices = changed_indices[changed_indices != index]
            else:
               yb[index] = (yb_trans - 48) / 48
               
        empty_arr = np.empty((offset, Xb_cols))
        empty_arr.fill(self.EMPTY_SPACE_FILL)
        

        for xb_value, index in zip( Xb[changed_indices], changed_indices):
            Xb[index][0] = np.vstack((empty_arr, Xb[index][0][0 : Xb_cols - offset, : ]))       


        return Xb, yb
                                      
        
    # transformuje paczke danych na poczatku iteracji algorytmu uczacego na poczatku treningu     
    def transform(self, Xb, yb):
        Xb, yb = super(CustBatchIterator, self).transform(Xb, yb)
        
        batch_shape = Xb.shape[0] 

        # odbicie lustrzane tablic        
        # z np.arange(batch_shape) wez batch_shape / 2 losowych indeksow
        mirror_image_indices = np.random.choice(batch_shape, batch_shape / 2, replace=False)        
        Xb, yb = self.get_mirror_image(Xb, yb, mirror_image_indices)
        
        #
        # rotation_indices = np.random.choice(batch_shape, batch_shape / 2  , replace=False) 
        #print(rotation_indices)
        # rotation_indices = np.array([i for i in range(128)])
        # Xb, yb = self.apply_rotation(Xb,yb, rotation_indices) 
        
        #offset_indices = np.random.choice(batch_shape, batch_shape / 4  , replace=False) 
        #offset_indices = np.array([i for i in range(128)])
        # Xb, yb = self.offset_image(Xb, yb, offset_indices)        

        return Xb, yb

# zmienia learning rate w czasie uczenia
# nie uzywam, poprawa jest niewielka na moim zbiorze testowym
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

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
 

# wczytanie pliku
NET_NAME = 'net_conv2_bigger_pooling_tanh_no_reg'

def load_neural_network(path):
    with open(path, 'rb') as f:
        nnet = pickle.load(f)
    return nnet
nnet = load_neural_network(NET_NAME + '.pickle')




# liczenie bledu

#orl_faces = OrlFaces()
#orl_faces.laod_orl_faces_2d_np_arr()

#orl_faces.make_orl_predictions(nnet)
#orl_faces.save_orl_predictions('pred_' + NET_NAME + '.csv')

#orl_faces.load_orl_keypoints("C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv")
####orl_faces.load_orl_predictions('pred_' + NET_NAME + '.csv')
#print(orl_faces.calculate_total_error())
#print(nnet.train_history_[-1])
#orl_faces.plot_orl_predictions()
###orl_faces.save_rearranged_keypoints_and_predictions(net_name=NET_NAME)




# plotowanie historii uczenia
#from neuralNetVisualizations import plot_training_history
#plot_training_history(nnet)


# plotowanie feature map
#nnet.save_params_to(NET_NAME + '_weights.pickle')
#from neuralNetVisualizations import plot_feature_maps
#plot_feature_maps(NET_NAME + '_weights.pickle', 'conv2', (12,11))


# przepuszczanie twarzy przez siec

from neuralNetVisualizations import plot_conv_layer_output
from neuralNetVisualizations import get_layer_output
from neuralNetVisualizations import plot_pool_layer_output

orl_faces = OrlFaces()
orl_faces.laod_orl_faces_2d_np_arr()
input =  orl_faces.orl_faces_reshaped[0:1,:,:,:].astype('float64')
#plot_conv_layer_output(nnet, 1, input) # conv1



# conv1
output_conv1 = get_layer_output(nnet, 1, input)
plot_conv_layer_output(nnet, 1, input) 


# pool1
plot_pool_layer_output(nnet, 2, output_conv1[0], (8,8))

output_pool1 = get_layer_output(nnet, 2, output_conv1[0])
output_pool1_reshaped = np.zeros(shape=(1,64,23,23))
for feature_map in range(output_pool1.shape[0]):
    output_pool1_reshaped[0][feature_map] = output_pool1[feature_map]

# conv2
output_conv2 = get_layer_output(nnet, 4, output_pool1_reshaped)
plot_conv_layer_output(nnet, 4, output_pool1_reshaped, (12,11))

# pool2
output_pool2 = get_layer_output(nnet, 5, output_conv2[0])
plot_pool_layer_output(nnet, 5, output_conv2[0], (12,11))