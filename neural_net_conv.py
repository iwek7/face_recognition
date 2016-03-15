# conv-net 
from load_images import load2d

from lasagne import layers
from nolearn.lasagne import NeuralNet
import pickle 
import numpy as np
import sys
import math
from scipy import ndimage
sys.setrecursionlimit(10000)

import theano

def float32(k):
    return np.cast['float32'](k)

from nolearn.lasagne import BatchIterator

from matplotlib import pyplot
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)



class CustBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
    
    # pusta przestrzen ktora pojawila sie po obrocie lub przesunieciu obrazka
    # jest zapelniana ta wartoscia
    empty_space_fill = 0
      
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
        empty_arr.fill(self.empty_space_fill)
        

        for xb_value, index in zip( Xb[changed_indices], changed_indices):
            Xb[index][0] = np.vstack((empty_arr, Xb[index][0][0 : Xb_cols - offset, : ]))       


        return Xb, yb
                                      
        
        
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
        
        # offset_indices = np.random.choice(batch_shape, batch_shape / 2  , replace=False) 
        offset_indices = np.array([i for i in range(128)])
        Xb, yb = self.offset_image(Xb, yb, offset_indices)        

        #
        #fig = pyplot.figure(figsize=(6, 6))
        #fig.subplots_adjust(
        #    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        #a = 0
        #print("/n/n/n newline /n/n")
        #for i in range(a,a+16):
        #    ax = fig.add_subplot(4, 4, i - a + 1, xticks=[], yticks=[])
        #    plot_sample(Xb[i], yb[i], ax)
        #    print(Xb.shape)
        #    print(yb.shape)
        #
   
       
            
        #pyplot.show()


        return Xb, yb

        
             

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
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size=(3, 3), 
    pool1_pool_size=(2, 2),
    dropout1_p = 0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p = 0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p = 0.3,  # !
    hidden4_num_units=1000,
    dropout4_p = 0.5,  # !
    hidden5_num_units=1000,
    output_num_units = 30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    objective_l2=0.0001, 
    regression=True,
    batch_iterator_train=CustBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=800,
    verbose=1,
    )

#X, y = load2d()  
#print("images loaded")
#net.fit(X, y)

#with open('net6.pickle', 'wb') as f:
#    pickle.dump(net, f, -1)


############################################################################################


import orl_faces
def load_neural_network(path):
    with open(path, 'rb') as f:
        nnet = pickle.load(f)
    return nnet


from matplotlib import pyplot
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


orl_faces_reshaped = orl_faces.get_orl_faces_2d_np_arr("C:/Users/Michal/Documents/magisterka/dane/orl_faces/")
nnet = load_neural_network('net3.pickle')
orl_faces_predictions = orl_faces.make_orl_predictions(orl_faces_reshaped, nnet)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 0)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 16)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 32)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 48)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 64)
orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16, 80)

#X, _ = load2d(test=True)
#y_pred = nnet.predict(X)

#fig = pyplot.figure(figsize=(6, 6))
#fig.subplots_adjust(
#    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#a = 64
#for i in range(a,a+16):
#    ax = fig.add_subplot(4, 4, i - a + 1, xticks=[], yticks=[])
#    plot_sample(X[i], y_pred[i], ax)


#pyplot.show()


###print(y_pred[0][0::2] * 48 + 48)
###print(y_pred[0][1::2] * 48 + 48)
##pyplot.show()
#train_loss = np.array([i["train_loss"] for i in net1.train_history_])
#valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
##pyplot.plot(train_loss, linewidth=3, label="train")
#pyplot.plot(valid_loss, linewidth=3, label="valid")
#pyplot.grid()
#pyplot.legend()
#pyplot.xlabel("epoch")
#pyplot.ylabel("loss")
##pyplot.ylim(1e-3, 1e-2)
#pyplot.yscale("log")
#pyplot.show()
##