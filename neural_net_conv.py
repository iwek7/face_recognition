# conv-net 
from load_images import load2d
from lasagne import layers
from nolearn.lasagne import NeuralNet
import pickle 
import numpy as np
import sys
sys.setrecursionlimit(10000)

import theano

def float32(k):
    return np.cast['float32'](k)

from nolearn.lasagne import BatchIterator


class CustBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
      
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
    def apply_rotation_random(self, Xb, yb, changed_indices, rotation_max_angle = 30):
   
        Xb_cols = Xb.shape[2]
        Xb_rows = Xb.shape[3]

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
                temp_yb_frame[int(x), int(y)] = 1
           
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

        
        
        
        
    def transform(self, Xb, yb):
        Xb, yb = super(CustBatchIterator, self).transform(Xb, yb)
        
        batch_shape = Xb.shape[0] 

        # odbicie lustrzane tablic        
        # z np.arange(batch_shape) wez batch_shape / 2 losowych indeksow
        mirror_image_indices = np.random.choice(batch_shape, batch_shape / 2, replace=False)        
        Xb, yb = self.get_mirror_image(Xb, yb, mirror_image_indices)
        
        #
        rotation_indices = np.random.choice(batch_shape, batch_shape , replace=False) 
        #print(rotation_indices)
        rotation_indices = np.array([i for i in range(128)])
        Xb, yb = self.apply_rotation_random(Xb,yb, rotation_indices) 
        
        fig = pyplot.figure(figsize=(1,1))
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plot_sample(Xb[0]*255,yb[0], ax)

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
    conv1_num_filters = 32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
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

    regression=True,
    batch_iterator_train=CustBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=100,
    verbose=1,
    )

X, y = load2d()  
print("images loaded")
net3.fit(X, y)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)

#import orl_faces
#def load_neural_network(path):
#    with open(path, 'rb') as f:
#        nnet = pickle.load(f)
#    return nnet
#from matplotlib import pyplot
#
##def plot_sample(x, y, axis):
##    img = x.reshape(96, 96)
##    axis.imshow(img, cmap='gray')
##    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
#
##orl_faces_reshaped = orl_faces.get_orl_faces_2d_np_arr("C:/Users/Michal/Documents/magisterka/dane/orl_faces/")
##net = load_neural_network('net3.pickle')
##orl_faces_predictions = orl_faces.make_orl_predictions(orl_faces_reshaped, nnet)
##orl_faces.plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16)
#
#
#
##X, _ = load2d(test=True)
##y_pred = nnet.predict(X)
#
##fig = pyplot.figure(figsize=(6, 6))
##fig.subplots_adjust(
##    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
##a = 70
##for i in range(a,a+16):
##    ax = fig.add_subplot(4, 4, i - a + 1, xticks=[], yticks=[])
##    plot_sample(X[i], y_pred[i], ax)
#
###pyplot.show()
##train_loss = np.array([i["train_loss"] for i in net1.train_history_])
##valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
##pyplot.plot(train_loss, linewidth=3, label="train")
##pyplot.plot(valid_loss, linewidth=3, label="valid")
##pyplot.grid()
##pyplot.legend()
##pyplot.xlabel("epoch")
##pyplot.ylabel("loss")
##pyplot.ylim(1e-3, 1e-2)
##pyplot.yscale("log")
##pyplot.show()
#
#from collections import OrderedDict
#
#from sklearn.base import clone
#
#
#SPECIALIST_SETTINGS = [
#    dict(
#        columns=(
#            'left_eye_center_x', 'left_eye_center_y',
#            'right_eye_center_x', 'right_eye_center_y',
#            ),
#        flip_indices=((0, 2), (1, 3)),
#        ),
#
#    dict(
#        columns=(
#            'nose_tip_x', 'nose_tip_y',
#            ),
#        flip_indices=(),
#        ),
#
#    dict(
#        columns=(
#            'mouth_left_corner_x', 'mouth_left_corner_y',
#            'mouth_right_corner_x', 'mouth_right_corner_y',
#            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
#            ),
#        flip_indices=((0, 2), (1, 3)),
#        ),
#
#    dict(
#        columns=(
#            'mouth_center_bottom_lip_x',
#            'mouth_center_bottom_lip_y',
#            ),
#        flip_indices=(),
#        ),
#
#    dict(
#        columns=(
#            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
#            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
#            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
#            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
#            ),
#        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
#        ),
#
#    dict(
#        columns=(
#            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
#            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
#            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
#            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
#            ),
#        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
#        ),
#    ]
#
#
#def fit_specialists(fname_pretrain=None):
#    if fname_pretrain:
#        with open(fname_pretrain, 'rb') as f:
#            net_pretrain = pickle.load(f)
#    else:
#        net_pretrain = None
#
#    specialists = OrderedDict()
#
#    for setting in SPECIALIST_SETTINGS:
#        cols = setting['columns']
#        X, y = load2d(cols=cols)
#
#        model = clone(net)
#        model.output_num_units = y.shape[1]
#        model.batch_iterator_train.flip_indices = setting['flip_indices']
#        model.max_epochs = int(4e6 / y.shape[0])
#        if 'kwargs' in setting:
#            # an option 'kwargs' in the settings list may be used to
#            # set any other parameter of the net:
#            vars(model).update(setting['kwargs'])
#
#        if net_pretrain is not None:
#            # if a pretrain model was given, use it to initialize the
#            # weights of our new specialist model:
#            model.load_params_from(net_pretrain)
#
#        print("Training model for columns {} for {} epochs".format(
#            cols, model.max_epochs))
#        model.fit(X, y)
#        specialists[cols] = model
#
#    with open('net-specialists.pickle', 'wb') as f:
#        # this time we're persisting a dictionary with all models:
#        pickle.dump(specialists, f, -1)
#
#fit_specialists('net3.pickle')