import re
import numpy
import pandas as pd
import random
import pickle
from lasagne import layers
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot

# struktura danych importowanych:
# person_id
# image_id
# train - czy nalezy do zbioru treningowego czy walidacyjnego
# image

# number of all people
NUM_PPL = 40
# images per person
NUM_IMAGES = 10

def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def load_orl_faces(catalog_loc, file_name = "s", num_ppl = 40, num_images = 10,
                       num_train = 7):
    cols_names =  ['person_id','image_id', 'train', 'image']                    
    orl_images_df = pd.DataFrame(columns=cols_names)
                                 
    for person in range(1, num_ppl + 1):
        # z gory ustalamy co bedzie walidacyjne a co treningowe
        img_subset = ([1 for i in range(num_train)] + 
                    [0 for i in range(num_images - num_train)])
        random.shuffle(img_subset) 
        for img_idx in range(1, num_images + 1):
            path = (catalog_loc + file_name + str(person) + "/" + 
                str(img_idx) + ".pgm"
                    )
            # obrazki maja fromat 112x92, siec jest wytrenowana na 96x96
            # tymczasowe trywialne rozwiazanie : ucinamy nadliczbowe pixele z pionu (po polowie gora i dol)
            # dodajemy puste (0) paski z lewej i prawej (po 2 z kazdej strony o szerokosci pixela)
            image = read_pgm(path)[8 : 112 - 8,]
            image = numpy.insert(image,2,92+numpy.zeros((2,image.shape[0])),1)
            image = numpy.insert(image,image.shape[1],numpy.zeros((2,image.shape[0])),1)

            orl_images_df= orl_images_df.append(pd.DataFrame(
                            [[person, img_idx, img_subset[img_idx - 1], image]], 
                            columns=cols_names))
            
    return orl_images_df


# dodaje dodatkowy wymiar potrzebny do feedowania tych danych do conv_neta
# ponadto wrzucam wszystkie obrazki do ndarray to przyjac siec neuronowa
def get_orl_faces_2d_np_arr(file_path):
    orl_faces = load_orl_faces(file_path)
    orl_faces_reshaped = numpy.vstack(orl_faces["image"].values) / 255
    orl_faces_reshaped = orl_faces_reshaped.astype(numpy.float32)
    orl_faces_reshaped = orl_faces_reshaped.reshape(-1, 1, 96, 96)
    return orl_faces_reshaped


#==============================================================================
#pyplot.imshow(orl_faces.iloc[22]['image'], pyplot.cm.gray)
#pyplot.show()
#==============================================================================

# wczytuje siec neuronowa z pliku
def load_neural_network(path):
    with open(path, 'rb') as f:
        nnet = pickle.load(f)
    return nnet

# 'net2.pickle'
def make_orl_predictions(orl_faces, nnet):
    return nnet.predict(orl_faces)


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

# plot predicted points over faces
def plot_orl_predictions(orl_faces, orl_predictions, limit = 16):
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    a = 0
    for i in range(a,a+16):
        ax = fig.add_subplot(4, 4, i - a + 1, xticks=[], yticks=[])
        plot_sample(orl_faces[i], orl_predictions[i], ax)

    pyplot.show()

#orl_faces_reshaped = get_orl_faces_2d_np_arr("C:/Users/Michal/Documents/magisterka/dane/orl_faces/")
#nnet = load_neural_network('net3.pickle')
#orl_faces_predictions = make_orl_predictions(orl_faces_reshaped, nnet)
#plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16)



