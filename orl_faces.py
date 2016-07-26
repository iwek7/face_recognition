import re
import numpy
import pandas as pd
import random
import pickle
import csv
from lasagne import layers
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error

class OrlFaces():

    # struktura danych importowanych:
    # person_id
    # image_id
    # train - czy nalezy do zbioru treningowego czy walidacyjnego
    # image

    # number of all people
    NUM_PPL = 40
    # images per person
    NUM_IMAGES = 10

    # komputer stacjonarny "C:/Users/Michal/Documents/magisterka/dane/orl_faces/"
    # laptop prywatny "C:/SciSoft/orl_faces/"
    ORL_FACES_CATALOG = "C:/Users/Michal/Documents/magisterka/dane/orl_faces/"

    KEYPOINT_NAMES = [
        "LEFT_EYE_MIDDLE",
        "RIGHT_EYE_MIDDLE",
        "LEFT_EYE_RIGHT",
        "LEFT_EYE_LEFT",
        "RIGHT_EYE_LEFT",
        "RIGHT_EYE_RIGHT",
        "LEFT_BROW_RIGHT",
        "LEFT_BROW_LEFT",
        "RIGHT_BROW_LEFT",
        "RIGHT_BROW_RIGHT",
        "NOSE",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "MOUTH_TOP",
        "MOUTH_DOWN"]
    
    
    FULL_REARRANGED_KEYPOINTS = ["LEFT_EYE_MIDDLE_X",
                                "LEFT_EYE_MIDDLE_Y",
                                "RIGHT_EYE_MIDDLE_X",
                                "RIGHT_EYE_MIDDLE_Y",
                                "LEFT_EYE_RIGHT_X",
                                "LEFT_EYE_RIGHT_Y",
                                "LEFT_EYE_LEFT_X",
                                "LEFT_EYE_LEFT_Y",
                                "RIGHT_EYE_LEFT_X",
                                "RIGHT_EYE_LEFT_Y",
                                "RIGHT_EYE_RIGHT_X",
                                "RIGHT_EYE_RIGHT_Y",
                                "LEFT_BROW_RIGHT_X",
                                "LEFT_BROW_RIGHT_Y",
                                "LEFT_BROW_LEFT_X",
                                "LEFT_BROW_LEFT_Y",
                                "RIGHT_BROW_LEFT_X",
                                "RIGHT_BROW_LEFT_Y",
                                "RIGHT_BROW_RIGHT_X",
                                "RIGHT_BROW_RIGHT_Y",
                                "NOSE_X",
                                "NOSE_Y",
                                "MOUTH_LEFT_X",
                                "MOUTH_LEFT_Y",
                                "MOUTH_RIGHT_X",
                                "MOUTH_RIGHT_Y",
                                "MOUTH_TOP_X",
                                "MOUTH_TOP_Y",
                                "MOUTH_DOWN_X",
                                "MOUTH_DOWN_Y"]
        
    def read_pgm(self, filename, byteorder='>'):
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

    def load_orl_faces(self, file_name = "s", num_ppl = 40, num_images = 10,
                           num_train = 7):
        cols_names =  ['person_id','image_id', 'train', 'image']                    
        orl_images_df = pd.DataFrame(columns=cols_names)
                                 
        for person in range(num_ppl):
            # z gory ustalamy co bedzie walidacyjne a co treningowe
            img_subset = ([1 for i in range(num_train)] + 
                        [0 for i in range(num_images - num_train)])
            random.shuffle(img_subset) 
            for img_idx in range(num_images):
                path = (self.ORL_FACES_CATALOG + file_name + str(person + 1) + "/" + 
                    str(img_idx + 1) + ".pgm"
                        )
                # obrazki maja fromat 112x92, siec jest wytrenowana na 96x96
                # tymczasowe trywialne rozwiazanie : ucinamy nadliczbowe pixele z pionu (po polowie gora i dol)
                # dodajemy puste (0) paski z lewej i prawej (po 2 z kazdej strony o szerokosci pixela)
                #image = self.read_pgm(path)
                image = self.read_pgm(path)[8 : 112 - 8,]
                image = numpy.insert(image,2,92+numpy.zeros((2,image.shape[0])),1)
                image = numpy.insert(image,image.shape[1],numpy.zeros((2,image.shape[0])),1)

                orl_images_df= orl_images_df.append(pd.DataFrame(
                                [[person, img_idx, img_subset[img_idx], image]], 
                                columns=cols_names))
            
        self.orl_images_df = orl_images_df


    # dodaje dodatkowy wymiar potrzebny do feedowania tych danych do conv_neta
    # ponadto wrzucam wszystkie obrazki do ndarray to przyjac siec neuronowa
    def laod_orl_faces_2d_np_arr(self):
        orl_faces = self.load_orl_faces()
        self.orl_faces_reshaped = numpy.vstack(self.orl_images_df["image"].values) / 255
        self.orl_faces_reshaped = self.orl_faces_reshaped.astype(numpy.float32)
        self.orl_faces_reshaped = self.orl_faces_reshaped.reshape(-1, 1, 96, 96)

    # 'net2.pickle'
    def make_orl_predictions(self, nnet, reshaped = True):
        if reshaped:
            self.orl_predictions = nnet.predict(self.orl_faces_reshaped)
        else:
            self.orl_predictions = nnet.predict(self.orl_faces)      

    def save_orl_predictions(self, path):
        index = ['Row'+str(i) for i in range(1, len(self.orl_predictions)+1)]
        output_df = pd.DataFrame(
            data = self.orl_predictions,
            index = index,
            columns = self.FULL_REARRANGED_KEYPOINTS)

        # add columns with person and image indexing
        #img_indices = [i for j in range(self.NUM_PPL) 
        #                    for i in range(self.NUM_IMAGES)]

        #ppl_indices = [j for j in range(self.NUM_PPL) 
        #                    for i in range(self.NUM_IMAGES)]

        #output_df['person'] = ppl_indices
        #output_df['image'] = img_indices
        output_df.to_csv(path)                  

    def plot_sample(self, x, y, axis): 
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
       
    def plot_orl_predictions(self, reshaped = True, limit = 16, start_index = 0):
        
        fig = pyplot.figure(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(start_index,start_index+limit):
            ax = fig.add_subplot(4, 4, i - start_index + 1, xticks=[], yticks=[])
            if reshaped:
                self.plot_sample(self.orl_faces_reshaped[i][0], self.orl_predictions[i], ax)
            else:
                self.plot_sample(self.orl_faces[i], self.orl_predictions[i], ax)
        pyplot.show()

    def load_orl_keypoints(self, path):
        # lapek C:/Users/iwek/Documents/magisterka/FFM
        # stacjonarny C:/Users/Michal/Documents/Visual Studio 2013/Projects/faceFeaturesMarker/faceFeaturesMarker/orl_faces_keypoints.csv
        self.orl_keypoints = pd.DataFrame.from_csv(path,index_col=False)
        self.orl_keypoints = self.orl_keypoints[self.FULL_REARRANGED_KEYPOINTS]

    def save_rearranged_keypoints_and_predictions(self, net_name = "", path = ""):
         y = self.orl_keypoints.values.astype('float64')
         y = (y - 48) / 48
         numpy.savetxt(path + net_name + "_keypoints_rearranged.csv", y, delimiter=",")
         numpy.savetxt(path + net_name + "_predictions_rearranged.csv", self.orl_predictions, delimiter=",")

    def load_orl_predictions(self, path):
        self.orl_predictions = pd.DataFrame.from_csv(path)

    def calculate_total_error(self):
        y = self.orl_keypoints.values.astype('float64')
        y = (y - 48) / 48
        
        return numpy.sqrt(mean_squared_error(self.orl_predictions, y)) * 48

    def calculate_error_by_keypoints(self):
        y = self.orl_keypoints.values.astype('float64')
        y = (y - 48) / 48

        list_of_errors = numpy.zeros(shape=y.shape[1])
        for keypoint in range(y.shape[1]):
            list_of_errors[keypoint] = numpy.sqrt(mean_squared_error(
                                    y[:,keypoint], 
                                    self.orl_predictions.values[:,keypoint]
                                    )) * 48
        return list_of_errors

#orl_faces_reshaped = get_orl_faces_2d_np_arr("C:/Users/Michal/Documents/magisterka/dane/orl_faces/")
#nnet = load_neural_network('net3.pickle')
#orl_faces_predictions = make_orl_predictions(orl_faces_reshaped, nnet)
#plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16)



