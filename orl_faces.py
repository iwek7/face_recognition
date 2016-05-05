import re
import numpy
import pandas as pd
import random
import pickle
import csv
from lasagne import layers
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot

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
    ORL_FACES_CATALOG = "C:/SciSoft/orl_faces/"

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
        keypoints_column_names = [keypoint + '_' + coor 
                                            for keypoint in self.KEYPOINT_NAMES 
                                            for coor in ('X','Y')]
        index = ['Row'+str(i) for i in range(1, len(self.orl_predictions)+1)]
        output_df = pd.DataFrame(
            data = self.orl_predictions,
            index = index,
            columns = keypoints_column_names)

        # add columns with person and image indexing
        img_indices = [i for j in range(self.NUM_PPL) 
                            for i in range(self.NUM_IMAGES)]

        ppl_indices = [j for j in range(self.NUM_PPL) 
                            for i in range(self.NUM_IMAGES)]

        output_df['person'] = ppl_indices
        output_df['image'] = img_indices
        output_df.to_csv(path)                  

    def plot_sample(self, x, y, axis): 
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    # plot predicted points over faces
    def plot_orl_faces(self, reshaped = True, limit = 16, start_index = 0):
        
        fig = pyplot.figure(figsize=(6, 6))
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(start_index,start_index+limit):
            ax = fig.add_subplot(4, 4, i - start_index + 1, xticks=[], yticks=[])
            if reshaped:
                self.plot_sample(self.orl_faces_reshaped[i], self.orl_predictions[i], ax)
            else:
                self.plot_sample(self.orl_faces[i], self.orl_predictions[i], ax)
        pyplot.show()

    def load_orl_keypoints(self):
        orl_keypoins_path = "C:/Users/Michal/Documents/magisterka/orl_faces_keypoints.csv"
        self.orl_keypoints = pd.DataFrame.from_csv(orl_keypoins_path)

    def calculate_prediction_errors(self):
        
        #print(list(self.orl_keypoints.columns.values))
         # add columns with person and image indexing
        img_indices = [i for j in range(self.NUM_PPL) 
                            for i in range(self.NUM_IMAGES)]

        ppl_indices = [j for j in range(self.NUM_PPL) 
                            for i in range(self.NUM_IMAGES)]


        keypoints_column_names = [keypoint + '_' + coor 
                                            for keypoint in self.KEYPOINT_NAMES 
                                            for coor in ('X','Y')]
        self.MSE_table = pd.DataFrame(columns = keypoints_column_names)
        self.MSE_table['person'] = ppl_indices
        self.MSE_table['image'] = img_indices
        # tu zle
        self.MSE_table.loc[self.MSE_table['person'] == 1, 'LEFT_EYE_MIDDLE'] = 11

        index = ['Row'+str(i) for i in range(1, len(self.orl_predictions)+1)]
        predictions_df = pd.DataFrame(
            data = self.orl_predictions,
            index = index,
            columns = keypoints_column_names)
        predictions_df['person'] = ppl_indices
        predictions_df['image'] = img_indices

        for person_num in range(self.NUM_PPL):
            for img_num in range(self.NUM_IMAGES):
                for keypoit in self.KEYPOINT_NAMES:
                    self.MSE_table.loc[self.MSE_table['person'] == person_num and 
                                       self.MSE_table['image'] == img_num, 
                                       keypoit] = (
                     predictions_df.loc[predictions_df['person'] == person_num and 
                                       predictions_df['image'] == img_num][keypoit] -
                     self.orl_keypoints[ self.orl_keypoints['person'] == person_num and 
                                        sself.orl_keypoints['image'] == img_num][keypoit]) ** 2




        print(self.MSE_table.loc[self.MSE_table['person'] == 1])




#orl_faces_reshaped = get_orl_faces_2d_np_arr("C:/Users/Michal/Documents/magisterka/dane/orl_faces/")
#nnet = load_neural_network('net3.pickle')
#orl_faces_predictions = make_orl_predictions(orl_faces_reshaped, nnet)
#plot_orl_predictions(orl_faces_reshaped, orl_faces_predictions, 16)



