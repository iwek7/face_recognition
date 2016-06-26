import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = 'C:/Users/Michal/Documents/magisterka/dane/training.csv'
FTEST = 'C:/Users/Michal/Documents/magisterka/dane/test.csv'

# to cudo zwracawielowymiarowy array numpy
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    # ladowanie danych do dataFrame pandowego 
    df = read_csv(os.path.expanduser(fname))  
    # print(df.columns)
    
    # print(df['Image'][1])
    # Kazdy image jest zakodowany jako pixele oddzielone spacja
    # transformacja do wektora numpy
    # funckja fromstring robi 1d array z tekstu (po separatorze)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # dobranie subsetu kolumn
    if cols:  
        df = df[list(cols) + ['Image']]
 
    # print(df.count())  # prints the number of values for each column

    # usuwanie pustych danych metoda pandowa
    df = df.dropna()  
   
    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state = 42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y
