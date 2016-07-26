import numpy as np
from scipy import ndimage
from sklearn.utils import shuffle

from nolearn.lasagne import BatchIterator
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
