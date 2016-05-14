from load_images import load
from matplotlib import pyplot
import numpy as np
from orl_faces import OrlFaces





orl_faces = OrlFaces()
orl_faces.laod_orl_faces_2d_np_arr()
orl_faces.load_orl_keypoints("C:/Users/iwek/Documents/magisterka/FFM/orl_faces_keypoints.csv")
orl_faces.load_orl_predictions("C:/Users/iwek/Documents/magisterka/FFM/predictions_test.csv")

orl_faces.calculate_prediction_errors()
print(orl_faces.orl_faces_reshaped.shape)
