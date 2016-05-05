from orl_faces import OrlFaces
from matplotlib import pyplot
import numpy as np
import pandas as pd
###
# wykres losowych orl faces do magisterki
# razem z keypointami
###

orl_faces = OrlFaces()
orl_faces.load_orl_faces()

fig = pyplot.figure(figsize=(3,3))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

keypoints = pd.read_csv("C:/Users/iwek/Documents/magisterka/FFM/orl_faces_keypoints.csv")


for i in range(9):
	# - 20 poniewaz mam tylko polowe danych oznaczonych
	rand_person_id = np.random.randint(low=0,high=orl_faces.NUM_PPL - 20)
	rand_img_id = np.random.randint(low=0,high=orl_faces.NUM_IMAGES)
	img = orl_faces.orl_images_df.loc[(orl_faces.orl_images_df['person_id'] == 0 + rand_person_id) &
		(orl_faces.orl_images_df['image_id'] == 0 + rand_img_id)]["image"].values[0]

	ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
	ax.imshow(img, cmap='gray')
	# get row of keypoints dataframe
	# select only columns with keypoints data
	img_keypoints = keypoints.loc[(keypoints['person'] == 0 + rand_person_id)  & 
					(keypoints['image'] == 0 + rand_img_id)][keypoints.columns[2:]]
	# turn int to np array
	np_img_keypoints = img_keypoints.as_matrix()[0]

	ax.scatter(np_img_keypoints[0::2], np_img_keypoints[1::2], marker='o')

pyplot.show()