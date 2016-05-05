from orl_faces import OrlFaces
from matplotlib import pyplot
import numpy as np


###
# przecietna twarz orl faces
###

orl_faces = OrlFaces()
orl_faces.load_orl_faces()

average_orl_image = np.zeros((96,96))
for person_num in range(orl_faces.NUM_PPL):
	for image_num in range(orl_faces.NUM_IMAGES):
		average_orl_image = average_orl_image + ( 
				orl_faces.orl_images_df.loc[(orl_faces.orl_images_df['person_id'] == person_num)  &
					(orl_faces.orl_images_df['image_id'] == image_num )]["image"].values[0]
			)
average_orl_image = average_orl_image / (orl_faces.NUM_PPL * orl_faces.NUM_IMAGES)
fig = pyplot.figure(figsize=(1,1))

# skala szarosci
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
ax.imshow(average_orl_image, cmap="gray")

# kolormapy - http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
#ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
#ax.imshow(average_orl_image, cmap="afmhot")

pyplot.show()