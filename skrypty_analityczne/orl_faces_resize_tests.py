from PIL import Image
from matplotlib import pyplot
from orl_faces import OrlFaces
import numpy as np
# http://matplotlib.org/users/image_tutorial.html
## aby to działało trzeba wyłączyć resize w ladowaniu orl faces
orl_faces = OrlFaces()
orl_faces.load_orl_faces()


img = Image.fromarray(orl_faces.orl_images_df.loc[
	(orl_faces.orl_images_df['person_id'] == 0) &
		(orl_faces.orl_images_df['image_id'] == 0 )]["image"].values[0]
		)


fig = pyplot.figure(figsize=(1,2))

# img.thumbnail((96,96), Image.ANTIALIAS)
img = img.resize((96,96), Image.ANTIALIAS)
ax = fig.add_subplot(1,2,1, xticks=[], yticks=[])
ax.imshow(img, cmap="gray")

img2 = Image.fromarray(orl_faces.orl_images_df.loc[
	(orl_faces.orl_images_df['person_id'] == 0) &
		(orl_faces.orl_images_df['image_id'] == 0 )]["image"].values[0]
		)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax.imshow(img2, cmap="gray")




pyplot.show()
