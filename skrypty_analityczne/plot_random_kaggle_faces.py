from load_images import load
from matplotlib import pyplot
import numpy as np

fig = pyplot.figure(figsize=(3, 3))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
X, y = load()

for i in range(9):
	ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
	random_image_id = np.random.randint(X.shape[0])
	img = X[random_image_id].reshape(96, 96)
	ax.imshow(img, cmap="gray")
	ax.scatter(y[random_image_id][0::2] * 48 + 48, y[random_image_id][1::2] * 48 + 48, marker="o")

pyplot.show()