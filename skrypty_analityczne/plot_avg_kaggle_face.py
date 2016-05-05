from load_images import load
from matplotlib import pyplot
import numpy as np

X, y = load()

avg_kaggle_image = np.zeros((96, 96))

for i in range(X.shape[0]):	
	avg_kaggle_image += X[i].reshape(96, 96)

avg_kaggle_image /= X.shape[0]
fig = pyplot.figure(figsize=(1, 1))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
ax.imshow(avg_kaggle_image, cmap="gray")
pyplot.show()