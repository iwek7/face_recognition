
from matplotlib import pyplot
import numpy as np
import pickle


# load nets to compare
with open('net1.pickle', 'rb') as f:
    net_one_hidden_layer = pickle.load(f)









#==============================================================================
train_loss_net_1 = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss_net_1 = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss_net_1, linewidth=2, label="train")
pyplot.plot(valid_loss_net_1, linewidth=2, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()

#==============================================================================
#==============================================================================

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
y_pred = net1.predict(X)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()