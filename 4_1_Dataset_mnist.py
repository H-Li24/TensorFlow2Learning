import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = np.load("mnist.npz")
img_train = mnist['x_train']
label_train = mnist['y_train']

img_train = np.expand_dims(img_train, axis=-1)
label_train = np.expand_dims(label_train, axis=-1)
print(img_train.shape)
print(label_train.shape)

mnist_dataset = tf.data.Dataset.from_tensor_slices((img_train, label_train))

flag = 0
for img, label in mnist_dataset.take(5):
    print(img.shape)
    print(label)
    flag += 1
    print(flag)
    plt.imshow(img)
    plt.show()
    break