import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from numpy.random import random_sample


class single_in_single_out(tf.keras.Model): # 自定义网络
    def __init__(self, number_classes=10):
        super(single_in_single_out, self).__init__(name="my_model")
        self.number_classes = number_classes
        self.dense_1 = layers.Dense(64, activation="relu")
        self.dense_2 = layers.Dense(number_classes, activation="softmax")
    
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

model = single_in_single_out(number_classes=10)
loss_object = losses.CategoricalCrossentropy()
optimizer = optimizers.SGD(1e-3)

data = random_sample((1000,64))
labels = random_sample((1000,10))

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((data, labels)) # 建立一一对应的数据集对象
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 5
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x_batch_predict = model(x_batch_train, training=True)
            loss = loss_object(y_batch_train, x_batch_predict)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        if step % 200 == 0:
            print("Training loss (for one batch) at step %s : %s" % (step, float(loss))) # 或者是loss.numpy()

