from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metric=["accuracy"])

data = np.random.random((1000,32))
labels = np.random.random((1000,10))

model.fit(data, labels, epochs=10, batch_size=32)