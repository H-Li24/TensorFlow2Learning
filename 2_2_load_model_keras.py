from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

inputs1 = tf.keras.Input(shape=(32,))
inputs2 = tf.keras.Input(shape=(32,))
x1 = layers.Dense(64, activation="relu")(inputs1)
x2 = layers.Dense(64, activation="relu")(inputs2)
x = layers.concatenate([x1,x2])
outputs1 = layers.Dense(10, activation=tf.nn.relu)(x)
outputs2 = layers.Dense(10, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[outputs1,outputs2])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

tf.print(model.summary())

data1 = np.random.random((1000,32))
data2 = np.random.random((1000,32))
labels1 = np.random.random((1000,10))
labels2 = np.random.random((1000,10))

test1 = np.random.random((1000,32))
test2 = np.random.random((1000,32))

# model.load_weights("./checkpoints/model_keras/mykerasmodel")
model.load_weights("./checkpoints/model_h5/weight_keras.h5") # pip install h5py==2.10.0
tf.print(model.predict([test1, test2]))