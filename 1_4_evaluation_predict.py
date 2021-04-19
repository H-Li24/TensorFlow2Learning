import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

inputs1 = tf.keras.Input(shape=(32,))
inputs2 = tf.keras.Input(shape=(32,))
x1 = layers.Dense(64)(inputs1)
x2 = layers.Dense(64)(inputs2)
print(tf.shape(x1))
x = tf.concat([x1, x2], axis=-1)
print(tf.shape(x))
outputs1 = layers.Dense(10)(x)
outputs2 = layers.Dense(10)(x)

model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[outputs1, outputs2])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

x_1_train = np.random.random((1000,32))
x_2_train = np.random.random((1000,32))
y_1_train = np.random.random((1000,10))
y_2_train = np.random.random((1000,10))

x_1_eval = np.random.random((200,32))
x_2_eval = np.random.random((200,32))
y_1_eval = np.random.random((200,10))
y_2_eval = np.random.random((200,10))

x_1_test = np.random.random((100,32))
x_2_test = np.random.random((100,32))
y_1_test = np.random.random((100,10))
y_2_test = np.random.random((100,10))


model.fit((x_1_train, x_2_train), (y_1_train, y_2_train), batch_size=64, epochs=5, validation_data=((x_1_eval, x_2_eval), (y_1_eval, y_2_eval)))

eval_results = model.evaluate((x_1_test, x_2_test), (y_1_test, y_2_test), batch_size=128)

x_1_pred = np.random.random((50, 32))
x_2_pred = np.random.random((50, 32))

predicts = model.predict((x_1_pred, x_2_pred))

print(predicts)