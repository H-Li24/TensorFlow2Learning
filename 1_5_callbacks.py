import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

inputs1 = tf.keras.Input(shape=(32,), name="input_1")
inputs2 = tf.keras.Input(shape=(32,), name="input_2")
x1 = layers.Dense(64, name="dense_1")(inputs1)
x2 = layers.Dense(64,name="dense_2")(inputs2)
print(tf.shape(x1))
x = tf.concat([x1, x2], axis=-1)
print(tf.shape(x))
outputs1 = layers.Dense(10,name="pred_1")(x)
outputs2 = layers.Dense(10,name="pred_2")(x)

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

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/model_callbacks/mymodel_{epoch}",
        save_best_only=False,
        monitor="val_loss",
        save_weights_only=True,
        verbose=1
    )
]


model.fit((x_1_train, x_2_train), (y_1_train, y_2_train), batch_size=64, epochs=5, validation_split=0.2, callbacks=callbacks)