import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np

mnist = np.load("mnist.npz")
x_train, y_train, x_test, y_test = mnist["x_train"], mnist["y_train"], mnist["x_test"], mnist["y_test"]
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis] # 没有对y做one-hot编码

def MyConvModelKeras():
    inputs = tf.keras.Input(shape=(28,28,1,), name="images")
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = MyConvModelKeras()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

tensorboard_callback = [
    tf.keras.callbacks.TensorBoard(
        log_dir="keras_logv1", 
        histogram_freq=1, 
        profile_batch=100000000
    )
]

model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test), callbacks=tensorboard_callback)