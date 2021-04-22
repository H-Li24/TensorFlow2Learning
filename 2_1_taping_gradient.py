import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from numpy.random import random_sample

# 网络搭建

image_input = tf.keras.Input(shape=(32,32,3), name="img_input")
timeseries_input = tf.keras.Input(shape=(20,10), name="ts_input")

x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3,3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)

# 模型(对象)构建

model = tf.keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])
loss_score_object = losses.MeanSquaredError()
loss_class_object = losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 数据构建

img_data = random_sample(size=(100,32,32,3))
ts_data = random_sample(size=(100,20,10))
score_targets = random_sample(size=(100,1))
class_targets = random_sample(size=(100,5))

# 使用Tape进行一步参数更新

with tf.GradientTape() as tape:
    [score_predict, class_predict] = model({"img_input":img_data, "ts_input":ts_data})
    loss_score = loss_score_object(score_targets, score_predict)
    loss_class = loss_class_object(class_targets, class_predict)
    loss = loss_score + loss_class

gradients = tape.gradient(loss, model.trainable_weights)
optimizer.apply_gradients(zip(gradients, model.trainable_weights))

tf.print(model.trainable_weights)