import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from numpy.random import random_sample

# 网络搭建

image_input = tf.keras.Input(shape=(32,32,3,), name="img_input")
timeseries_input = tf.keras.Input(shape=(20,10,), name="ts_input")

x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3,3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)

# 模型构建

model = tf.keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])

# 自定义损失函数

class MyMeanSquaredError_class(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return loss


def MyMeanSquaredError_func(): # 两个方法都可以
    def mean_squared_error(y_pred, y_true):
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return loss
    return mean_squared_error

# 模型配置

model.compile(
    optimizer=optimizers.RMSprop(1e-3), 
    loss={"score_output":MyMeanSquaredError_func(), "class_output":losses.CategoricalCrossentropy(from_logits=True)}, 
    metrics={"score_output":[metrics.MeanAbsolutePercentageError(), metrics.MeanAbsoluteError()], "class_output":[metrics.CategoricalAccuracy()]}
    )

tf.print(model.summary())
# 数据构建

img_data = random_sample(size=(100,32,32,3))
ts_data = random_sample(size=(100,20,10))
score_targets = random_sample(size=(100,1))
class_targets = random_sample(size=(100,5))

# 模型训练

model.fit({"img_input":img_data, "ts_input":ts_data}, {"score_output":score_targets, "class_output":class_targets}, batch_size=32, epochs=5)