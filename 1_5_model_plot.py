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

# 模型构建

model = tf.keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])

# 绘制模型数据流图

tf.keras.utils.plot_model(model, "multi_io_model.png", show_shapes=True, dpi=500)