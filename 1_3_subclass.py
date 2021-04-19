import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class my_test_model(tf.keras.Model): # 传入tf.keras.Model，Model的M要大写
    def __init__(self, class_number=10): # 你要把self传进来你才可以定义self的东西
        super(my_test_model, self).__init__(name="my_model_name") # super,传入类名和self，.__init__初始化，设置模型名
        self.dense_1 = layers.Dense(32, activation="relu") # layers.Dense Dense要大写
        self.dense_2 = layers.Dense(class_number) # 你定义的是属性，所以是self.dense_2 = ...

    def call(self, inputs):# 你要把self传进来才可以用__init__里面self定义好的东西
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x # return 要写在函数里面

My_Test_Model = my_test_model(class_number=10) # 实例化的时候，传入的变量要指定参数名

My_Test_Model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

data = np.random.random((1000,32))

labels = np.random.random((1000,10))

My_Test_Model.fit(data, labels, batch_size=32, epochs=5) # 输入，输出，batch size，epoch数