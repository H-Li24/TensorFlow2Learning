from sklearn import datasets
import tensorflow as tf
import numpy as np

# custom layer

class MyDense(tf.keras.layers.Layer): # 自定义层：如果输入数据的维数不知道，使用build方法
    def __init__(self, units=32, **kwargs): # 传入可变参数
        super(MyDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units,), initializer="random_normal", trainable=True, name="w") # 如果变量不命名，模型保存save会报错
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True, name="b")
        super(MyDense,self).build(input_shape)

    def call(self, inputs):
        predictions = tf.matmul(inputs, self.w) + self.b
        return predictions

    def get_config(self):
        config = super(MyDense, self).get_config()
        config.update({"units":self.units})
        return config

# functional model building

inputs = tf.keras.Input(shape=(4,))
x = MyDense(units=16)(inputs)
x = tf.nn.tanh(x)
x = MyDense(units=3)(x)
predictions = tf.nn.softmax(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# data
iris = datasets.load_iris()

data = iris.data
labels = iris.target

data = np.concatenate((data, labels.reshape(150,1)), axis=-1)
np.random.shuffle(data)

labels = data[:,-1]
data = data[:,:4]

print(labels)

# training

model.fit(data, labels, batch_size=32, epochs=100, shuffle=True)

# save model

model.save("./checkpoints/model_h5/model_keras_custom.h5")

print("variable:", model.weights) # 你也可以看model的权重
print("non-trainable variable:", model.non_trainable_weights)
print("trainable variable:", model.trainable_weights)

_custom_objects = {
    "MyDense" : MyDense
}

# load model

new_model = tf.keras.models.load_model("./checkpoints/model_h5/model_keras_custom.h5", custom_objects=_custom_objects)

y_test = new_model.predict(data)

print(np.argmax(y_test, axis=1))
print(np.shape(y_test))