from sklearn import datasets
import tensorflow as tf

iris = datasets.load_iris()

data = iris.data
target = iris.target

class Linear(tf.keras.layers.Layer): # 继承add_weight方法来进行初始化
    def __init__(self, input_dims=4):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dims, 1,), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

x = tf.constant(data)
linear_layer = Linear()
y_pred = linear_layer(x)
tf.print(y_pred)
