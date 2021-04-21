from sklearn import datasets
import tensorflow as tf

iris = datasets.load_iris()

data = iris.data
target = iris.target

class Linear(tf.keras.layers.Layer): # 继承这个类，就可以做类似1_3_layers_demo的计算了
    def __init__(self, input_dims=4):
        super(Linear, self).__init__()
        w_init_object = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init_object(shape=(input_dims,1,), dtype="float32"), trainable=True) # shape=(input_dims,1,)
        b_init_object = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init_object(shape=(1,), dtype="float32"), trainable=True)

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

x = tf.constant(data)
linear_layer = Linear()
y_pred = linear_layer(x)
tf.print(y_pred)
