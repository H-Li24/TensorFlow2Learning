import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
import datetime
import os

mnist = np.load("mnist.npz")
x_train, y_train, x_test, y_test = mnist["x_train"], mnist["y_train"], mnist["x_test"], mnist["y_test"]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyConvModel(tf.keras.Model):
    def __init__(self):
        super(MyConvModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.fc1 = Dense(10, activation="softmax")

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        predictions = self.fc1(x)
        return predictions

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss") #生成估函数对象
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape: # 定义函数关系
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_loss(loss) # 其实不是很理解评估函数的原理，不知道传了loss进去会修改些什么
    train_accuracy(labels, predictions) # 这个是先label，后prediction

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

model = MyConvModel()

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs/"+stamp)

summary_writer = tf.summary.create_file_writer(logdir)

EPOCHS = 20

for epoch in range(EPOCHS):
    for (x_train, y_train) in train_ds:
        train_step(x_train, y_train)

    with summary_writer.as_default():
        tf.summary.scalar("loss", train_loss.result(), step=epoch)
        tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

    template = "Epoch:{}, Train Loss:{}, Test Loss:{}， Train Accuracy:{}, Test Accuracy:{}"
    print(template.format(epoch +1, train_loss.result(), test_loss.result(), train_accuracy.result()*100, test_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()

with summary_writer.as_default():
    tf.summary.trace_on(graph=True, profiler=True) #开启trace，可以记录图结构和profile信息

    tf.summary.trace_export(name="model_trace", step=3, profiler_outdir=logdir) # 保存trace信息到文件