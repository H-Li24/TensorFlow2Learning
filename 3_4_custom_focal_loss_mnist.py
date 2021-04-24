import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np

mnist = np.load("mnist.npz")
x_train, y_train, x_test, y_test = mnist["x_train"], mnist["y_train"], mnist["x_test"], mnist["y_test"]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel
x_train = x_train[..., tf.newaxis] # (28,28) -> (28,28,1)
x_test = x_test[..., tf.newaxis]

# one-hot encode 对类别标签进行one-hot编码
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyConvModel(tf.keras.Model):
    def __init__(self):
        super(MyConvModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu") # 全连接层只接受向量
        self.fc1 = Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        predictions = self.fc1(x)
        return predictions

def MyConvModelKeras():
    inputs = tf.keras.Input(shape=(28,28,1,), name="images")
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1) # -1表示最后一个轴，假设是列向量或者是按列分块的矩阵
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.cast(y_true, dtype=tf.float32)
        loss = - y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=-1)
        return loss

def FocalLoss_func(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1) # -1表示最后一个轴，假设是列向量或者是按列分块的矩阵
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.cast(y_true, dtype=tf.float32)
        loss = - y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=-1)
        return loss
    return focal_loss

model = MyConvModelKeras()
model.compile(loss=FocalLoss_func(gamma=2.0, alpha=0.25), optimizer=tf.keras.optimizers.Adam(), metrics = [tf.keras.metrics.CategoricalAccuracy()])

'''
loss_object = FocalLoss(gamma=2.0, alpha=0.25)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
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

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = "Epoch:{}, Train Loss:{}, Test Loss:{}， Train Accuracy:{}, Test Accuracy:{}"
    print(template.format(epoch +1, train_loss.result(), test_loss.result(), train_accuracy.result()*100, test_accuracy.result()*100))
'''

model.fit(train_ds, validation_data=test_ds, epochs=5) # 不写validation_data会报错