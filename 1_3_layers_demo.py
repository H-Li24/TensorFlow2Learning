import tensorflow as tf

a = tf.random.uniform((10,50,100), 0, 1)

x = tf.keras.layers.LSTM(100)(a)

x = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

tf.print(x)