import tensorflow as tf
'''
print(tf.__version__)

print(tf.test.is_gpu_available())
'''
mammal = tf.Variable("elephant", tf.string)

mammal_rank = tf.rank(mammal)

mammal_shape = tf.shape(mammal)

tf.print(tf.shape(mammal))

print(mammal)

print(mammal_rank)

print(mammal_shape)

print("helloworld")

mystr = tf.Variable(["hello"], tf.string)

tf.print(tf.shape(mystr))

my2dtensor = tf.Variable([[1,2,3,4],[5,6,7,8]], dtype = tf.float32)

tf.print(tf.shape(my2dtensor))

myconstant = tf.constant([[1,2,3,4],[3,4,5,6]], tf.float32)

tf.print(tf.shape(myconstant))

tf.print(myconstant)

myzeros = tf.zeros((3,4,5), tf.float32)

tf.print(myzeros)

mysummation = tf.reduce_sum(my2dtensor, axis = 1)

tf.print(mysummation)

tf.print(mysummation.get_shape())

myzeros_shape = myzeros.get_shape()

tf.print(myzeros_shape)

reshape2dtensor = tf.reshape(my2dtensor, (2,2,2))

tf.print(reshape2dtensor)

myones = tf.ones(my2dtensor.get_shape(), tf.float32)

tf.print(my2dtensor.dtype)

tf.print(myones.dtype)

# mymatmul = tf.matmul(my2dtensor, myones, transpose_b = True)

# tf.print(mymatmul)

tf.print(tf.shape(reshape2dtensor))

tf.print(reshape2dtensor)

mysummation2 = tf.reduce_sum(reshape2dtensor, axis = 1)

tf.print(mysummation2)

mytimes = tf.cast(mysummation2, dtype = tf.float32) * tf.constant([0.1], dtype = tf.float32)

tf.print(mytimes)

tf.print(reshape2dtensor)

slice2dtensor = reshape2dtensor[:,:,0]

tf.print(slice2dtensor)

myuniram = tf.random.uniform((3,3), 0, 1)

tf.print(myuniram)

tf.print(my2dtensor)

tf.print(myones)

mymatmul2 = tf.math.multiply(my2dtensor, myones)

tf.print(mymatmul2)