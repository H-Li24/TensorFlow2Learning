import tensorflow as tf 
import numpy as np

'''
# one data

dataset1 = tf.data.Dataset.from_tensors(np.zeros(shape=(10,5,2), dtype=np.float32))
flag = 0
for element in dataset1:
    flag += 1
    print(element.shape)
    print(flag)

dataset2 = tf.data.Dataset.from_tensor_slices(np.zeros(shape=(10,5,2), dtype=np.float32))
flag = 0
for element in dataset2:
    flag += 1
    print(element.shape)
    print(flag)
'''
# two data

dataset3 = tf.data.Dataset.from_tensors({"a":np.zeros(shape=(10,5,2), dtype=np.float32), "b":np.zeros(shape=(10,5,2), dtype=np.float32)}) # 传入是字典，传出是规整好的字典
flag = 0
for element in dataset3:
    flag += 1
    print(element["a"].shape, element["b"].shape)
    print(flag)

dataset3 = tf.data.Dataset.from_tensor_slices({"a":np.zeros(shape=(10,5,2), dtype=np.float32), "b":np.zeros(shape=(10,5,2), dtype=np.float32)}) # 传入是字典，传出是规整好的字典
flag = 0
for element in dataset3:
    flag += 1
    print(element["a"].shape, element["b"].shape)
    print(flag)

