import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([[1,2,3],[4,5,6],[7,8,9]])

for element_1 in dataset:
    print(element_1)

dataset_flat = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) # 如果这里面的操作涉及lambda函数，必须用flat_map

for element_2 in dataset_flat:
    print(element_2)

a = tf.data.Dataset.range(1,6)
b = a.flat_map(lambda x:tf.data.Dataset.from_tensors(x).repeat(6))

for item in b:
    print(item.numpy(), end=',')