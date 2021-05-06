import tensorflow as tf

# 把tensor当列向量看
a = tf.data.Dataset.range(1,4)
b = tf.data.Dataset.range(4,7)

# zip是横向结合，多个特征结合
ds1 = tf.data.Dataset.zip((a,b))

for line in ds1:
    tf.print(line)

print('\n')
ds2 = tf.data.Dataset.zip((b,a))

for line in ds2:
    tf.print(line)

print('\n')

# concat是纵向结合，多个数据集结合
ds3 = a.concatenate(b)

for line in ds3:
    tf.print(line)

print('\n')

ds4 = b.concatenate(a)

for line in ds4:
    tf.print(line)