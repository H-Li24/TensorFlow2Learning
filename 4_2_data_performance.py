import tensorflow as tf
import time
import os

data_dir = './datasets'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
valid_cats_dir = data_dir + '/valid/cats/'
valid_dogs_dir = data_dir + '/valid/dogs/'

# 构建训练数据集

train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)[:1000]]) # 用了1000张
train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)[:1000]])
train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
train_labels = tf.concat([tf.zeros(train_cat_filenames.shape, dtype=tf.int32),tf.ones(train_dog_filenames.shape,dtype=tf.int32)], axis=-1)

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename) # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string) # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256,256]) / 255.0
    return image_resized, label

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)

'''
benchmark(train_dataset.map(
    map_func=_decode_and_resize
),num_epochs=1)

# 多进程执行
benchmark(train_dataset.map(
    map_func=_decode_and_resize,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
),num_epochs=1)

'''
# prefetch方法
benchmark(train_dataset.map(
    map_func=_decode_and_resize,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .prefetch(tf.data.experimental.AUTOTUNE),
    num_epochs=1)