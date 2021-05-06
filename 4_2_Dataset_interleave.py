import tensorflow as tf

filenames = ["./interleave_data/train.csv", "./interleave_data/eval.csv", "./interleave_data/train.csv", "./interleave_data/eval.csv"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

def data_func(line):
    line = tf.strings.split(line, sep=',')
    return line

# 读取多个csv文件
dataset_1 = dataset.interleave(lambda x:
    tf.data.TextLineDataset(x).skip(1).map(data_func),
    cycle_length=4, block_length=16
)

for line in dataset_1.take(2):
    print(line)