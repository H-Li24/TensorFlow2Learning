import tensorflow as tf

titanic_lines = tf.data.TextLineDataset(['train.csv', 'eval.csv']) # 还在磁盘里，没有进内存

def data_func(line):
    line = tf.strings.split(line, sep=',')
    return line

titanic_data = titanic_lines.skip(1).map(data_func)

for line in titanic_data:
    print(line)
    break