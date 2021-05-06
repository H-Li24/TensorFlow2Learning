import tensorflow as tf

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256 ,256]) / 255.0 # tf.image.resize出来就是一个矩阵了，然后像素值归一化
    return feature_dict['image'], feature_dict['label']

train_dataset = tf.data.TFRecordDataset('sub_train.tfrecords')

train_dataset = train_dataset.map(_parse_example)

for image, label in train_dataset:
    print(image, label)
    break