import tensorflow as tf
import matplotlib.pyplot as plt

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

flowers = './flower_photos/flower_photos'

def Gen():
    gen = img_gen.flow_from_directory(flowers)
    for (x,y) in gen:
        yield (x,y)

flower_dataset = tf.data.Dataset.from_generator(Gen, output_types=(tf.float32, tf.float32)) # 默认batch是32, 参数output_shapes不会写

for image, label in flower_dataset:
    print(image.shape, label.shape)
    break