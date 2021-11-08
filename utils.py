'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-08 12:32:26
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import settings

def normalization(image):
    mean = tf.constant(settings.ImageNet_mean)
    std = tf.constant(settings.ImageNet_std)
    return (image - mean) / std

def load_image(image_path, width=settings.WIDTH, height=settings.HEIGHT):
    x = tf.io.read_file(image_path)
    # decode
    x = tf.image.decode_jpeg(x, channels=3)
    # resize
    x = tf.image.resize(x, [height, width])
    # compression
    x = x / 255
    # normalization
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    return x

def save_image(image, image_name):
    mean = tf.constant(settings.ImageNet_mean)
    std = tf.constant(settings.ImageNet_std)
    x = tf.reshape(image, image.shape[1:])
    x = x * std + mean
    x = x * 255
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(image_name, x)

def generage_noise_image(content_image):
    noise = np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))
    return tf.Variable((content_image + noise) / 2)

def get_vgg19(layers):
    # load pre-trained vgg19 based on imagenet
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # extract the output of needed layers
    outputs = [vgg.get_layer(layer).output for layer in layers]
    # use the outputs to constuct new model
    model = tf.keras.Model([vgg.input, ], outputs)
    # mute the parameters, doesn't use it by train
    model.trainable = False
    return model

def visualize_loss(out_put_dir, losses):
    plt.plot(list(range(len(losses))), losses, color='blue')
    plt.xlabel('Epoch(s)')
    plt.ylabel('Loss(es)')
    plt.title("Losses in training process")
    plt.savefig(f'{out_put_dir}/training_loss.png')

    plt.show()
