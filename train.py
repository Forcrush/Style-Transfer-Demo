'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-07 00:20:33
'''

import os
import tensorflow as tf
import settings
import utils
import model
from tqdm import tqdm



# compute the content loss of current image
def compute_content_loss(noise_content_features, content_image_features):
    content_losses = []
    M, N = settings.WIDTH * settings.HEIGHT, 3
    for (noise_feature, weight), (content_feature, _) in zip(noise_content_features, content_image_features):
        content_loss = tf.reduce_sum(tf.square(noise_feature - content_feature))
        factor = 2. * M * N
        content_loss /= factor
        content_losses.append(content_loss * weight)
    return tf.reduce_sum(content_losses)

# compute the Gram maxtrix of feature
def get_gram_matrix(feature):
    # swap dimension, put `channel` dim on the 1st postion
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape to 2d
    x = tf.reshape(x, (x.shape[0], -1))
    return x @ tf.transpose(x)

# compute the style loss of current image
def compute_style_loss(noise_style_features, style_image_features):
    style_losses = []
    M, N = settings.WIDTH * settings.HEIGHT, 3
    for (noise_feature, weight), (style_feature, _) in zip(noise_style_features, style_image_features):
        noise_gram = get_gram_matrix(noise_feature)
        style_gram = get_gram_matrix(style_feature)
        style_loss = tf.reduce_sum(tf.square(noise_gram - style_gram))
        style_loss /= 4. * (M**2) * (N**2)
        style_losses.append(style_loss * weight)
    return tf.reduce_sum(style_losses)

def get_total_loss(noise_image_features, content_image_features, style_image_features):
    content_loss = compute_content_loss(noise_image_features["content"], content_image_features)
    style_loss = compute_style_loss(noise_image_features["style"], style_image_features)
    return settings.CONTENT_LOSS_WEIGHT * content_loss + settings.STYLE_LOSS_WEIGHT * style_loss

# train
def train(content_image_path, style_image_path):
    """
    Preprocessing
    """
     # create dir to save generated images
    if not os.path.exists(settings.OUTPUT_DIR):
        os.mkdir(settings.OUTPUT_DIR)

    style_transfer_model = model.StyleTransferNNModel()
    content_image = utils.load_image(content_image_path)
    style_image = utils.load_image(style_image_path)

    content_image_features = style_transfer_model([content_image, ])["content"]
    style_image_features = style_transfer_model([style_image, ])["style"]

    optimizer = tf.keras.optimizers.Adam(settings.LR)
    noise_image = utils.generage_noise_image(content_image)

    # use tf.function to accelerate training
    @tf.function
    def train_one_iteration():
        with tf.GradientTape() as tape:
            noise_outputs = style_transfer_model(noise_image)
            total_loss = get_total_loss(noise_outputs, content_image_features, style_image_features)
        # compute gradient
        grad = tape.gradient(total_loss, noise_image)
        # optimize
        optimizer.apply_gradients([(grad, noise_image)])
        return total_loss

    for epoch in range(settings.EPOCH):
        with tqdm(total=settings.ITERATION_PER_EPOCH, desc=f"Epoch {epoch+1}/{settings.EPOCH}") as pbar:
            for iteration in range(settings.ITERATION_PER_EPOCH):
                _loss = train_one_iteration()
                pbar.set_postfix({"loss": "%.4f" % float(_loss)})
                pbar.update(1)
            # save images per epoch
            utils.save_image(noise_image, f"{settings.OUTPUT_DIR}/{epoch+1}.jpg")
