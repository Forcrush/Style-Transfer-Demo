'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-08 12:33:28
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
def train(content_choice, style_choice, epochs, iteration_per_epoch, lr):
    """
    Preprocessing
    """
    content_image_path = f"images/content/content{content_choice}.jpg"
    style_image_path = f"images/style/style{style_choice}.jpg"

    out_put_dir = f"output/content{content_choice}-style{style_choice}"
     # create dir to save generated images
    if not os.path.exists(out_put_dir):
        os.mkdir(out_put_dir)

    style_transfer_model = model.StyleTransferNNModel()
    content_image = utils.load_image(content_image_path)
    style_image = utils.load_image(style_image_path)

    content_image_features = style_transfer_model([content_image, ])["content"]
    style_image_features = style_transfer_model([style_image, ])["style"]

    optimizer = tf.keras.optimizers.Adam(lr)
    noise_image = utils.generage_noise_image(content_image)

    """
    Training
    """
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

    training_losses = []
    for epoch in range(epochs):
        with tqdm(total=iteration_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in range(iteration_per_epoch):
                _loss = train_one_iteration()
                pbar.set_postfix({"loss": "%.4f" % float(_loss)})
                pbar.update(1)
            # save loss of last iteration per epoch
            training_losses.append(_loss)
            # save images per 20 epoch
            if (epoch+1) % 20 == 0:
                utils.save_image(noise_image, f"{out_put_dir}/{epoch+1}.jpg")

    # loss visualization
    utils.visualize_loss(out_put_dir, training_losses)