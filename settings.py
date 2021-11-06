'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-07 00:20:28
'''

# Path of content image
CONTENT_IMAGE_PATH = './images/content.jpg'
# Path of style image
STYLE_IMAGE_PATH = './images/style.jpg'
# Path of generated image
OUTPUT_DIR = './output'

# feature layer and weight coefficient of content image
CONTENT_LAYERS = {'block4_conv2': 0.4, 'block5_conv2': 0.6}
# feature layer and weight coefficient of style image
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 
                'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}

# Content loss weight
CONTENT_LOSS_WEIGHT = 30
# Style loss weight
STYLE_LOSS_WEIGHT = 100

# Width and height of images
WIDTH, HEIGHT = 500, 500

# how many epochs
EPOCH = 20
# how many turns in each epoch
ITERATION_PER_EPOCH = 100
# Learning Rate
LR = 0.01

# Since the pre-trained network we used is trained on ImageNet,
# so we need to use the mean and std of ImageNet to normalize
ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.299, 0.224, 0.225]
