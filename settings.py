'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-08 11:27:58
'''

# feature layer and weight coefficient of content image
CONTENT_LAYERS = {'block4_conv2': 0.4, 'block5_conv2': 0.6}
# feature layer and weight coefficient of style image
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 
                'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}

# Content loss weight
CONTENT_LOSS_WEIGHT = 1
# Style loss weight
STYLE_LOSS_WEIGHT = 100

# Width and height of images
WIDTH, HEIGHT = 500, 500

# Since the pre-trained network we used is trained on ImageNet,
# so we need to use the mean and std of ImageNet to normalize
ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.299, 0.224, 0.225]
