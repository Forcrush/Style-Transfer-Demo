'''
Author: Puffrora
Date: 2021-11-06 23:01:12
LastModifiedBy: Puffrora
LastEditTime: 2021-11-07 00:20:35
'''

import argparse
import settings, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer NN Model")
    parser.add_argument("-c", "--content_image_path", default=settings.CONTENT_IMAGE_PATH, type=str, help="path of content image")
    parser.add_argument("-s", "--style_image_path", default=settings.STYLE_IMAGE_PATH, type=str, help="path of style image")
    args = parser.parse_args()
    
    train.train(args.content_image_path, args.style_image_path)