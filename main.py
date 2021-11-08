'''
Author: Puffrora
Date: 2021-11-06 23:01:12
LastModifiedBy: Puffrora
LastEditTime: 2021-11-08 11:27:33
'''

import argparse
import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer NN Model")
    parser.add_argument("-c", "--content_choice", default=1, type=int, help="category of content image")
    parser.add_argument("-s", "--style_choice", default=1, type=int, help="category of style image")
    parser.add_argument("-epoch", "--epochs", default=200, type=int, help="traning eopchs")
    parser.add_argument("-iter", "--iteration_per_epoch", default=100, type=int, help="iterations per epoch")
    parser.add_argument("-lr", "--lr", default=0.01, type=float, help="learning rate")
    
    args = parser.parse_args()
    
    train.train(args.content_choice, args.style_choice, args.epochs, args.iteration_per_epoch, args.lr)