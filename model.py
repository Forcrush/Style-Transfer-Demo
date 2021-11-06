'''
Author: Puffrora
Date: 2021-11-06 14:03:13
LastModifiedBy: Puffrora
LastEditTime: 2021-11-06 20:07:55
'''

import typing
import utils
import tensorflow as tf
import settings


class StyleTransferNNModel(tf.keras.Model):
    def __init__(self, content_layers: typing.Dict[str, float]=settings.CONTENT_LAYERS,
                style_layers: typing.Dict[str, float]=settings.STYLE_LAYERS):
        super(StyleTransferNNModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers

        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        self.vgg = utils.get_vgg19(layers)
    
    def call(self, inputs, training=None, mask=None):
        """
        front-forward propogation
        """
        outputs = self.vgg(inputs)
        content_outputs = []
        for layer, weight in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], weight))
        style_outputs = []
        for layer, weight in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], weight))
        
        return {"content": content_outputs, "style": style_outputs}