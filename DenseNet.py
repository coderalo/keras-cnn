"""
Reference:
    https://arxiv.org/abs/1608.06993
"""

from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from Net import Net
import numpy as np
import random

class DenseNet(Net):
    def __init__(self, model_data):
        # model_data: the parameters of all layers
        self.input_shape = model_data['input_shape']
        self.output_class = model_data['output_class']
        self.growth_rate = model_data['growth_rate']
        self.blocks_data = model_data['blocks']
        self.repeat_blocks = model_data['repeat']
        self.compression_factor = model_data['compression_factor']
        self.bottleneck = model_data['bottleneck']
        self.build_model()
        
    
    def _res_unit(self, filters, kernel_size, strides, I):
        I = BatchNormalization()(I)
        I = Activation('relu')(I)
        I = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(I)
        return I


    def build_dense_block(self, block, repeat, I):
        LI = I
        for _ in range(repeat):
            if self.bottleneck:
                I = self._res_unit(4*self.growth_rate, block[0][0], block[0][1], LI)
                I = self._res_unit(self.growth_rate, block[1][0], block[1][1], I)
                LI = Concatenate()([LI, I])
            else:
                I = self._res_unit(self.growth_rate, block[0][0], block[0][1], LI)
                LI = Concatenate()([LI, I])
        return LI


    def build_transition_layer(self, block, filters, _type, I):
        I = self._res_unit(filters, block[0][0], block[0][1], I)
        if _type == "max":
            I = MaxPooling2D(
                    pool_size=block[1][0],
                    strides=block[1][1],
                    padding='same')(I)
        elif _type == "avg":
            I = AveragePooling2D(
                    pool_size=block[1][0],
                    strides=block[1][1],
                    padding='same')(I)
        return I


    def build_blocks(self, I):
        for idx, block in enumerate(self.blocks_data):
            if idx % 2 == 0:
                if idx == 0:
                    if self.compression_factor == 1: filters = 16
                    else: filters = 2 * self.growth_rate
                else:
                    channels = int(I.shape[3])
                    filters = int(channels * self.compression_factor)
                I = self.build_transition_layer(block, filters, ("max" if idx == 0 else "avg"), I)
            else:
                I = self.build_dense_block(block, self.repeat_blocks[idx // 2], I)
        return I


    def build_model(self):
        """
            You should add any layer which is not in DenseNet block here.
            I copy the setting in original paper.
        """
        FI = I = Input(self.input_shape)
        I = self.build_blocks(I)
        
        I = GlobalAveragePooling2D()(I)
        I = Dense(self.output_class)(I)
        I = Activation('softmax')(I)
        self.model = Model(inputs=FI, outputs=I)
