"""
Reference:
    https://arxiv.org/abs/1709.01507
"""

from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Add, Multiply
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from Net import Net
import numpy as np
import random

class SENet(Net):
    def __init__(self, model_data):
        # model_data: the parameters of all layers
        self.input_shape = model_data['input_shape']
        self.output_class = model_data['output_class']
        self.reduction_ratio = model_data['reduction_ratio']
        self.blocks_data = model_data['blocks']
        self.repeat_blocks = model_data['repeat']
        self.build_model()
        
    
    def _res_unit(self, layer, I):
        I = BatchNormalization()(I)
        I = Activation('relu')(I)
        I = Conv2D(
                filters=layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(I)
        return I

    
    def _SE_unit(self, I):
        shape = list(map(int, I.shape[1: 3]))
        units_fc0, units_fc1 = int(I.shape[3]) // self.reduction_ratio, int(I.shape[3])
        I = MaxPooling2D(pool_size=shape)(I)
        I = Dense(units_fc0, use_bias=False)(I)
        I = Activation("relu")(I)
        I = Dense(units_fc1, use_bias=False)(I)
        I = Activation("sigmoid")(I)
        return I

    
    def build_block(self, block, I):
        BI = I
        for layer in block: I = self._res_unit(layer, I)
        I = Multiply()([I, self._SE_unit(I)])
        input_shape = K.int_shape(BI)
        output_shape = K.int_shape(I)
        if input_shape != output_shape: 
            sh = int(round(float(input_shape[1]) / output_shape[1]))
            sw = int(round(float(input_shape[2]) / output_shape[2]))
            BI = Conv2D(
                    filters=output_shape[3],
                    kernel_size=1,
                    strides=(sh, sw),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.0001))(BI)
        return Add()([BI, I])


    def build_blocks(self, I):
        for idx, block in enumerate(self.blocks_data):
            for r_idx in range(self.repeat_blocks[idx]):
                I = self.build_block(block, I)
        return I 


    def build_model(self):
        """
            You should add any layer which is not in SENet block here.
            I copy the setting in original paper.
        """
        FI = I = Input(self.input_shape)
        layer = [64, 7, 2]
        I = self._res_unit(layer, I)
        
        I = self.build_blocks(I)

        I = GlobalAveragePooling2D()(I)
        I = Dense(self.output_class)(I)
        I = Activation('softmax')(I)
        self.model = Model(inputs=FI, outputs=I)
