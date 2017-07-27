from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K
from module import *

class ResNet:
    def __init__(self, input_shape, output_shape, num_layers, block_function=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        assert self.num_layers in [18, 34, 50, 101, 152], "ModelError"
        if not block_function: self.block_function = bn_relu_conv
        else: self.block_function = block_function
        self.build_model()
    
    def _basic_block(self, tensor, num_filters, strides):
        tensor = self.block_function(tensor, num_filters, (3, 3), strides)
        tensor = self.block_function(tensor, num_filters, (3, 3), (1, 1))
        
        return tensor

    def _bottleneck_block(self, tensor, num_filters, strides):
        tensor = self.block_function(tensor, num_filters, (1, 1), strides)
        tensor = self.block_function(tensor, num_filters, (3, 3), (1, 1))
        tensor = self.block_function(tensor, num_filters * 4, (1, 1), (1, 1))
        
        return tensor

    def _shortcut(self, input_tensor, tensor):
        input_shape = K.int_shape(input_tensor)
        residual_shape = K.int_shape(tensor)
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        channel_equal = input_shape[3] == residual_shape[3]

        if stride_height > 1 or stride_width > 1 or not channel_equal:
            shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_height, stride_width), padding='valid',
                    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)

        else: shortcut = input_tensor

        return add([shortcut, tensor])

    def _residual_block(self, input_tensor, num_filters, num_blocks, idx, block_type):
        if idx == 0:
            tensor = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
        else:
            if block_type == "bottleneck":
                tensor = self._bottleneck_block(input_tensor, num_filters, strides=(2, 2))
            else:
                tensor = self._basic_block(input_tensor, num_filters, strides=(2, 2))

        for i in range(num_blocks-1):
            if block_type == "bottleneck":
                tensor = self._bottleneck_block(tensor, num_filters, strides=(1, 1))
            else:
                tensor = self._basic_block(tensor, num_filters, strides=(1, 1))
            
        return self._shortcut(input_tensor, tensor)

    def build_model(self):
        input_tensor = Input(shape=self.input_shape)
        conv1 = conv_bn_relu(input_tensor, num_filters=64, kernel_size=(7, 7), strides=(2, 2))
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        tensor = pool1
        num_filters = 64

        if self.num_layers <= 34: self.block_type = "basic"
        else: self.block_type = "bottleneck"

        if self.num_layers == 18: self.num_blocks = [2, 2, 2, 2]
        elif self.num_layers == 34: self.num_blocks = [3, 4, 6, 3]
        elif self.num_layers == 50: self.num_blocks = [3, 4, 6, 3]
        elif self.num_layers == 101: self.num_blocks = [3, 4, 23, 3]
        else: self.num_blocks = [3, 8, 36, 3]

        for idx, cnt in enumerate(self.num_blocks):
            tensor = self._residual_block(tensor, 64, cnt, idx, self.block_type)

        tensor_shape = K.int_shape(tensor)
        tensor = AveragePooling2D(pool_size=tensor_shape[1:3], strides=(1, 1))(tensor)
        tensor = Flatten()(tensor)
        tensor = Dense(units=1000, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(tensor)
        tensor = Activation('relu')(tensor)
        tensor = Dense(units=self.output_shape, kernel_initializer="he_normal", activation='softmax')(tensor)

        self.model = Model(inputs=input_tensor, outputs=tensor)
