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

class DenseNet:
    def __init__(self, num_layers, input_shape, output_shape, block_function, dropout_rate):
        self.num_layers = num_layers
        assert self.num_layers in [121, 169, 201, 161], "ModelError"
        if self.num_layers in [121, 169, 201]: self.growth_rate = 32
        else: self.growth_rate = 48
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.block_function = block_function
        self.dropout_rate = dropout_rate
        self.filters = 16

        self.build_model()

    def build_model(self):
        input_tensor = Input(shape=self.input_shape)
        conv1 = conv_bn_relu(input_tensor, num_filters=self.filters, kernel_size=(7, 7), strides=(2, 2))
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        tensor = pool1
        
        for idx in range(3):
            tensor = self.dense_block(tensor, self.growth_rate, self.dropout_rate)
