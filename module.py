from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def bn_relu_conv(tensor, num_filters, kernel_size, strides = (1, 1)):
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)
    tensor = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(tensor)
    
    return tensor

def selu_conv(tensor, num_filters, kernel_size, strides = (1, 1)):
    tensor = Activation('selu')(tensor)
    tensor = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(tensor)
    
    return tensor
    
def conv_bn_relu(tensor, num_filters, kernel_size, strides = (1, 1)):
    tensor = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)
    
    return tensor

def conv_selu(tensor, num_filters, kernel_size, strides = (1, 1)):
    tensor = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(tensor)
    tensor = Activation('selu')(tensor)
    
    return tensor
