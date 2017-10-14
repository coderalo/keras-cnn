from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras import metrics, losses
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

class ResNet:
    def __init__(self, model_data):
        """
            model_data: the parameters of all layers
            lr_scheduler: custom learning rate scheduler
        """   
        self.input_shape = model_data['input_shape']
        self.output_class = model_data['output_class']
        self.blocks_data = model_data['blocks']
        self.repeat_blocks = model_data['repeat']
        self.block_type = model_data['block_type'] 
        self.build_model()

    
    def _original_unit(self, layer, I, is_act=True):
        I = Conv2D(
                filters=layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(I)
        I = BatchNormalization()(I)
        I = Activation('relu')(I) if is_act else I
        return I


    def _new_unit(self, layer, I):
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


    def build_block(self, block, I):
        BI = I
        for idx, layer in enumerate(block):
            if self.block_type == "original":
                I = self._original_unit(layer, I, is_act=(idx != len(block-1)))
            elif self.block_type == "new":
                I = self._new_unit(layer, I)
        
        if self.block_type == "original":
            return Activation('relu')(Add()([BI, I]))
        else:
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
            You should add any layer which is not in resnet block here.
            I copy the setting in original paper.
        """
        FI = I = Input(self.input_shape)
        layer = [64, 7, 2]
        if self.block_type == "original":
            I = self._original_unit(layer, I, is_act=True)
        elif self.block_type == "new":
            I = self._new_unit(layer, I)
        
        I = self.build_blocks(I)

        I = GlobalAveragePooling2D()(I)
        I = Dense(self.output_class)(I)
        I = Activation('softmax')(I)
        self.model = Model(inputs=FI, outputs=I)


    def train(self, images, labels, nb_epoch, batch_size=32, lr=0.1, decay=0.9, momentum=0.9, val_split=0.2, log_path="train.log", model_path="model.hdf5"):
        """
            You should carefully read the code here, and modify it to fit your need.
            e.g. the settings of ImageDataGenerator, optimizer, callbacks, e.t.c.
        """
        data = list(zip(images, labels))
        data_size = len(data)
        random.shuffle(data)
        X, y = zip(*data)
        X, y = np.array(X), np.array(y)
        train_X, valid_X, train_y, valid_y = X[:int(data_size*(1-val_split))], X[int(data_size*(1-val_split)):], y[:int(data_size*(1-val_split))], y[int(data_size*(1-val_split)):]
        
        steps_per_epoch = len(train_X) // batch_size

        train_datagen = ImageDataGenerator(
                samplewise_center=True,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
        valid_datagen = ImageDataGenerator(
                samplewise_center=True,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
        
        train_datagen.fit(train_X)
        valid_datagen.fit(valid_X)

        optimizer = SGD(lr=lr, momentum=momentum) 
        logger = History()
        reducer = ReduceLROnPlateau(factor=decay)
        saver = ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor="val_acc")
        self.model.compile(
                optimizer=optimizer,
                loss=losses.categorical_crossentropy,
                metrics=["accuracy"])
        self.model.fit_generator(
                train_datagen.flow(train_X, train_y, batch_size=batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                callbacks=[logger, reducer, saver],
                validation_data=valid_datagen.flow(valid_X, valid_y, batch_size=batch_size),
                validation_steps=steps_per_epoch)
        
        with open(log_path, 'w') as file:
            file.write("tr_acc, tr_loss, val_acc, val_loss\n") 
            for idx in range(nb_epoch):
                file.write("{}, {}, {}, {}\n".format(logger.history['acc'][idx], logger.history['loss'][idx], logger.history['val_acc'][idx], logger.history['val_loss'][idx]))


    def load_model(self, model_path):
        self.model = load_model(model_path)


    def test(self, images, batch_size=32, model_path="model.hdf5"):
        prediction = self.model.predict(images, batch_size=batch_size)
        tokens = prediction.argmax(1)
        return prediction, tokens

