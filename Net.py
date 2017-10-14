from keras.models import load_model
from keras.callbacks import ModelCheckpoint, History, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras import metrics, losses
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

class Net:
    def __init__(self):
        pass
    

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
