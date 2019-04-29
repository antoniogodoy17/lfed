from keras.callbacks import  ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
import pandas as pd
import cv2
import numpy as np


dR =  ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

class Trainer:

    # Init class
    # def __init__(self, num_epochs=90):
    def __init__(self, imageSize=(48,48), datasetPath='fer2013.csv', batchSize=32, dataGenerator=dR, numClass=7):
        self.imageSize = imageSize
        self.datasetPath = datasetPath
        self.batchSize = batchSize
        self.numEpochs=90
        self.inputShape= (48,48,1)
        self.basePath = 'models/'
        self.l2Regularization = 0.01
        self.dataGenerator = dataGenerator
        self.regularization = l2(self.l2Regularization)
        self.numClass = numClass
        self.model = None
        self.callbacks = None



    def loadCSV(self):
        print('Loading Csv...')
        data = pd.read_csv(self.datasetPath)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.imageSize)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

    def preProcessImages(self, x):
        print('Preproccessing Image')
        x = x.astype('float32')
        x = x / 255.0
        x = (x - 0.5) * 2.0
        return x

    def defineCallbacks(self):
        modelsPath = self.basePath + 'modelTrained'
        modelName = modelsPath + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        modelFunc = ModelCheckpoint(modelName, 'val_loss', verbose=1, save_best_only=True)
        self.callbacks = [modelFunc]


    def loadDataSet(self):
        faces, emotions = self.loadCSV()
        faces = self.preProcessImages(faces)
        num_samples, num_classes = emotions.shape
        xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)
        self.model.fit_generator(self.dataGenerator.flow(xtrain, ytrain,
                                                self.batchSize),
                            steps_per_epoch=len(xtrain) / self.batchSize,
                            epochs=self.numEpochs, verbose=1, callbacks=self.callbacks,
                            validation_data=(xtest, ytest))

    def createModel(self, regularization):
        img_input = Input(self.inputShape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # layer 1
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 2
        residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 3
        residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 4
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        #layer5
        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(256, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])
        x = Conv2D(self.numClass, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        self.model = Model(img_input, output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

#  USAGE

trainer = Trainer()
trainer.createModel(trainer.regularization)
trainer.loadDataSet()