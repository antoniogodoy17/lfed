from keras import layers
from keras.callbacks import  ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, MaxPooling2D, SeparableConv2D
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, imageSize, imagesPath, batchSize):
        self.imageSize = (imageSize, imageSize)
        self.imagesPath = imagesPath
        self.batchSize = batchSize
        self.datagen_train = ImageDataGenerator()
        self.datagen_validation = ImageDataGenerator()
        self.basePath = './models/'
        self.regularization = l2(0.01)
        self.train_generator = None
        self.validation_generator = None
        self.classes = None
        self.model = None
        self.callbacks = None

    def setUpDataGenerators(self):
        self.train_generator = self.datagen_train.flow_from_directory(self.imagesPath + "train",
                                                                      target_size=self.imageSize,
                                                                      color_mode="grayscale",
                                                                      batch_size=self.batchSize,
                                                                      class_mode='categorical',
                                                                      shuffle=True)

        self.validation_generator = self.datagen_validation.flow_from_directory(self.imagesPath + "validation",
                                                                                target_size=self.imageSize,
                                                                                color_mode="grayscale",
                                                                                batch_size=self.batchSize,
                                                                                class_mode='categorical',
                                                                                shuffle=False)

    def createModel(self, numOfClasses):
        self.classes = numOfClasses
        size = (48,48,1)
        img_input = Input(size)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=self.regularization, use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # layer 1
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 2
        residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 3
        residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # layer 4
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        #layer 5
        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(256, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', kernel_regularizer=self.regularization, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])
        x = Conv2D(self.classes, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        self.model = Model(img_input, output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def defineCallbacks(self, checkpoints):
        defaultModel = f'{self.basePath}trainedModel'
        modelName = defaultModel + '.{val_acc:.2f}.hdf5'
        saveDefaultModel = ModelCheckpoint(defaultModel, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        saveCheckpoints = ModelCheckpoint(modelName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.callbacks = [saveDefaultModel, saveCheckpoints] if checkpoints else [saveDefaultModel]

    def train(self, epochs, showResults):
        self.history = self.model.fit_generator(generator=self.train_generator,
                                                steps_per_epoch=self.train_generator.n//self.train_generator.batch_size,
                                                epochs=epochs,
                                                verbose=1,
                                                validation_data=self.validation_generator,
                                                validation_steps=self.validation_generator.n//self.validation_generator.batch_size,
                                                callbacks=self.callbacks)  

        if showResults:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.suptitle('Optimizer : Adam', fontsize=10)
            plt.ylabel('Loss', fontsize=16)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.legend(loc='upper right')

            plt.subplot(1, 2, 2)
            plt.ylabel('Accuracy', fontsize=16)
            plt.plot(self.history.history['acc'], label='Training Accuracy')
            plt.plot(self.history.history['val_acc'], label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.savefig('./models/results.png')                           
            plt.show()

    def execute(self):
        self.setUpDataGenerators()
        self.createModel(numOfClasses=6)
        self.defineCallbacks(checkpoints=True)
        self.train(epochs=50, showResults=True)

if __name__ == '__main__':
    trainer = Trainer(imageSize=48, imagesPath='./img/', batchSize=5)
    trainer.execute()