import os
import warnings
import pickle
import numpy as np
from scipy.io import loadmat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Sequential, save_model
from keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

warnings.filterwarnings("ignore")

path = os.path.dirname(os.path.realpath(__file__))

mat = loadmat(path+'/data/emnist-balanced.mat')

mapper = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
pickle.dump(mapper, open('mapper.p', 'wb' ))

num_classes = len(mapper)

data_len = len(mat['dataset'][0][0][0][0][0][0])
X_train = mat['dataset'][0][0][0][0][0][0].reshape(data_len, 28, 28, 1)
y_train = mat['dataset'][0][0][0][0][0][1]

data_len = len(mat['dataset'][0][0][1][0][0][0])
X_test = mat['dataset'][0][0][1][0][0][0].reshape(data_len, 28, 28, 1)
y_test = mat['dataset'][0][0][1][0][0][1]

for i in range(len(X_train)):
    X_train[i] = np.rot90(np.fliplr(X_train[i]))

for i in range(len(X_test)):
    X_test[i] = np.rot90(np.fliplr(X_test[i]))

X_train = X_train .astype('float32')
X_test = X_test.astype('float32')

X_train  /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                        height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')

batches = gen.flow(X_train, y_train, batch_size=128)
test_batches = gen.flow(X_test, y_test, batch_size=128)
steps_per_epoch = int(np.ceil(batches.n/128))
validation_steps = int(np.ceil(test_batches.n/128))

model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape =(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(batches, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1,
                            validation_data=test_batches, validation_steps=validation_steps)

score = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy on test set:', score[1])

save_model(model, path+'/character_recognition_model.h5')