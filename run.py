#from __future__ import absolute_import
#from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import *
from keras.optimizers import SGD, Adadelta, Adagrad,Adam
from keras.utils import np_utils, generic_utils
from keras.callbacks import *
from six.moves import range
#from data import load_data

import os
from PIL import Image
import numpy as np

import theano


# Read folder mnist under 42,000 pictures, pictures to grayscale, so as a channel,
# If a color map as input, then a replacement is 3, and the data [i,:,:,:] = arr to Data [i,:,:,:] = [arr [:,:, 0], arr [:,:, 1], arr [:,:, 2]]
def  load_data():
    imgs = [x for x in os.listdir("./images") if '.jpg' in x]
    N = len(imgs)
    data = np.empty((N,3,100,100),dtype="float32")
    label = np.empty((N,),dtype ="uint8")
    num = len(imgs)
    j=0
    for i in range(num):
        if not imgs[i].startswith('.'):
            label[j]=0 if '0' in imgs[i].split('_')[1] else 1
            img = Image.open("./images/"+imgs[i])
            arr = np.asarray (img, dtype ="float32")

            data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]]
            j=j+1
    return data, label



# Load data
data,label = load_data()
print(data.shape[0],'samples')
#
##label 0 to 9 of 10 categories, keras requested format is binary class matrices, transforming it, directly call this function keras provided
label = np_utils.to_categorical(label,2)



batch_size = 256
nb_classes = 2
nb_epoch = 120

# input image dimensions
img_rows, img_cols = 100,100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3



model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).

checkpoint = ModelCheckpoint('model.checkpoint')

model.compile(loss='categorical_crossentropy', optimizer='adam')
data=data.astype("float32")
data/=255
model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.1,callbacks = [checkpoint])

model.save_weights('meowwow.h5')
