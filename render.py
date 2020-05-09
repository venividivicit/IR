# make convenutinal neaural network
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle

IMG_SIZE = 100;

X = pickle.load(open("/Users/veni_vedi_veci/Projects/image-recognition/X.pickle","rb"))
y = pickle.load(open("/Users/veni_vedi_veci/Projects/image-recognition/y.pickle","rb"))

y = np.array(y)

#normalize the data, scale it
X = X/255

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# dense needs 1D dataset
model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3)