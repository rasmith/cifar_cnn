import keras
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from cifar import Cifar
from keras import backend as K

num_train_images = 50000
num_test_images = 10000

print(keras.__version__)

# Load CIFAR-10 data.
c = Cifar(num_train_images , num_test_images)
((train_data, train_labels), (test_data, test_labels)) = c.load_data()

# Make Keras happy.

# Reshape and convert to float in range [0, 1] for Keras.
train_data = train_data.reshape(train_data.shape[0], 32, 32, 3)
train_data = train_data.astype('float32')
train_data /= 255

# Check the shape.
print("train_data.shape = %s" % str(train_data.shape))

# Convert from Python list to categorical.
train_labels = np_utils.to_categorical(train_labels)

# Check the shape.
print("train_labels = %s" % str(train_labels.shape))

# OK, do the same for the test data.

# Reshape and convert to float in range [0, 1] for Keras.
test_data = test_data.reshape(test_data.shape[0], 32, 32, 3)
test_data = test_data.astype('float32')
test_data /= 255

# Check the shape.
print("test_data.shape = %s" % str(test_data.shape))

# Convert from Python list to categorical.
test_labels = np_utils.to_categorical(test_labels)

# Check the shape.
print("test_labels = %s" % str(test_labels.shape))

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32,32, 3)))

print(model.output_shape)

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint_file = 'model.hdf5'

if sys.argv[1] == "train":
	model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss')
	model.fit(train_data, train_labels,
	  batch_size=32, epochs=100, callbacks=[model_checkpoint], verbose=1)
else:
	model.load_weights(checkpoint_file)
	score = model.evaluate(test_data, test_labels, verbose=1)
	print(score) 
