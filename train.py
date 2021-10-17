import tensorflow as tf
import numpy as np
from skimage.color import rgb2lab

import os 
import sys
import model

np.set_printoptions(suppress=True)

INPUT_SIZE = 224
PATH_PREFIX = './landscapes2/'
DATASET_PATH = PATH_PREFIX + 'images/'
SAVED_MODEL_PATH = 'saved_model'
DATA_SIZE = len(os.listdir(DATASET_PATH))
GET_EXISTING_MODEL = True

def get_existing_model():
	return tf.keras.models.load_model(SAVED_MODEL_PATH)

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = data_gen.flow_from_directory(
	PATH_PREFIX,
	(INPUT_SIZE, INPUT_SIZE), 
	class_mode=None,
	batch_size=DATA_SIZE,
	shuffle=True,
)

"""
train_gen is a DirectoryIterator of RGB images
To get X, we cast RGB images to grayscale, then back to RGB to reobtain three dimensions
To get Y, we cast RGB images to LAB, then isolate the AB components
It's worth noting that the L component of LAB is equivalent to grayscale
"""
X = tf.image.grayscale_to_rgb((tf.image.rgb_to_grayscale(train_gen[0]))) 
Y = rgb2lab(train_gen[0])[:, :, :, 1:] / 128

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

print('Fetching existing model...' if GET_EXISTING_MODEL else 'Creating model...')
model = get_existing_model() if GET_EXISTING_MODEL else model.model

model.summary()
history = model.fit(X, Y, epochs=10, validation_split=0.2, callbacks=[early_stopping])

model.save(SAVED_MODEL_PATH)
