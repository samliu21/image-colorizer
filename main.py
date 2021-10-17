import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

import os 
import sys

from model import model

np.set_printoptions(suppress=True)

INPUT_SIZE = 224

DATASET_PATH = './flowers/images/'

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

DATA_SIZE = len(os.listdir(DATASET_PATH))

train_gen = data_gen.flow_from_directory(
	'./flowers/',
	(INPUT_SIZE, INPUT_SIZE), 
	class_mode=None,
	batch_size=13233,
	shuffle=True,
)

X = tf.image.grayscale_to_rgb((tf.image.rgb_to_grayscale(train_gen[0]))) 
Y = rgb2lab(train_gen[0])[:, :, :, 1:] / 128

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

model.summary()
history = model.fit(X, Y, epochs=5, validation_split=0.2, callbacks=[early_stopping])

model.save('saved_model')

