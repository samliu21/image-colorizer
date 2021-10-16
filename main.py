import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import os 

from model import model

# Size of training images
INPUT_SIZE = 100

DATASET_PATH = './dataset/images'

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

DATA_SIZE = len(os.listdir(DATASET_PATH))

# Generators for coloured and grayscale pictures respectively
train_gen = data_gen.flow_from_directory(
	'./dataset/', 
	(INPUT_SIZE, INPUT_SIZE), 
	class_mode=None,
	batch_size=DATA_SIZE,
	shuffle=False,
)

train_gen_grayscale = data_gen.flow_from_directory(
	'./dataset/', 
	(INPUT_SIZE, INPUT_SIZE), 
	color_mode = 'grayscale',
	class_mode=None,
	batch_size=DATA_SIZE,
	shuffle=False,
)

train = train_gen[0]
train_grayscale = train_gen_grayscale[0]

# plt.figure(figsize=(5, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(train[0])
# plt.subplot(1, 2, 2)
# plt.imshow(train_grayscale[0], cmap='gray')
# plt.show()

print(np.max(train[1]))
print(np.min(train[1]))
print(np.mean(train[1]))
print(train[1])

# sys.exit()

print(model.summary())

model.fit(train_grayscale, train, epochs=5)

model.save('saved_model')
