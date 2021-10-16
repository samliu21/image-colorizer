import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import os 
import shutil
import PIL

from model import model

INPUT_SIZE = 100

ROOT_PATH = './images/'
DATASET_PATH = ROOT_PATH + '_small_dataset/images/'

if not os.path.isdir(DATASET_PATH):
	PICTURES_PER_CATEGORY = 5
	os.makedirs(DATASET_PATH, exist_ok=True)

	image_categories = list(map(lambda x: ROOT_PATH + x + '/', os.listdir(ROOT_PATH)))

	for idx, category in enumerate(image_categories):
		for pic in os.listdir(category)[: PICTURES_PER_CATEGORY]:
			source = category + pic
			destination = '{}{}.{}'.format(DATASET_PATH, idx, pic)
			try:
				shutil.copy2(source, destination)
			except:
				pass

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

DATA_SIZE = len(os.listdir(DATASET_PATH))

train_gen = data_gen.flow_from_directory(
	ROOT_PATH + '_small_dataset', 
	# (INPUT_SIZE, INPUT_SIZE), 
	class_mode=None,
	batch_size=DATA_SIZE,
	shuffle=False,
)

train_gen_grayscale = data_gen.flow_from_directory(
	ROOT_PATH + '_small_dataset', 
	# (INPUT_SIZE, INPUT_SIZE), 
	color_mode = 'grayscale',
	class_mode=None,
	batch_size=DATA_SIZE,
	shuffle=False,
)

train = train_gen[0]
train_grayscale = train_gen_grayscale[0]

print(model.summary())

model.fit(train_grayscale, train, epochs=1)

model.save('saved_model')
