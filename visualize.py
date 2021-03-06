import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from skimage.color import lab2rgb

model = tf.keras.models.load_model('saved_model')

def add_picture(url_path):
	"""
	Assumes the given image is black and white
	Returns a 224 x 224 x 3 numpy array
	R, G, and B are the same in black and white pictures
	"""
	img = np.array(PIL.Image.open(url_path).convert('RGB')) / 255.0
	img = np.array(tf.image.resize(img, (224, 224)))
	img = img.reshape((1,) + img.shape) 
	return img

URLS = [
	'test/landscape11.jpg',
	'test/landscape12.jpg',
	'test/landscape10.webp',
]

sizes = list(map(lambda x: np.array(PIL.Image.open(x)).shape[:2], URLS))
print(sizes)
imgs = list(map(lambda x: add_picture(x), URLS))

plt.figure(figsize=(5, 2 * len(imgs)))

def plot(matrix, loc):
	ab = model.predict(matrix) * 128

	# 224 x 224 x 3 matrix of zeroes
	image = np.zeros(ab.shape[1:3] + (3,))

	matrix = matrix[0] 
	# Set first column to grayscale values, which correspond to L in LAB
	# Second and third columns are AB values
	image[:, :, :1] = matrix[:, :, :1] * 100
	image[:, :, 1:] = ab
	image = lab2rgb(image)

	# RGB
	plt.subplot(len(imgs), 2, 2 * loc + 1)
	plt.axis('off')
	image = tf.image.resize(image, sizes[loc])
	plt.imshow(image)

	# Grayscale
	plt.subplot(len(imgs), 2, 2 * loc + 2)
	plt.axis('off')
	matrix = tf.image.resize(matrix, sizes[loc])
	plt.imshow(matrix, cmap='gray')

for idx, img in enumerate(imgs):
	plot(img, idx)

plt.show()