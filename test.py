import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from skimage.color import lab2rgb

np.set_printoptions(suppress=True)

import sys

# model = tf.keras.models.load_model('model')
# new_model = tf.keras.models.load_model('new_model')
model = tf.keras.models.load_model('saved_model')

def add_picture(url_path):
	img = np.array(PIL.Image.open(url_path)) / 255.0 
	img = np.array(tf.image.resize(img, (224, 224)))
	img = img.reshape((1,) + img.shape) 
	return img

img1, img2, img3 = add_picture('test/flower.jpeg'), add_picture('test/rose.jpeg'), add_picture('test/flower2.jpg')


plt.figure(figsize=(5, 6))
def plot(matrix, loc):
	ab = model.predict(matrix) * 128

	if loc == 0:
		print(matrix)
		print(ab)
		print(np.max(ab), np.mean(ab), np.min(ab))
		print(np.max(matrix), np.mean(matrix), np.min(matrix))
		
		# diff = np.abs(matrix[:, :, 0] - matrix[:, :, 1])
		# print(np.max(diff), np.min(diff), np.mean(diff))

		# plt.hist(ab[:, :, 1].flatten())
		# plt.show()

	image = np.zeros(ab.shape[1:3] + (3,))

	matrix = matrix[0] 
	image[:, :, :1] = matrix[:, :, :1] * 100
	image[:, :, 1:] = ab
	print(image)

	plt.subplot(3, 2, 2 * loc + 1)
	plt.imshow(lab2rgb(image))
	plt.subplot(3, 2, 2 * loc + 2)
	plt.imshow(matrix, cmap='gray')

print(img1)
plot(img1, 0)
plot(img2, 1)
plot(img3, 2)

plt.show()