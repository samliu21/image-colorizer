import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from skimage.color import lab2rgb

model = tf.keras.models.load_model('saved_model')

def add_picture(url_path):
	i = PIL.Image.open(url_path)
	i = i.convert('RGB')
	img = np.array(i) / 255.0 
	img = np.array(tf.image.resize(img, (224, 224)))
	img = img.reshape((1,) + img.shape) 
	return img

img1, img2, img3 = add_picture('test/landscape6.jpg'), add_picture('test/landscape2.jpeg'), add_picture('test/landscape5.jpg')

plt.figure(figsize=(5, 6))
def plot(matrix, loc):
	ab = model.predict(matrix) * 128

	image = np.zeros(ab.shape[1:3] + (3,))

	matrix = matrix[0] 
	image[:, :, :1] = matrix[:, :, :1] * 100
	image[:, :, 1:] = ab

	plt.subplot(3, 2, 2 * loc + 1)
	plt.imshow(lab2rgb(image))
	plt.subplot(3, 2, 2 * loc + 2)
	plt.imshow(matrix, cmap='gray')

plot(img1, 0)
plot(img2, 1)
plot(img3, 2)

plt.show()