import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL

import sys

model = tf.keras.models.load_model('saved_model')

def add_picture(url_path):
	img_rgb = np.array(PIL.Image.open(url_path)) / 255.0
	img = np.array(tf.image.rgb_to_grayscale(img_rgb))
	img = img.reshape((1,) + img.shape)
	print(img.shape)
	return img

img1, img2 = add_picture('test/0.jpg'), add_picture('test/test_picture2.jpg')

plt.figure(figsize=(5, 5))
def plot(picture, loc, cmap=None):
	plt.subplot(2, 2, loc)
	if cmap:
		plt.imshow(picture, cmap)
	else:
		plt.imshow(picture)

res1 = model.predict(img1)[0]
res2 = model.predict(img2)[0]

print(res1)
print(np.mean(res1))
print(np.max(res1))
print(np.min(res1))

plot(res1, 1)
plot(img1[0], 2, 'gray')
plot(res2, 3)
plot(img2[0], 4, 'gray')

print()

plt.show()