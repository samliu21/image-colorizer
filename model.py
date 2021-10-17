import tensorflow as tf

"""
Input: (None, 224, 224, 3)
Grayscale image: R, G, and B are all the same value

______________________________________________________
Layer                Output Shape              
______________________________________________________
VGG16                (None, 7, 7, 512)
Conv2D               (None, 7, 7, 256)
Conv2DTranpose       (None, 14, 14, 16)
Conv2D               (None, 14, 14, 256)
Dropout              (None, 14, 14, 256)
UpSampling2D		 (None, 28, 28, 256)
Conv2D               (None, 28, 28, 128)
Dropout              (None, 28, 28, 128)
UpSampling2D		 (None, 56, 56, 128)
Conv2D               (None, 56, 56, 64)
UpSampling2D		 (None, 112, 112, 64)
Conv2D               (None, 112, 112, 32)
Conv2D               (None, 112, 112, 2)
UpSampling2D		 (None, 224, 224, 2)
______________________________________________________

Output: (None, 224, 224, 2)
AB values in the LAB colour scheme
"""
inception = tf.keras.applications.vgg16.VGG16()

model = tf.keras.Sequential()

for layer in inception.layers:
	if type(layer) == tf.keras.layers.Flatten:
		break
	model.add(layer)
	layer.trainable = False

model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(16, (8, 8)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(2, (3, 3), padding='same', activation='tanh'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse')
