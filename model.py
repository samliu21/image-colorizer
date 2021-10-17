import tensorflow as tf

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
