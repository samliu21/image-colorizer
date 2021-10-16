import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape=(None, None, 1)),
	tf.keras.layers.UpSampling3D(size=(1, 1, 3)),

	tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
	tf.keras.layers.Dropout(0.3),
	
	tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
	tf.keras.layers.Dropout(0.3),

	tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
	tf.keras.layers.Dropout(0.3),

	tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

	tf.keras.layers.Conv2D(3, 3, padding='same', activation='relu'),
])

model.compile(optimizer='adam', loss='mse')
