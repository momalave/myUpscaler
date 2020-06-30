import numpy as np
import tensorflow as tf
import imageio

# Load image
x = tf.io.read_file("../imgs/input.png")
x = tf.image.decode_png(x, channels=3)

# Preprocess image
x = tf.cast(x, tf.float32)
x = tf.expand_dims(x, axis=0)

# Load and apply model
model = tf.saved_model.load("../upscaler_model")
y = model(x)

# Save model
# tf.saved_model.save(model, 'model')

# Process output
y = tf.clip_by_value(y, 0, 255)
y = tf.round(y)
y = tf.cast(y, tf.uint8)

# Write output to file
imageio.imwrite('../imgs/outfile.png', np.array(y[0]) )
