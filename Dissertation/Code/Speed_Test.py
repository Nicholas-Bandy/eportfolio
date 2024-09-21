# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:31:37 2024

@author: nicho
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Custom ResizeLayer
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        target_height, target_width = self.target_size[0], self.target_size[1]
        resized = tf.image.resize(inputs, [target_height, target_width])
        return resized

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_size': self.target_size
        })
        return config

# Ensure the custom objects dictionary includes the ResizeLayer
custom_objects = {'ResizeLayer': ResizeLayer}

# Load the model from H5 format
generator_h5 = tf.keras.models.load_model(
    r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\generator_model_cycle_day_50buffer.h5",
    custom_objects=custom_objects
)

# Load the model from SavedModel format
generator_h5_cycle = tf.keras.models.load_model(
    r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\cycle_gen.h5",
    custom_objects=custom_objects
)

# Constants
IMG_HEIGHT = 720
IMG_WIDTH = 1280

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Path to the test image
test_image_folder = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\YoungYoung619 pedestrian-detection-in-hazy-weather master dataset-hazy_person\PICTURES_LABELS_TRAIN\PICTURES"
test_image_path2 = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\synthetic_foggy_val\foggy_b1d7b3ac-5744370e.jpg"


def generate_images(model, test_input):
    
    prediction = model(test_input, training=False)
    return prediction
  
image_count = 0
start_time = time.time()

# Process each image
for test_image_path in Path(test_image_folder).glob('*.jpg'):
    # Preprocess the test image
    test_image = preprocess_image(str(test_image_path))
    
    # Generate and display images using the models loaded from H5 format
    generate_images(generator_h5, test_image)
    
    # Increment the counter
    image_count += 1

# Calculate the total time taken
end_time = time.time()
total_time = end_time - start_time

# Output the total number of images and the time taken
print(f"Total number of images processed: {image_count}")
print(f"Total time taken: {total_time:.2f} seconds")