# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:02:57 2024

@author: nicho
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.utils import plot_model
import os
from datetime import datetime

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
    r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\conditional_gen.h5",
    custom_objects=custom_objects
)

# Load the model from SavedModel format
generator_h5_cycle = tf.keras.models.load_model(
    r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\cycle_gen.h5",
    custom_objects=custom_objects
)

# Load the model from SavedModel format
generator_h5_cycle_haze = tf.keras.models.load_model(
    r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\generator_model_cycle_day_final_fogging.h5",
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
test_image_folder = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\YoungYoung619 pedestrian-detection-in-hazy-weather master dataset-hazy_person\PICTURES_LABELS_TEMP_TEST\PICTURES"
test_image_path2 = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\synthetic_foggy_val_day\foggy_b2e54795-0dc285dd.jpg"
test_image_path3 = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_val_day\b2e54795-0dc285dd.jpg"
test_image_folder2 = r'C:\Users\nicho\OneDrive\Desktop\Dissertation\YoungYoung619 pedestrian-detection-in-hazy-weather master dataset-hazy_person\PICTURES_LABELS_TRAIN\PICTURES'
test_clear = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_val_day\b2e431b6-332c438d.jpg"

def generate_images(cond_model, cycle_model, test_input):
    cond_prediction = cond_model(test_input, training=False)
    cycle_prediction = cycle_model(test_input, training=False)
    
    # Ensure tensors are float32 before plotting
    test_input = tf.cast(test_input, tf.float32)
    cond_prediction = tf.cast(cond_prediction, tf.float32)
    cycle_prediction = tf.cast(cycle_prediction, tf.float32)

    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], cond_prediction[0], cycle_prediction[0]]
    title = ['Input Image', 'Conditional GAN', 'Cycle GAN']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.33 + 0.33)
        plt.axis('off')
    plt.show()

def generate_images_synth(cond_model, clear_input, fog_input):
    cond_prediction = cond_model(fog_input, training=False)
    
    # Ensure tensors are float32 before plotting
    cond_prediction = tf.cast(cond_prediction, tf.float32)

    plt.figure(figsize=(12, 12))
    display_list = [clear_input[0], fog_input[0], cond_prediction[0]]
    title = ['Input Image', 'Synthetic Fog', 'De-Fogged']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.33 + 0.33)
        plt.axis('off')
    plt.show()
    
def generate_images_cycle(haze_model, cycle_model, clear_input):
    haze_prediction = haze_model(clear_input, training=False)
    haze_prediction = tf.cast(haze_prediction, tf.float32)
    # Ensure tensors are float32 before plotting
    cycled_dehaze_prediction = cycle_model(haze_prediction, training=False)
    cycled_dehaze_prediction = tf.cast(cycled_dehaze_prediction, tf.float32)
    plt.figure(figsize=(12, 12))
    display_list = [clear_input[0], haze_prediction[0], cycled_dehaze_prediction[0]]
    title = ['Input Image', 'Hazed', 'Cycled']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.33 + 0.33)
        plt.axis('off')
    plt.show()



plt.close()

"""
for test_image_path in Path(test_image_folder2).glob('*.jpg'):
    # Preprocess the test image
    test_image = preprocess_image(str(test_image_path))
    # Generate and display images using the model loaded from H5 format
    generate_images(generator_h5, generator_h5_cycle, test_image)


clear_image = preprocess_image(str(test_image_path3))
fog_image = preprocess_image(str(test_image_path2))
# Generate and display images using the model loaded from H5 format
generate_images_synth(generator_h5, clear_image, fog_image)
"""

input_clear = preprocess_image(str(test_clear))
generate_images_cycle(generator_h5_cycle_haze,generator_h5_cycle, input_clear)