# -*- coding: utf-8 -*-
"""
2024/06/07
Author: Nicholas Bandy
"""

import tensorflow as tf
import pathlib
from matplotlib import pyplot as plt
import time
import datetime
import os
from IPython import display

# Set the environment variable for disabling JIT compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Set CUDA environment variables
os.environ['CUDA_HOME'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8'  
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8'  
os.environ['LD_LIBRARY_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64'  
os.environ['PATH'] += r';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
os.environ['PATH'] += r';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp'

# Verify TensorFlow is using the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only using 1 gpu, but if multiple are used memory growth needs to be consistent
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


AUTOTUNE = tf.data.AUTOTUNE # Constant provided by tensorflow used for optimization, removes need for manually tune some parameters

BUFFER_SIZE = 1000 #The number of images loaded into memory to be shuffled at a given time. Ensures model doesnt learn order of data
BATCH_SIZE = 1 #Number of images processed at a time before updating weights

# Size of all images in the dataset
IMG_WIDTH = 1280
IMG_HEIGHT = 720

LAMBDA = 100 # Used as multiplier in generator loss function. Raising lambda prioritizes mean difference in loss, making the result closer to the original.

# Initialize optimizers. Uses Adam algorithm
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Initialize loss object for BCE. Used in both the generator loss and discriminator loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Details for keeping logs
log_dir = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\logs\\"
summary_writer = tf.summary.create_file_writer(log_dir + "fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size):
        super(ResizeLayer, self).__init__()
        self.target_size = target_size  # Target size to resize the input to

    def build(self, input_shape):
        super().build(input_shape)  # Initializes the layer; no custom weights are used here

    def call(self, inputs):
        # Resizes the input tensor to the specified target size
        resized = tf.image.resize(inputs, self.target_size)
        return resized

    def get_config(self):
        config = super().get_config()
        # Stores the target size in the layer's configuration
        config.update({'target_size': self.target_size})
        return config

def load(clear_image_file, foggy_image_file):
    """This function loads the clear and foggy images from the provided paths and prepares them for training"""
   
    # Read and decode the input image file to a uint8 tensor
    clear_image = tf.io.read_file(clear_image_file)
    clear_image = tf.io.decode_jpeg(clear_image)
  
    # Read and decode the real image file to a uint8 tensor
    foggy_image = tf.io.read_file(foggy_image_file)
    foggy_image = tf.io.decode_jpeg(foggy_image)
  
    # Convert both images to float32 tensors
    clear_image = tf.cast(clear_image, tf.float32)
    foggy_image = tf.cast(foggy_image, tf.float32)
    
    # Apply normalization and random jitter to each tensor
    clear_image, foggy_image = normalize(clear_image, foggy_image)
    clear_image, foggy_image = random_jitter(clear_image, foggy_image)

    # Check for Nan values. These were used for troubleshooting
    tf.debugging.check_numerics(clear_image, "Clear image contains NaNs")
    tf.debugging.check_numerics(foggy_image, "Foggy image contains NaNs")
  
    return clear_image, foggy_image


def resize(clear_image, foggy_image, height, width):
  """This function uses the nearest neighbour method to resize provided clear and foggy images to the provided height and width"""
 
  clear_image = tf.image.resize(clear_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  foggy_image = tf.image.resize(foggy_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return clear_image, foggy_image

def random_crop(clear_image, foggy_image):
    """This function randomly crops an image to the global height and width required for the model"""

    # Stack the foggy and clear images to ensure the same crop
    stacked_image = tf.stack([clear_image, foggy_image], axis=0)

    # Store the original shape to restore later if needed
    original_shape = stacked_image.shape

    # Remove singleton dimensions if present
    if original_shape[1] == 1:
        stacked_image = tf.squeeze(stacked_image, axis=1)

    print("Stacked image shape before crop:", stacked_image.shape)

    # Randomly crop the stacked images
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    # Restore the original shape if an extra dimension was removed
    if original_shape[1] == 1:
        cropped_image = tf.expand_dims(cropped_image, axis=1)
    print(cropped_image.shape)
    print(cropped_image[0].shape)
    # Return the cropped images
    return cropped_image[0], cropped_image[1]
'''
def random_crop(clear_image, foggy_image):
  """This function randomly crops an image to the global height and width required for the model"""
  
  #Stack the foggy and clear images to ensure the same crop
  stacked_image = tf.stack([clear_image, foggy_image], axis=0)
  try:
      stacked_image = tf.squeeze(stacked_image, axis=1)  # Remove the second dimension if it's not needed
  except: pass
  print("Stacked image shape:", stacked_image.shape)
  #Randomly crop  the stacked images
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]
'''

def normalize(clear_image, foggy_image):
  """This function normalizes the pixel values which originally range from 0 to 255, to be in the range of -1 to 1"""
  clear_image = (clear_image / 127.5) - 1
  foggy_image = (foggy_image / 127.5) - 1

  return clear_image, foggy_image

@tf.function()
def random_jitter(clear_image, foggy_image):
  """This function uses the resize and random crop functions to apply random jittering to the image, so that it is not identical each time it is used for training
  Random mirroring is also applied"""
  # Resizing to 1316x756
  clear_image, foggy_image = resize(clear_image, foggy_image, 756, 1316)
  print("foggy image shape:", foggy_image.shape)
  # Random cropping back to 1280x720
  clear_image, foggy_image = random_crop(clear_image, foggy_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    clear_image = tf.image.flip_left_right(clear_image)
    foggy_image = tf.image.flip_left_right(foggy_image)

  return clear_image, foggy_image

#Define file paths for the clear and foggy images
clear_image_path = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_train_day")
foggy_image_path = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\synthetic_foggy_train_day")

# Load file names into paired lists
clear_images = list(clear_image_path.glob('*.jpg'))
foggy_images = [foggy_image_path / f'foggy_{img.name}' for img in clear_images]

# Convert file names to TensorFlow dataset
clear_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in clear_images])
foggy_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in foggy_images])

# Zip the datasets
dataset = tf.data.Dataset.zip((foggy_dataset, clear_dataset))


train_dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)  # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)  # Shuffle the dataset with a buffer size to ensure randomness in the data order.
train_dataset = train_dataset.batch(BATCH_SIZE)  # Group the dataset into batches of the specified size.
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.

#Define file paths for the clear and foggy images
clear_image_path_test = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_val_day")
foggy_image_path_test = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\old_synth_fog_val_day")

# Load file names into paired lists
clear_images_test = list(clear_image_path_test.glob('*.jpg'))
foggy_images_test = [foggy_image_path_test / f'foggy_{img.name}' for img in clear_images_test]

# Convert file names to TensorFlow dataset
clear_dataset_test = tf.data.Dataset.from_tensor_slices([str(p) for p in clear_images_test])
foggy_dataset_test = tf.data.Dataset.from_tensor_slices([str(p) for p in foggy_images_test])

# Zip the datasets
dataset_test = tf.data.Dataset.zip((foggy_dataset_test, clear_dataset_test))

test_dataset = dataset_test.map(load, num_parallel_calls=tf.data.AUTOTUNE) # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
test_dataset = test_dataset.shuffle(BUFFER_SIZE) # Shuffle the dataset with a buffer size to ensure randomness in the data order.
test_dataset = test_dataset.batch(BATCH_SIZE) # Group the dataset into batches of the specified size.
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE) # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.

#rgb format for output
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  """This function is used to reduce the height/width of the input and increase the number of feature channels"""
  
  initializer = tf.random_normal_initializer(0., 0.02)  # Initialize the weights with a normal distribution

  #initialize sequential model
  result = tf.keras.Sequential()
  
  # Apply 2D convolution with 'filters' number of output channels, 'size' kernel size, and stride of 2 to reduce spatial dimensions
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    # Apply batch normalization to stabilize and accelerate training
    result.add(tf.keras.layers.BatchNormalization())
    
  # Apply LeakyReLU activation to introduce non-linearity and allow some negative values
  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  """This function is used to increase the height/width of the input and decrease the number of feature channels. occurs after downsampling to recover dimensions""" 
  
  initializer = tf.random_normal_initializer(0., 0.02) # Initialize the weights with a normal distribution
  
  #initialize sequential model
  result = tf.keras.Sequential()
  
  # Apply transposed convolution (also known as deconvolution) to increase the spatial dimensions of the input. The 'filters' parameter determines the number of output channels
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

  # Apply batch normalization to stabilize and accelerate training
  result.add(tf.keras.layers.BatchNormalization())
  
  # Apply dropout to prevent overfitting by randomly setting some of the activations to zero
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  
    # Apply ReLU activation to introduce non-linearity
  result.add(tf.keras.layers.ReLU())

  return result

def generator():
    """This function defines the generator architecture. a Unet architecture is used, consisting of 8 downsample layers, 7 upsample layers, and a final conv layer"""
    # Define the input shape for the generator model
    inputs = tf.keras.layers.Input(shape=[720, 1280, 3])

    down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 640, 360, 64)
      downsample(128, 4),  # (batch_size, 320, 180, 128)
      downsample(256, 4),  # (batch_size, 160, 90, 256)
      downsample(512, 4),  # (batch_size, 80, 45, 512)
      downsample(512, 4),  # (batch_size, 40, 23, 512)
      downsample(512, 4),  # (batch_size, 20, 12, 512)
      downsample(512, 4),  # (batch_size, 10, 6, 512)
      downsample(512, 4),  # (batch_size, 5, 3, 512)
    ]

    up_stack = [
      upsample(512, 4, apply_dropout=True),  # (batch_size, 10, 6, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 20, 12, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 40, 23, 1024)
      upsample(512, 4),  # (batch_size, 80, 45, 1024)
      upsample(256, 4),  # (batch_size, 160, 90, 512)
      upsample(128, 4),  # (batch_size, 320, 180, 256)
      upsample(64, 4),  # (batch_size, 640, 360, 128)
    ]
    
    # Initialize weights for the last layer
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Last layer to produce the output imag
    last = tf.keras.layers.Conv2DTranspose(3, 4,strides=2,padding='same',kernel_initializer=initializer, activation='tanh')  # (batch_size, 720, 1280, 3)

    x = inputs #input image

    # Downsampling through the model
    skips = [] #list for skip connections
    for i, down in enumerate(down_stack):
      x = down(x) # Apply each downsampling layer
      #print(f"Downsample layer {i} output shape:", x.shape)  # Print shape of x after each downsample layer
      skips.append(x) # Store the output of each downsampling layer

    skips = reversed(skips[:-1])  # Reverse the list of skip connections (excluding the last one)

    # Upsampling and establishing the skip connections
    for i, (up, skip) in enumerate(zip(up_stack, skips)):
      x = up(x) # Apply each upsampling layer
      if x.shape[1:3] != skip.shape[1:3]:  # Check height and width dimensions
          x = ResizeLayer(target_size=[skip.shape[1], skip.shape[2]])(x) # Resize to match the skip connection
      #print(f"Upsample layer {i} output shape:", x.shape)  # Print shape of x after each upsample layer
      #print(f"Skip connection {i} shape:", skip.shape)     # Print shape of skip connection tensor
      x = tf.keras.layers.Concatenate()([x, skip]) # Concatenate with the skip connection
      #print(f"After concatenation {i} shape:", x.shape)    # Print shape of x after concatenation

    x = last(x) # Apply the last layer to produce the final output

    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  """This function describes the loss function used for training the generator"""
  # Compute GAN loss: Encourages the generator to produce images that are indistinguishable from real images
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Compute L1 loss: Measures the difference between the generated image and the target image
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  # Total generator loss: Combines GAN loss and L1 loss
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
  """This function describes the loss function used for training the discriminator"""
  # Compute loss for real images: Encourages the discriminator to correctly classify real images as real
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  # Compute loss for generated images: Encourages the discriminator to correctly classify generated images as fake
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  # Total discriminator loss: Combines the loss for real and generated images
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def discriminator():
  """This function is used to classify pairs of images as real or fake, based on how well the input matches the target"""
  
  # Initialize weights for the model
  initializer = tf.random_normal_initializer(0., 0.02)
  
  # Input layers for the discriminator
  inp = tf.keras.layers.Input(shape=[720, 1280, 3], name='foggy_image')
  tar = tf.keras.layers.Input(shape=[720, 1280, 3], name='clear_image')
  
  # Concatenate the foggy and clear images along the channel axis
  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  # Apply downsampling layers with batch normalization and LeakyReLU activations
  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  # Zero-padding to maintain spatial dimensions for the convolutional layer
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  
  # Convolutional layer to increase the depth of feature maps
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  # Batch normalization to stabilize training
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  # Apply LeakyReLU activation to introduce non-linearity
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  # Further zero-padding before final convolutional layer
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  # Final convolutional layer to output a single-channel classification
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def generate_images(model, test_input, tar):
  """This function is used to generate a plot consisting of the foggy input image, the model generated clear image, and the actual clear image"""
  # Generate predictions using the model for the given test input
  prediction = model(test_input, training=True)
  
  # Create a figure for displaying images
  plt.figure(figsize=(15, 15))

  # List of images to display: input image, ground truth (target), and generated prediction
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  # Iterate over the list of images and display them
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    
    # Normalize pixel values for plotting from [-1, 1] to [0, 1]
    plt.imshow(display_list[i] * 0.5 + 0.5)
    
    # Remove axes for better visualization
    plt.axis('off')
  
  # Show the plot with the images
  plt.show()


@tf.function
def train_step(input_image, target, step):
  """This function is a single training step for both the generator and discriminator, updating their weights based on the computed gradients and logging the results for analysis."""  
  target, input_image = random_jitter(target, input_image)
  # Record the operations for automatic differentiation
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # Generate output images from the input images using the generator
    gen_output = generator(input_image, training=True)
    
    # Get the discriminator's output for real and generated images
    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    # Calculate the generator's total loss, GAN loss, and L1 loss
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    
    # Calculate the discriminator's loss
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    # Check for NaNs or Infs in the loss values
    tf.debugging.check_numerics(gen_total_loss, "Generator total loss has NaNs or Infs")
    tf.debugging.check_numerics(disc_loss, "Discriminator loss has NaNs or Infs")

  # Compute gradients for generator and discriminator
  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  # Apply gradients to update model weights
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  # Log the losses for monitoring and visualization
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
    """This function is used to train the model over a defined number of steps while saving checkpoints to resume training if interrupted"""
    
    # Retrieve a single batch from the test dataset for visualization
    example_input, example_target = next(iter(test_ds.take(1)))
    
    start = time.time()  # Record the start time for performance measurement

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            # Clear the output to update the display with new images
            display.clear_output(wait=True)

            if step != 0:
                # Print the time taken for every 1000 steps
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()  # Reset the start time for the next interval

            # Generate and display images from the current model
            generate_images(generator, example_input, example_target)
            print(f"Step: {step//1000}k")  # Print the current training step in thousands

        # Perform a single training step
        train_step(input_image, target, step)

        # Print a dot for every 10 steps to indicate progress
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

# Set variables for the generator and discriminator
generator = generator()
discriminator = discriminator()

#Create checkp[oint object to manage saving and resuming
checkpoint_dir = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\checkpoints\train_conditional_day_final"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)


#Train the model
fit(train_dataset, test_dataset, steps=200000)


# Save the entire model
generator.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\generator_model_conditional_day_final.h5")
discriminator.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\discriminator_model_conditional_day_final.h5")

# Save the model in the TensorFlow SavedModel format
tf.saved_model.save(generator, r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\conditional_day_final")



# Load the entire model
#generator = tf.keras.models.load_model('path_to_save_the_model/generator_model')

# Load the SavedModel format
#loaded_model = tf.saved_model.load('path_to_save_the_model/saved_model')