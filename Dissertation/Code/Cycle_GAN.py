# -*- coding: utf-8 -*-
"""
2024/06/07
Author: Nicholas Bandy
"""

import tensorflow as tf
import pathlib
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
tf.keras.backend.clear_session()

#Using mixed precision (float 16 and float32) due to gpu memory errors only using float32
#Due to this, values need to be cast back to float32 before calculations
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

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

#Using mixed precision (float 16 and float32) due to gpu memory errors only using float32
#Due to this, values need to be cast back to float32 before calculations
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #define binary cross entropy for losses

AUTOTUNE = tf.data.AUTOTUNE # Constant provided by tensorflow used for optimization, removes need for manually tune some parameters
BUFFER_SIZE = 100 #The number of images loaded into memory to be shuffled at a given time. Ensures model doesnt learn order of data
BATCH_SIZE = 1 #Number of images processed at a time before updating weights

# Size and shape of all images for training in the model
IMG_WIDTH = 1280
IMG_HEIGHT = 720
input_shape = (720, 1280, 3)
inputs = tf.keras.layers.Input(shape=input_shape)

def random_crop(image):
    """This function randomly crops an image to the global height and width required for the model"""
    
    #Check for grayscale and convert to rgb (later found to be a bug but kept this part)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    original_shape = image.shape
    #randomly crop the image
    if original_shape[0] == 1:
        image = tf.squeeze(image, axis=0)
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    if original_shape[0] == 1:
        cropped_image = tf.expand_dims(cropped_image, axis=0)
    return cropped_image

def normalize(image):
    """This function normalizes the pixel values which originally range from 0 to 255, to be in the range of -1 to 1"""
    
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    """This function uses the resize and random crop functions to apply random jittering to the image, so that it is not identical each time it is used for training
    Random mirroring is also applied"""
    
    # Resizing to 1316x756 using crop or padding
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)
   
    # Random cropping back to 1280x720
    image = tf.image.resize(image, [756, 1316], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    if image.shape[-1] == 1:  # Check if image is grayscale. No longer needed but code remains
        print(image.shape)
        image = tf.image.grayscale_to_rgb(image)
        print(image.shape)
        
    image = random_crop(image)
    
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    
    return image

def load_image(image_file):
    """This funcion is used to load and decode an image"""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    
    if image.shape[-1] == 1:  # Check if image is grayscale. no longer required, kept for troubleshooting
        print(image.shape)
        image = tf.image.grayscale_to_rgb(image)
        print(image.shape)
        
    return image

def preprocess_image_train(image_file):
    """This function prepares an image for training by applying random jitter and normalization"""
    image = load_image(image_file)
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image_file):
    """This function prepares an image for test by loading and normalizing it"""
    image = load_image(image_file)
    image = normalize(image)
    return image

# Define paths for train and test data
clear_image_path = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_train_day")
foggy_image_path = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\foggy_train_new")
clear_image_path_test = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\clear_val_day")
foggy_image_path_test = pathlib.Path(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\foggy_val_new")

def check_image_shape(image_path):
    """This function finds and returns the shape of an image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image.shape

# Load file names into seperate lists
clear_images = list(clear_image_path.glob('*.jpg'))
foggy_images = list(foggy_image_path.glob('*.jpg'))

# Convert file names to TensorFlow dataset
clear_train_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in clear_images])
foggy_train_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in foggy_images])

foggy_train_dataset = foggy_train_dataset.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
foggy_train_dataset = foggy_train_dataset.shuffle(BUFFER_SIZE)  # Shuffle the dataset with a buffer size to ensure randomness in the data order.
foggy_train_dataset = foggy_train_dataset.batch(BATCH_SIZE) # Group the dataset into batches of the specified size.
foggy_train_dataset = foggy_train_dataset.prefetch(AUTOTUNE) # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.

clear_train_dataset = clear_train_dataset.map(preprocess_image_train, num_parallel_calls=AUTOTUNE) # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
clear_train_dataset = clear_train_dataset.shuffle(BUFFER_SIZE) # Shuffle the dataset with a buffer size to ensure randomness in the data order.
clear_train_dataset = clear_train_dataset.batch(BATCH_SIZE) # Group the dataset into batches of the specified size.
clear_train_dataset = clear_train_dataset.prefetch(AUTOTUNE) # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.
 
# Load file names into seperate lists
clear_images_test = list(clear_image_path_test.glob('*.jpg'))
foggy_images_test = list(foggy_image_path_test.glob('*.jpg'))

# Convert file names to TensorFlow dataset
clear_test_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in clear_images_test])
foggy_test_dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in foggy_images_test])

foggy_test_dataset = foggy_test_dataset.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
foggy_test_dataset = foggy_test_dataset.shuffle(BUFFER_SIZE)  # Shuffle the dataset with a buffer size to ensure randomness in the data order.
foggy_test_dataset = foggy_test_dataset.batch(BATCH_SIZE) # Group the dataset into batches of the specified size.
foggy_test_dataset = foggy_test_dataset.prefetch(AUTOTUNE) # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.

clear_test_dataset = clear_test_dataset.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) # Apply the 'load' function to each element in the dataset in parallel, optimizing performance with AUTOTUNE.
clear_test_dataset = clear_test_dataset.shuffle(BUFFER_SIZE)  # Shuffle the dataset with a buffer size to ensure randomness in the data order.
clear_test_dataset = clear_test_dataset.batch(BATCH_SIZE) # Group the dataset into batches of the specified size.
clear_test_dataset = clear_test_dataset.prefetch(AUTOTUNE) # Preload data for the next step while the current step is still running, optimizing input pipeline performance with AUTOTUNE.

# Define a sample foggy image from test to visualize generator performance after each epoch
for image in foggy_train_dataset.take(1):
    sample_foggy = image[0]
print("Sample foggy image shape:", sample_foggy.shape)
sample_foggy = next(iter(foggy_train_dataset.take(1)))


OUTPUT_CHANNELS = 3 # rgb format
LAMBDA = 100 # Used as a multiplier in cycle loss and identity loss. Raising lambda prioritizes mean difference in loss, making the result closer to the original.
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True) # define bce loss
EPOCHS = 100 #number of epochs to train for

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size):
        super(ResizeLayer, self).__init__()
        self.target_size = target_size # Target size to resize the input to

    def build(self, input_shape):
        super().build(input_shape)  # Initializes the layer; no custom weights are used here

    def call(self, inputs):
        # Resizes the input tensor to the specified target size
        target_height, target_width = self.target_size[0], self.target_size[1]
        resized = tf.image.resize(inputs, [target_height, target_width])
        return resized

    def get_config(self):
        config = super().get_config()
        # Stores the target size in the layer's configuration
        config.update({'target_size': self.target_size})
        return config


def downsample(filters, size, apply_batchnorm=True):
    """This function is used to reduce the height/width of the input and increase the number of feature channels"""
    
    initializer = tf.random_normal_initializer(0., 0.02) # Initialize the weights with a normal distribution
    
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
    
    # Initialize the weights with a normal distribution
    initializer = tf.random_normal_initializer(0., 0.02)

    #initialize sequential model
    result = tf.keras.Sequential()
    
    # Apply transposed convolution (also known as deconvolution) to increase the spatial dimensions of the input. The 'filters' parameter determines the number of output channels
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))

     # Apply batch normalization to stabilize and accelerate training
    result.add(tf.keras.layers.BatchNormalization())

    # Apply dropout to prevent overfitting by randomly setting some of the activations to zero
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

      # Apply ReLU activation to introduce non-linearity
    result.add(tf.keras.layers.ReLU())

    return result



def discriminator():
    """This function is used to classify pairs of images as real or fake, based on how well the input matches the target"""
    
    initializer = tf.random_normal_initializer(0., 0.02)  # Initialize weights for the model

    #input layer
    x = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

      # Apply downsampling layers with batch normalization and LeakyReLU activations
    down1 = downsample(64, 4, False)(x)  # (batch_size, 640, 360, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 320, 180, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 160, 90, 256)
    down4 = downsample(512, 4)(down3)  # (batch_size, 80, 45, 512)
    down5 = downsample(512, 4)(down4)  # (batch_size, 40, 23, 512)
    down6 = downsample(512, 4)(down5)  # (batch_size, 20, 12, 512)
    down7 = downsample(512, 4)(down6)  # (batch_size, 10, 6, 512)
    
    # Zero-padding to maintain spatial dimensions for the convolutional layer
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down7)  # (batch_size, 12, 8, 512)
    
    # Convolutional layer to increase the depth of feature maps
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (batch_size, 9, 5, 512)

    # Batch normalization to stabilize training
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    
    # Apply LeakyReLU activation to introduce non-linearity
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    # Further zero-padding before final convolutional layer
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 11, 7, 512)

    # Final convolutional layer to output a single-channel classification
    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)  # (batch_size, 8, 4, 1)

    return tf.keras.Model(inputs=x, outputs=last)

def resnet_block(x, filters, size=3):
    """A residual block for a ResNet-based generator."""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Save the input tensor to add it back after the convolutions
    res = x
    
    # First convolution layer in the residual block
    x = tf.keras.layers.Conv2D(filters, size, padding='same', kernel_initializer=initializer)(x)
    
    # Apply Batch Normalization to stabilize and accelerate training
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Apply ReLU activation to introduce non-linearity
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolution layer in the residual block
    x = tf.keras.layers.Conv2D(filters, size, padding='same', kernel_initializer=initializer)(x)
    
    # Apply Batch Normalization again
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add the input tensor (residual connection) to the output of the second convolution
    x = tf.keras.layers.Add()([x, res])
    
    # Apply ReLU activation to introduce non-linearity
    return tf.keras.layers.ReLU()(x)

def resnet_generator():
    """Creates a ResNet-based generator model for the CycleGAN. The generator consists of  an initial convolution layer,
    downsampling layers, multiple residual blocks, upsampling layers, final convolution layer."""
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Define the input tensor for the generator
    inputs = tf.keras.layers.Input(shape=[720, 1280, 3])  # Define the input tensor
    
    x = inputs
    
    # Initial Convolution Layer
    x = tf.keras.layers.Conv2D(64, 7, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Downsampling layers
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Residual blocks
    for _ in range(12 ):
        x = resnet_block(x, 256) # Apply 12 residual blocks with 256 filters each
    
    # Upsampling layers
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Final Convolution Layer, use 'tanh' activation to produce outputs in the range [-1, 1]
    x = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 7, padding='same', kernel_initializer=initializer, activation='tanh')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


# define generators and discriminators
generator_g = resnet_generator()
generator_f = resnet_generator()
discriminator_x = discriminator()
discriminator_y = discriminator()

# unused generator loss, updated to bce
#def generator_loss(disc_generated_output):
#    return tf.cast(tf.reduce_mean(tf.losses.mean_squared_error(tf.ones_like(disc_generated_output), disc_generated_output)), tf.float32)

def generator_loss(generated_output):
    """This function describes the loss function used for training the generator"""
    return tf.cast(bce(tf.ones_like(generated_output), generated_output), tf.float32)

def calc_cycle_loss(real_image, cycled_image):
    """This function calculates cycle consistency loss for the generator, used to ensure an image resembles the original after domain transfer"""
    real_image = tf.cast(real_image, tf.float32)
    cycled_image = tf.cast(cycled_image, tf.float32)
    return LAMBDA * tf.cast(tf.reduce_mean(tf.abs(real_image - cycled_image)), tf.float32)

def identity_loss(real_image, same_image):
    """This function calcyulates identity loss, helps in preserving color and content in target domain"""
    real_image = tf.cast(real_image, tf.float32)
    same_image = tf.cast(same_image, tf.float32)
    return LAMBDA * 0.5 * tf.cast(tf.reduce_mean(tf.abs(real_image - same_image)), tf.float32)

def discriminator_loss(real_output, generated_output):
    """Compute the adversarial loss for the discriminator."""
    real_loss = tf.cast(bce(tf.ones_like(real_output), real_output), tf.float32)     # Discriminator loss for real images (label is 1)
    generated_loss = tf.cast(bce(tf.zeros_like(generated_output), generated_output), tf.float32)    # Total loss is the sum of losses for real and fake images
    total_disc_loss = real_loss + generated_loss     # Total loss is the sum of losses for real and fake images
    return total_disc_loss


# Initialize optimizers. Uses Adam algorithm
generator_g_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(6e-5, beta_1=0.5)

#Create checkp[oint object to manage saving and resuming
checkpoint_path = r"C:\Users\nicho\OneDrive\Desktop\Dissertation\checkpoints\train_cycle_day\train_cycle_fin"
ckpt = tf.train.Checkpoint(generator_g=generator_g,generator_f=generator_f,discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


def generate_images(model, test_input):
    """This function is used to generate a plot consisting of the foggy input image, and the model generated clear image"""
    pass
    # Generate predictions using the model for the given test input
    prediction = model(test_input)
    
    prediction = tf.cast(prediction, dtype=tf.float32)
    test_input = tf.cast(test_input, dtype=tf.float32)
    
    # Create a figure for displaying images
    plt.figure(figsize=(12, 12))

    # List of images to display: input image, and generated prediction
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    
    # Iterate over the list of images and display them
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def train_step(real_x, real_y):
    """This function performs one step of training the model, which consists of 2 generators and 2 discriminators"""
    
    real_x = random_jitter(real_x)
    real_y = random_jitter(real_y)
    with tf.GradientTape(persistent=True) as tape:
       # Generate fake images
        fake_y = generator_g(real_x, training=True)
        fake_x = generator_f(real_y, training=True)
        
        # Reconstruct images
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
        
        # Identity mappings
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        # Discriminator outputs
        disc_real_x = discriminator_x(real_x, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
        
        # Losses for generators
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        identity_loss_x = identity_loss(real_x, same_x)
        identity_loss_y = identity_loss(real_y, same_y)
        
        total_gen_g_loss = gen_g_loss + cycle_loss + identity_loss_y
        total_gen_f_loss = gen_f_loss + cycle_loss + identity_loss_x
        
        # Losses for discriminators
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
    # Calculate gradients for generators and discriminators
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
    
    # Apply gradients to update weights
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    del tape  # Free up memory

#Loop through the desired number of epochs for training. Plot sample after each epoch and save a checkpoint every 5 epochs
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((foggy_train_dataset, clear_train_dataset)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1
    
    clear_output(wait=True)
    generate_images(generator_g, sample_foggy)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Time taken for epoch {epoch+1} is {time.time()-start:.4f} sec')

#Generate from final model
generate_images(generator_g, sample_foggy)

# Save the entire model
generator_g.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\generator_model_cycle_day_final_defogging.h5")
discriminator_x.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\discriminator_x_model_cycle_final.h5")

generator_f.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\generator_model_cycle_day_final_fogging.h5")
discriminator_y.save(r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_models\discriminator_y_model_cycle_final.h5")

# Save the model in the TensorFlow SavedModel format
tf.saved_model.save(generator_g, r"C:\Users\nicho\OneDrive\Desktop\Dissertation\saved_model_cycle_day_final")

