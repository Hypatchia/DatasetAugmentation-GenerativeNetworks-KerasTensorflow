# %%
# Necessary Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import zipfile
import shutil
import tensorflow_addons as tfa

from google.colab import files

# %%
# Download dataset from drive
!gdown "DATASET ID"

# %%
# Enable mixed precision training which uses float16 for training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# %%
def extract_dataset(dataset_file_path, extraction_path):
    """
    Extracts a zipped dataset to a specified extraction path.

    Parameters:
    - dataset_file_path (str): Path to the zipped dataset file.
    - extraction_path (str): Directory where the images will be extracted.

    Returns:
    None

    """
    # Create the extraction directory if it doesn't exist
    os.makedirs(extraction_path, exist_ok=True)

    # Extract the zipped dataset
    with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
        



# %%
# Path to the dataset zip file
dataset_file_path = '/content/TuberculosisDataset.zip'
extraction_path = '/content/'


# %%
# Extract the dataset
extract_dataset(dataset_file_path, extraction_path)

# %%
# Set dataset Directory
Dataset_dir = '/content/Tuberculosis/'

# %%
# Set dataset information
img_shape=(64,64)
batch_size=16
# Get image paths from dataset directory
img_paths = [os.path.join(Dataset_dir, img) for img in os.listdir(Dataset_dir)]

# %%
def preprocess_image(img_path, img_shape):
    """
    Preprocesses an image by loading it, converting it to a grayscale array,
    and normalizing pixel values to the range [0, 1].

    Parameters:
    - img_path (str): Path to the image file.
    - img_shape (tuple): Target size of the image (default is (224, 224)).

    Returns:
    - img_array (numpy.ndarray): Preprocessed image as a NumPy array.
    """
    # Load the image
    img = load_img(img_path, color_mode='grayscale', target_size=img_shape)

    # Convert the image to a NumPy array
    img_array = img_to_array(img)

    # Normalize pixel values to the range [0, 1]
    img_array /= 255.0

    return img_array

# %%

def create_tf_dataset(img_paths, batch_size, img_shape):
    """
    Creates a TensorFlow dataset from a list of image paths.

    Parameters:
    - img_paths (list): List of file paths to the images.
    - batch_size (int): Batch size for the dataset (default is 16).
    - img_shape (tuple): Target size of the images (default is (224, 224)).

    Returns:
    - dataset (tf.data.Dataset): Configured TensorFlow dataset.
    """
    # Create a TensorFlow Dataset from image paths
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)

    # Map the preprocess_image function to each element in the dataset
    dataset = dataset.map(lambda x: tf.numpy_function(preprocess_image, [x, img_shape], tf.float32))

    # Specify batch size and shuffle
    dataset = dataset.shuffle(len(img_paths)).batch(batch_size)

    # Adjust the shape of the elements in the dataset
    dataset = dataset.map(lambda x: tf.reshape(x, (-1,) + img_shape + (1,)))

    # Prefetch for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


# %%
# Create a TensorFlow dataset
dataset = create_tf_dataset(img_paths, batch_size, img_shape)


# %%
# Visualize a batch of images
def visualize_batch(dataset):
    """
    Visualizes a batch of images from a TensorFlow dataset.

    Parameters:
    - dataset (tf.data.Dataset): TensorFlow dataset containing images.

    Returns:
    None
    """
    # Get a batch of images
    img_batch = next(iter(dataset))

    # Plot the images
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_batch[i].numpy().squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# %%
def Generator():
    """
    Generator model for generating images using a deep convolutional neural network.

    Returns:
    tf.keras.Sequential: A Sequential model representing the generator.
    """
    # Define the generator model using Keras Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4 * 4 * 64, input_shape=(50,)),  # Dense layer with 4x4x64 units
        tf.keras.layers.Reshape((4, 4, 64)),  # Reshape to 4x4x64 tensor
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        # First convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        # Second convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        # Third convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        # Fourth convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        # Output layer
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='tanh')
    ], name="Generator")

    print(model.output_shape)  # Print the output shape of the generator model
    return model

# %%
# Create the generator
generator = Generator()

# %%

def Discriminator(input_shape=(64, 64, 1)):
    """
    Discriminator model for classifying images as real or fake.

    Parameters:
    input_shape (tuple): The shape of the input images. Default is (64, 64, 1).

    Returns:
    tf.keras.Sequential: A Sequential model representing the discriminator.
    """

    # Define the discriminator model using Keras Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3


        tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3


        tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3

        
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2
        tf.keras.layers.Dropout(0.3), # Apply dropout rate of 0.3



        tf.keras.layers.Flatten(),  # Flatten the output tensor
        tf.keras.layers.Dense(1, activation='sigmoid'),  # Output layer with sigmoid activation
    ], name="Discriminator")

    return model


# %%
# Create the discriminator
discriminator = Discriminator()

# %%
# Get cross entropy loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# %%
# Define discriminator loss function
def discriminator_loss(real_output, fake_output):
    """
    Calculates the total discriminator loss using binary cross-entropy.

    Parameters:
    real_output (tf.Tensor): Output from the discriminator for real images.
    fake_output (tf.Tensor): Output from the discriminator for fake (generated) images.

    Returns:
    tf.Tensor: Total discriminator loss.
    """

    # Binary cross-entropy loss for real and fake samples
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # Total discriminator loss is the sum of real and fake losses
    total_loss = real_loss + fake_loss

    return total_loss


# %%
# Define generator loss function
def generator_loss(fake_output):
    """
    Calculates the generator loss using binary cross-entropy.

    Parameters:
    fake_output (tf.Tensor): Output from the discriminator for fake (generated) images.

    Returns:
    tf.Tensor: Generator loss.
    """

    # Binary cross-entropy loss for generated samples
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return loss

# %%

# Define a function to log losses
def log_losses(epoch, gen_loss, disc_loss):
    """
    Log generator and discriminator losses for a given epoch.

    Parameters:
        epoch (int): The current training epoch.
        gen_loss (tf.Tensor): Generator loss for the epoch.
        disc_loss (tf.Tensor): Discriminator loss for the epoch.
    """
    # Print the formatted log message containing epoch, generator loss, and discriminator loss.
    print(f"Epoch {epoch}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")


# %%
# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

# %%
# Set training parameters
EPOCHS = 2000
noise_dim = 50
num_examples_to_generate = 8
BUFFER_SIZE = 600000
batch_size = 64


# %%
# You will reuse this seed overtime (so it's easier)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# %%
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    """
    Executes a single training step for the Generative Adversarial Network (GAN).

    Parameters:
    images (tf.Tensor): A batch of real images used for training.

    Returns:
    None
    """

    # Generate random noise for the generator input
    noise = tf.random.normal([batch_size, noise_dim])

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images using the generator
        generated_images = generator(noise, training=True)

        # Get discriminator outputs for real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate generator and discriminator losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradients of generator and discriminator with respect to their parameters
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply the gradients to update the generator and discriminator weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss


# %%
def generate_and_save_images(model, epoch, test_input, save_dir='images'):
    """
    Generates and saves images using the provided generator model.

    Parameters:
    model (tf.keras.Model): The generator model used for generating images.
    epoch (int): The current training epoch.
    test_input (tf.Tensor): Random noise used as input for image generation.
    save_dir (str): The directory where generated images will be saved. Defaults to 'images'.

    Returns:
    None
    """

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a subplot grid for visualizing generated images
    fig, axs = plt.subplots(2, 4, figsize=(8, 4))
    axs = axs.flatten()

    # Display each generated image in the subplot grid
    for i in range(predictions.shape[0]):
        axs[i].imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        axs[i].axis('off')

    print("epoch:", epoch)

    # Save the figure to the specified directory
    file_path = os.path.join(save_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(file_path)

    # Close the figure without displaying it
    plt.close(fig)


# %%
def train(dataset, epochs):
    """
    Trains the Generative Adversarial Network (GAN) for the specified number of epochs.

    Parameters:
    dataset (tf.data.Dataset): The training dataset containing batches of real images.
    epochs (int): The number of training epochs.

    Returns:
    None
    """

    # Iterate through each epoch
    for epoch in range(epochs):

        # Iterate through each batch in the dataset
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        # Log losses at the end of each epoch
        log_losses(epoch, gen_loss, disc_loss)

        # Generate and save images after each epoch
        generate_and_save_images(generator, epoch + 1, seed)

    # Generate and save sample images after the final epoch
    generate_and_save_images(generator, epochs, seed)


# %%
# Train the model
train(dataset, EPOCHS)

# %%
# Zip the directory of generated images
zip_filename = 'images'
shutil.make_archive(zip_filename, 'zip', 'images')

# Download the zipped directory to local storage
files.download("/content/images.zip")
# %%
# Generate New Images
# generate a number of images

def generate_and_save_images(generator, discriminator, save_dir, num_images=2800,noise_dim=50):
    
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)

    os.makedirs(save_dir, exist_ok=True)
    rows, cols = int(num_images**0.5), int(num_images**0.5) + 1
    while rows * cols < num_images:
        cols += 1

    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
        image_path = os.path.join(save_dir, f'generated_image_{i}.png')
        plt.imsave(image_path, generated_images[i, :, :, 0].numpy(), cmap='gray')

    plt.close()

    return save_dir, num_images


# %%
# Generate and save images    
save_dir = 'content/Generated'
generate_and_save_images(generator,discriminator,save_dir,2800,noise_dim)

# %%
# Zip the directory
zip_filename = save_dir
shutil.make_archive(zip_filename, 'zip', 'Generated')

# %%
# Save Generator & Discriminator models

generator.save('Generator')
discriminator.save('Generator')

# Zip & Download the models
zip_filename = 'Generator'
shutil.make_archive(zip_filename, 'zip', 'Generator')

zip_filename = 'Discriminator'
shutil.make_archive(zip_filename, 'zip', 'Discriminator')


# %%
# Zip the directory of generated images
zip_filename = 'images'
shutil.make_archive(zip_filename, 'zip', 'images')

# %%
# Download the zipped directory to local storage
files.download("/content/images.zip")

# Download the zipped models to local storage
files.download("/content/Generator.zip")
files.download("/content/Discriminator.zip")


# %%
# Use shutil.rmtree to delete the directory and its contents
directory_path ="/content/Tuberculosis"
shutil.rmtree(directory_path)

# %%
# Use os.remove to delete the file at the specified path
file_path ="/content/TuberculosisDataset.zip"
os.remove(file_path)

# %%




