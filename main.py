
import os
import zipfile
import shutil
import tensorflow as tf
from scripts.load_process_data import extract_dataset, create_tf_dataset
from scripts.build_gan import Generator, Discriminator
from scripts.train_model import train




if __name__ == '__main__':
  
    # Enable mixed precision training which uses float16 for training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    # Path to the dataset zip file
    dataset_file_path = '/content/TuberculosisDataset.zip'
    extraction_path = '/content/'

    # Extract the dataset
    extract_dataset(dataset_file_path, extraction_path)


    # Set dataset Directory
    Dataset_dir = '/content/Tuberculosis/'


    # Set dataset information
    img_shape=(64,64)
    batch_size=64
    # Get image paths from dataset directory
    img_paths = [os.path.join(Dataset_dir, img) for img in os.listdir(Dataset_dir)]

    # Create a TensorFlow dataset from the image paths
    dataset = create_tf_dataset(img_paths, batch_size, img_shape)

    # Build the model
    generator = Generator()

    discriminator = Discriminator()
    # Set the optimizers
    learning_rate = 1e-5
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Set training parameters
    EPOCHS = 200
    batch_size = 64
    noise_dim = 50
    num_examples_to_generate = 8
   
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Train the model
    train(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, noise_dim, num_examples_to_generate, EPOCHS)

    # Save the trained models
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    # Zip the directory of generated images
    zip_filename = 'images'
    shutil.make_archive(zip_filename, 'zip', 'images')




