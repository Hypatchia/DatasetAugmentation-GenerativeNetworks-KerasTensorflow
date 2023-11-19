# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".


import tensorflow as tf
import matplotlib.pyplot as plt
import os

BUFFER_SIZE = 600000


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


# Define the cross-entropy loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()



@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, noise_dim):
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



def train(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, noise_dim, num_examples_to_generate, epochs):
    """
    Trains the Generative Adversarial Network (GAN) for the specified number of epochs.

    Parameters:
    dataset (tf.data.Dataset): The training dataset containing batches of real images.
    epochs (int): The number of training epochs.

    Returns:
    None
    """
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Iterate through each epoch
    for epoch in range(epochs):

        # Iterate through each batch in the dataset
        for image_batch in dataset:
            train_step(image_batch)  # Execute a single training step

        # Generate and save sample images at the end of each epoch
        generate_and_save_images(generator, epoch + 1, seed)

    # Generate and save sample images after the final epoch
    generate_and_save_images(generator, epochs, seed)
