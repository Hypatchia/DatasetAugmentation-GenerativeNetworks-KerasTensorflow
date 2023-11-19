

import tensorflow as tf
def Generator():
    """
    Generator model for generating images using a deep convolutional neural network.

    Returns:
    tf.keras.Sequential: A Sequential model representing the generator.
    """
    # Define the generator model using Keras Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4 * 4 * 64, input_shape=(100,)),  # Dense layer with 4x4x64 units
        tf.keras.layers.Reshape((4, 4, 64)),  # Reshape to 4x4x64 tensor
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        # First convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        # Second convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        # Third convolutional transpose layer
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        # Output layer
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh')
    ], name="Generator")

    print(model.output_shape)  # Print the output shape of the generator model
    return model


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
        tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),  # Apply Leaky ReLU activation with slope 0.2

        tf.keras.layers.Flatten(),  # Flatten the output tensor
        tf.keras.layers.Dense(1, activation='sigmoid'),  # Output layer with sigmoid activation
    ], name="Discriminator")

    return model

