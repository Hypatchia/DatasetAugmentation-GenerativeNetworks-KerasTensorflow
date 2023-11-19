from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
import zipfile


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
    dataset = dataset.map(lambda x: tf.numpy_function(preprocess_image, [x], tf.float32))

    # Specify batch size and shuffle
    dataset = dataset.shuffle(len(img_paths)).batch(batch_size)

    # Adjust the shape of the elements in the dataset
    dataset = dataset.map(lambda x: tf.reshape(x, (-1,) + img_shape + (1,)))

    # Prefetch for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
