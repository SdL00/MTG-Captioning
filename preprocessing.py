import os
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast, RandomTranslation, RandomZoom, RandomRotation
import numpy as np
import requests
from settings import *


# Function to preprocess the captions: data is the DataFrame containing all the captions
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())             # to lowercase
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+"," "))  # replace consecutive whitespaces characters with a single space
    data['caption'] = data['caption'].apply(lambda x: x.replace("\n", " "))  # replace newline characters with a space
    data['caption'] = "startseq "+data['caption']+" endseq"                  # add startseq and endseq tokens at the beginning and the end of a caption
    return data


# Dividing into training, validation, and test sets
def split_data(images, captions_dataset):
    train_size = round(len(images) * 0.9)
    val_size = round(train_size * 0.8)

    # Convert captions_dataset to a list
    captions_list = list(captions_dataset)

    x_train = images[:val_size]  # training inputs (art crops)
    y_train = captions_list[:val_size]  # training targets (captions)
    x_val = images[val_size:train_size]  # validation inputs (art crops)
    y_val = captions_list[val_size:train_size]  # validation outputs (captions) 
    x_test = images[train_size:]  # test inputs (art crops)
    y_test = captions_list[train_size:]  # test outputs (captions) 

    return x_train, y_train, x_val, y_val, x_test, y_test


# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    RandomContrast(factor=(0.05, 0.15)), # Adds random contrast to the image in the range [5%, 15%]
    RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)), # Translates the image by a random fraction of the height and width
    RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)), # Zooms the image by a random factor
    RandomRotation(factor=(-0.10, 0.10)), # Rotates the image by a random factor
])


# Reading images from the dataset, decoding them into tensors, and resize them to a fixed shape. Eventually, applying data augmentation
def load_and_preprocess_image(img_path_str, data_aug):
    img_path_str = img_path_str.numpy().decode('utf-8')  # Decode the image path
    try:
        lowercase_path = img_path_str.lower()  # Convert the image path to lowercase

        if lowercase_path not in ['nan', 'n/a', '']:  # Check if the image path is valid (not None or 'nan')    
            response = requests.get(lowercase_path)  # Send a GET request to the image path

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                img = tf.image.decode_image(response.content, channels=3)  # Decode the image from the response content
                img = tf.image.resize(img, IMAGE_SIZE)  # Resize the image to the desired size

                if data_aug:
                    img = tf.expand_dims(img, 0)  # Add a dimension to the tensor at index 0
                    img = data_augmentation(img)  # Apply data augmentation
                    img = tf.squeeze(img, 0)  # Remove the dimension at index 0

                img = tf.image.convert_image_dtype(img, tf.float32)  # Convert the image to float32
                return img
            else:
                print(f"Failed to fetch image from {lowercase_path}. Status code: {response.status_code}")
        else:
            print(f"Invalid path: {lowercase_path}")
    except Exception as e:
        print(f"Error loading image from {img_path_str}: {e}")

    return tf.zeros(IMAGE_SIZE + (3,), dtype=tf.float32)  # Return a placeholder image


# Creating the dataset from the images and the captions
def create_data(images, captions, data_aug, tokenizer):
    img_paths = tf.constant(np.array(images))  # Convert the list of image URLs to a NumPy array and then to a TensorFlow constant tensor in order to guarantee compatibility with tensorflow operations
    img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)  # Create a TensorFlow dataset from the tensor of image paths
    img_dataset = img_dataset.map(lambda x: tf.py_function(load_and_preprocess_image, [x, data_aug], tf.float32), num_parallel_calls=AUTOTUNE)  # Apply the 'load_and_preprocess_image' function to each element in 'img_dataset' in parallel
    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(tokenizer, num_parallel_calls=AUTOTUNE)  # Create a TensorFlow dataset from the tensor of captions and apply the 'tokenizer' function to each element in parallel

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))  # Zip together the image dataset and the caption dataset to create pairs of (image, caption)

    # Batch the dataset into batches of size 'BATCH_SIZE'
    # Shuffle the batches with a buffer size of 'SHUFFLE_DIM'. It acts as a sliding window from which elements are randomly chosen to form the shuffled batch
    # Prefetch the batches for improved performance during training
    dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset


# To be used in the inference part
def read_image_inf(img_path_str):
    try:
        if img_path_str and img_path_str.lower() != 'nan':
            response = requests.get(img_path_str)

            if response.status_code == 200:
                img = tf.image.decode_image(response.content, channels=3)
                img = tf.image.resize(img, IMAGE_SIZE)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.expand_dims(img, axis=0)
                print("IMAGE SHAPE: ", img.shape)

                return img
            else:
                print(f"Failed to fetch image from {img_path_str}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error loading image from {img_path_str}: {e}")

    return tf.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
