from inference_tools import *
import json
import tensorflow as tf
import numpy as np
import argparse
from settings import *

# Load Tokenizer model
tokenizer = tf.keras.models.load_model(tokenizer_path)
tokenizer = tokenizer.layers[1]

# Load inference model
model = get_inference_model(get_model_config_path)

# Load model weights
model.load_weights(get_model_weights_path)

# Test and generate caption for an input image
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument('--image', help="Path to image file.")
image_path = parser.parse_args().image

with open(get_model_config_path) as json_file:
    model_config = json.load(json_file)


print("STARTING PREDICTION...")
text_caption = generate_caption(image_path, model, tokenizer, model_config["SEQ_LENGTH"])
print("PREDICT CAPTION : %s" %(text_caption))
