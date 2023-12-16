import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
from model import build_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from preprocessing import read_image_inf
import numpy as np
import json
import re
from settings import *


def save_tokenizer(tokenizer, path_save):
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = tokenizer(input)
    model = tf.keras.Model(input, output)
    model.save(path_save + "tokenizer", save_format='tf')


def get_inference_model(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]

    cnn_model = build_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    # Initializing the model input
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, training, decoder_input])

    return caption_model


def get_inference_model_fine_tune(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]

    cnn_model = build_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    # Initializing the model input
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = True
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, training, decoder_input])

    return caption_model


def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1
    print("BEFORE READ IMAGE")
    img = read_image_inf(image_path)  # Read the image from the path and resize it
    print("AFTER READ IMAGE")
    img = caption_model.cnn_model(img)  # Extract the features of the input image
    print("IMG %s" %(img))
    encoded_img = caption_model.encoder(img, training=False)  # Encode the extracted features
    print("ENCODED IMG %s" %(encoded_img))

    # Generating the caption
    decoded_caption = "startseq" # Initializing the caption with the start sequence token
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]  # Converting the caption to tokens
        mask = tf.math.not_equal(tokenized_caption, 0)  # Creating a mask to ignore padding tokens 
        print("TOKENIZED CAPTION %s" %(tokenized_caption))
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask  # Predicting the next token using the decoder
        )
        print("PREDICTIONS %s" %(predictions))
        sampled_token_index = np.argmax(predictions[0, i, :])  # Getting the index of the most probable token
        print("SAMPLED TOKEN INDEX %s" %(sampled_token_index))
        sampled_token = index_lookup[sampled_token_index]  # Decoding the most probable token to a word
        print("SAMPLED TOKEN: %s" %(sampled_token))
        if sampled_token == "endseq":
            break
        #print("BEFORE DECODED CAPTION")
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("startseq ", "").strip()
    