import pandas as pd
import numpy as np
import warnings
import json
from preprocessing import *
from settings import *
from model import *
from inference_tools import *
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf

warnings.filterwarnings('ignore')

# Loading Tokenizer model
tokenizer = tf.keras.models.load_model("results_training_last/tokenizer")
tokenizer = tokenizer.layers[1]

VOCAB_SIZE = len(tokenizer.get_vocabulary())
print("VOCAB SIZE: ", VOCAB_SIZE)

# Loading the dataset
mtg_ita = pd.read_csv('data/modified_italian_dataset.csv')

mtg_ita = mtg_ita[['image_uris.art_crop', 'name', 'printed_name', 'mana_cost', 'colors', 'color_identity', 'type_line',
                 'set', 'rarity', 'power', 'toughness', 'oracle_text', 'printed_text', 'flavor_text']]

mtg_ita_cleaned = mtg_ita.dropna(axis=0)

# Shuffling the dataset
mtg_ita_cleaned = mtg_ita_cleaned.sample(frac=1).reset_index(drop=True)

# Selecting relevant columns, s.t. mtg_x is the input art crop (image) and mtg_y is the caption
mtg_x = mtg_ita_cleaned['image_uris.art_crop']
mtg_y = mtg_ita_cleaned[['name', 'printed_name', 'mana_cost', 'colors', 'color_identity', 'type_line',
                 'set', 'rarity', 'power', 'toughness', 'oracle_text', 'printed_text', 'flavor_text']]

# Combining the textual parts into one big caption
mtg_y['caption'] = mtg_y.apply(lambda row: '; '.join([f"{col}: {{{val}}}" for col, val in row.dropna().items()]), axis=1)

# Preprocessing our captions
mtg_y = text_preprocessing(mtg_y)
mtg_y = mtg_y['caption']

# Dividing into training, validation, and test sets
x_train_fine_tune, y_train_fine_tune, x_val_fine_tune, y_val_fine_tune, x_test_fine_tune, y_test_fine_tune = split_data(mtg_x, mtg_y)

print("Number of training samples: ", len(x_train_fine_tune))
print("Number of validation samples after splitting with training set: ", len(x_val_fine_tune))
print("Number of test samples after splitting with training set: ", len(x_test_fine_tune))

# Fine-tuning datasets
train_dataset_fine_tune = create_data(list(x_train_fine_tune), list(y_train_fine_tune), data_aug=True, tokenizer=tokenizer)
valid_dataset_fine_tune = create_data(list(x_val_fine_tune), list(y_val_fine_tune), data_aug=False, tokenizer=tokenizer)
test_dataset_fine_tune = create_data(list(x_test_fine_tune), list(y_test_fine_tune), data_aug=False, tokenizer=tokenizer)


# Defining the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

# Creating a learning rate schedule
initial_learning_rate = 0.000005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Loading the pre-trained model
caption_model = get_inference_model_fine_tune("results_training_last/config_train.json")

# Compiling the model for fine-tuning
caption_model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])

# Loading pre-trained weights
caption_model.load_weights("results_training_last/model_weights.h5")

# Fine-tuning the model
fine_tune_history = caption_model.fit(train_dataset_fine_tune,
                                      epochs=EPOCHS,
                                      shuffle=False,
                                      validation_data=valid_dataset_fine_tune,
                                      callbacks=[early_stopping])

# Computing definitive metrics
train_metrics_ft = caption_model.evaluate(train_dataset_fine_tune, batch_size=BATCH_SIZE)
valid_metrics_ft = caption_model.evaluate(valid_dataset_fine_tune, batch_size=BATCH_SIZE)
test_metrics_ft = caption_model.evaluate(test_dataset_fine_tune, batch_size=BATCH_SIZE)

print("Train Loss fine_tuned = %.4f - Train Accuracy fine_tuned = %.4f" % (train_metrics_ft[0], train_metrics_ft[1]))
print("Valid Loss fine_tuned = %.4f - Valid Accuracy fine_tuned = %.4f" % (valid_metrics_ft[0], valid_metrics_ft[1]))
print("Test Loss fine_tuned = %.4f - Test Accuracy fine_tuned = %.4f" % (test_metrics_ft[0], test_metrics_ft[1]))

# Saving fine-tuned weights and history
ft_history_dict = fine_tune_history.history
json.dump(ft_history_dict, open(SAVE_DIR_FINE_TUNE + "fine_tune_history.json", 'w'))
caption_model.save_weights(SAVE_DIR_FINE_TUNE + "fine_tuned_model_weights.h5")

# Saving config model
config_ft = {"IMAGE_SIZE": IMAGE_SIZE,
                "SEQ_LENGTH": SEQ_LENGTH,
                "EMBED_DIM": EMBED_DIM,
                "NUM_HEADS": NUM_HEADS,
                "FF_DIM": FF_DIM,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "VOCAB_SIZE": VOCAB_SIZE}

json.dump(config_ft, open(SAVE_DIR_FINE_TUNE + 'config_ft.json', 'w'))

# Saving tokenizer
save_tokenizer(tokenizer, SAVE_DIR_FINE_TUNE + "fine_tuned_tokenizer")
