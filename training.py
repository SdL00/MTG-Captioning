import pandas as pd
import numpy as np
import warnings
import json
from preprocessing import *
from model import *
from inference_tools import save_tokenizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

warnings.filterwarnings('ignore')

# Uploading the dataset
mtg_ita = pd.read_csv('data/modified_italian_dataset.csv')

# Selecting relevant columns
mtg_ita = mtg_ita[['image_uris.art_crop', 'name', 'printed_name', 'mana_cost', 'colors', 'color_identity', 'type_line',
                 'set', 'rarity', 'power', 'toughness', 'oracle_text', 'printed_text', 'flavor_text']]

# Dropping rows with NaN values
mtg_ita_cleaned = mtg_ita.dropna(axis=0)

# mtg_x is the input art crop (image) and mtg_y is the caption
mtg_x = mtg_ita_cleaned['image_uris.art_crop']
mtg_y = mtg_ita_cleaned[['name', 'printed_name', 'mana_cost', 'colors', 'color_identity', 'type_line',
                 'set', 'rarity', 'power', 'toughness', 'oracle_text', 'printed_text', 'flavor_text']]

# Combining the textual parts into one big caption
mtg_y['caption'] = mtg_y.apply(lambda row: '; '.join([f"{col}: {{{val}}}" for col, val in row.dropna().items()]), axis=1)

# Preprocessing our captions
mtg_y = text_preprocessing(mtg_y)
mtg_y = mtg_y['caption']

# Tokenizing the captions
tokenizer = TextVectorization(
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
)

# Computing a vocabulary of string terms
tokenizer.adapt(mtg_y)

VOCAB_SIZE = len(tokenizer.get_vocabulary())
print("VOCAB SIZE: ", VOCAB_SIZE)

# Splitting the dataset into training, validation, and test sets
x_train, y_train, x_val, y_val, x_test, y_test = split_data(mtg_x, mtg_y)

print("Number of training samples: ", len(x_train))
print("Number of validation samples after splitting with training set: ", len(x_val))
print("Number of test samples after splitting with training set: ", len(x_test))

# Datasets are batched, shuffled, and prefetched to improve performance
train_dataset = create_data(list(x_train), list(y_train), data_aug=True, tokenizer=tokenizer)
valid_dataset = create_data(list(x_val), list(y_val), data_aug=False, tokenizer=tokenizer)
test_dataset = create_data(list(x_test), list(y_test), data_aug=False, tokenizer=tokenizer)

# Building the model
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

# Defining the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# Defining the EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

# Creating the learning rate schedule (Exponential Decay)
initial_learning_rate = 0.00005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

# Defining the optimizer (Adam)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Compiling the model
caption_model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])

# Training the model
history = caption_model.fit(train_dataset,
                            epochs=EPOCHS,
                            validation_data=valid_dataset,
                            shuffle=False,
                            callbacks=[early_stopping])

# Evaluating the model
train_metrics = caption_model.evaluate(train_dataset, batch_size=BATCH_SIZE)
valid_metrics = caption_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
test_metrics = caption_model.evaluate(test_dataset, batch_size=BATCH_SIZE)

print("Train Loss = %.4f - Train Accuracy = %.4f" % (train_metrics[0], train_metrics[1]))
print("Valid Loss = %.4f - Valid Accuracy = %.4f" % (valid_metrics[0], valid_metrics[1]))
print("Test Loss = %.4f - Test Accuracy = %.4f" % (test_metrics[0], test_metrics[1]))

# Saving history of the trained model
history_dict = history.history
json.dump(history_dict, open(SAVE_DIR + 'history.json', 'w'))

# Saving weights of the trained model
caption_model.save_weights(SAVE_DIR + 'model_weights.h5')

# Saving config of the trained model
config_train = {"IMAGE_SIZE": IMAGE_SIZE,
                "SEQ_LENGTH": SEQ_LENGTH,
                "EMBED_DIM": EMBED_DIM,
                "NUM_HEADS": NUM_HEADS,
                "FF_DIM": FF_DIM,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "VOCAB_SIZE": VOCAB_SIZE}

json.dump(config_train, open(SAVE_DIR + 'config_train.json', 'w'))

# Saving tokenizer of the trained model
save_tokenizer(tokenizer, SAVE_DIR)
