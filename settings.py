import tensorflow as tf

SEQ_LENGTH = 130 # Sequence length of the captions (longer sequences are truncated)
AUTOTUNE = tf.data.AUTOTUNE # Automaticaly tune the parallelism of data loading
IMAGE_SIZE = (224, 224) # Size of the input images
BATCH_SIZE = 16
SHUFFLE_DIM = 1024 # Buffer size of the shuffle operation
EMBED_DIM = 512  # Embedding dimension
FF_DIM = 1024 # Hidden layer size in feed forward network inside transformer
NUM_HEADS = 8 # Number of attention heads inside transformer
EPOCHS = 100

# Tokenizer trained model saved path
tokenizer_path = "results_training/tokenizer"

get_model_config_path = "results_training/config_train.json"
# Weights trained model saved path
get_model_weights_path = "results_training/model_weights.h5"

# Tokenizer fine-tuned model saved path
tokenizer_path_fine_tune = "results_training_fine_tuned/fine_tuned_tokenizertokenizer"
# Config fine-tuned model saved path
get_model_config_path_fine_tune = "results_training_fine_tuned/config_ft.json"
# Weights fine-tuned model saved path
get_model_weights_path_fine_tune = "results_training_fine_tuned/fine_tuned_model_weights.h5"

# Directories where the results will be saved
SAVE_DIR = "results_training/"
SAVE_DIR_FINE_TUNE = "results_training_fine_tuned/"
