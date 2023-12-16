import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import efficientnet
from settings import *


# Defining a function to build a Convolutional Neural Network (CNN) model
def build_cnn_model():
    # Creating an instance of EfficientNetB0 with specified input shape and pre-trained weights
    base_model = efficientnet.EfficientNetB0(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False  # Freezing the layers of the feature extractor
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, 1280))(base_model_out)  # The output layer has 1280 filters
    cnn_model = tf.keras.Model(base_model.input, base_model_out)  # Creating a model with the specified input and output

    return cnn_model


# Defining a class for a Transformer Encoder Block
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Creating layers for multi-head attention, dense projection, and layer normalization
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense_proj(inputs)  # Apply dense projection to the input
        # Applying multi-head attention to the projected input
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        proj_input = self.layernorm(
            inputs + attention_output)  # Adding the input and attention output, then normalizing the result

        return proj_input


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # Creating embeddings for tokens and positions
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]  # Getting the length of the input sequence
        positions = tf.range(start=0, limit=length, delta=1)  # Generating positions from 0 to length-1
        # Embedding tokens and positions, then add them
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)  # Defining a mask to prevent attention being applied to padding tokens


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        # Creating layers for multi-head attention, dense projection, and layer normalization
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        # Creating a Positional Embedding layer
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=self.vocab_size
        )
        self.out = layers.Dense(self.vocab_size)  # Output layer for the model
        # Dropout layers for regularization
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True   

    def call(self, inputs, encoder_outputs, training, mask=None):
        # Applying positional embedding and dropout to the inputs
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        inputs = self.dropout_1(inputs, training=training)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)  # Masking the padding tokens
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)  # Masking the future tokens
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else :
            combined_mask = None
            padding_mask  = None

        # Applying the first multi-head attention layer
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask#None
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Applying the second multi-head attention layer with encoder outputs
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask#None
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Applying dense projection and layer normalization
        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.dropout_2(proj_out, training=training)  # Applying dropout for regularization

        preds = self.out(proj_out)  # Generating predictions using the output layer
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)  # Getting the shape of the input
        batch_size, sequence_length = input_shape[0], input_shape[1]  # Getting the batch size and sequence length
        
        # Creating indices for the lower triangular part of the matrix
        i = tf.range(sequence_length)[:, tf.newaxis]  
        j = tf.range(sequence_length)
        # Creating a boolean mask where each element is True if i >= j, and False otherwise
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        # Replicating the causal attention mask across the batch dimension   
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        # Initializing CNN, encoder, and decoder models
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

        # Initializing metrics for tracking loss and accuracy
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def call(self, inputs):
        x = self.cnn_model(inputs[0])  # Passing the input images to the CNN
        x = self.encoder(x, False)  # Passing the CNN outputs to the encoder
        x = self.decoder(inputs[2], x, training=inputs[1], mask=None)  # Passing the captions and encoder outputs to the decoder and training boolean
        return x

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)  # Calculating the loss    
        mask = tf.cast(mask, dtype=loss.dtype)  
        loss *= mask  # Applying the padding mask to the loss
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)  # Computing the masked loss

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2)) # Comparing the true and predicted sequences
        accuracy = tf.math.logical_and(mask, accuracy)  # Applying the padding mask
        accuracy = tf.cast(accuracy, dtype=tf.float32)  
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)  # Computing the masked accuracy  

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_features = self.cnn_model(batch_img)
        with tf.GradientTape() as tape:
            # Passing image features to encoder
            encoder_out = self.encoder(img_features, training=True)
            batch_seq_inp = batch_seq[:, :-1]
            batch_seq_true = batch_seq[:, 1:]
            # Passing the encoder outputs, sequence inputs to the decoder and the mask
            mask = tf.math.not_equal(batch_seq_inp, 0)
            batch_seq_pred = self.decoder(
                batch_seq_inp, encoder_out, training=True, mask=mask
            )
            # Computing metrics
            loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            acc = self.calculate_accuracy(
                batch_seq_true, batch_seq_pred, mask
            )
            # Updating the batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        # Getting the trainable weights and the gradients
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)

        # Applying the gradients to the optimizer to update the weights
        self.optimizer.apply_gradients(zip(grads, train_vars))

        loss = batch_loss
        acc = batch_acc

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_features = self.cnn_model(batch_img)
        encoder_out = self.encoder(img_features, training=False)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]  
        mask = tf.math.not_equal(batch_seq_inp, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=False, mask=mask
        )
 
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        batch_loss += loss
        batch_acc += acc
        loss = batch_loss
        acc = batch_acc

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
