import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam
from config import LATENT_DIM, EMBEDDING_DIM, MAX_BODY_LEN, LEARNING_RATE


def build_encoder(vocab_size: int) -> tuple:
    """
    Build the encoder model.

    Architecture:
      Input: integer sequence of shape (MAX_BODY_LEN,)
      Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)
      LSTM(LATENT_DIM, return_state=True)
      Outputs: encoder_outputs, state_h, state_c

    The encoder reads the email body character by character.
    The LSTM's final hidden state (state_h) and cell state (state_c)
    capture the meaning of the entire input sequence in a fixed-size
    vector — the context vector. This is what gets passed to the decoder.

    Returns:
        (encoder_model, encoder_inputs, encoder_states)
        where encoder_states = [state_h, state_c]
    """
    encoder_inputs = Input(shape=(MAX_BODY_LEN,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(LATENT_DIM, return_state=True, name='encoder_lstm')(encoder_embedding)
    encoder_states = [state_h, state_c]

    encoder_model = Model(encoder_inputs, encoder_states, name='encoder')

    return encoder_model, encoder_inputs, encoder_states


def build_decoder(vocab_size: int, encoder_states: list) -> tuple:
    """
    Build the decoder model for training.

    Architecture:
      Input 1: decoder input sequence (teacher forcing during training)
      Input 2: encoder states [state_h, state_c] as initial states
      Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)
      LSTM(LATENT_DIM, return_sequences=True, return_state=True,
           initial_state=encoder_states)
      Dense(vocab_size, activation='softmax')

    Teacher forcing feeds the correct previous character as input at each
    decoder step during training. This makes training stable and fast. At
    inference, we use the model's own previous output instead — this
    difference between training and inference is why we need separate
    training and inference models.

    Returns:
        (decoder_outputs, decoder_states, decoder_inputs,
         decoder_embedding, decoder_lstm, decoder_dense)
    """
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name='decoder_embedding')
    decoder_embedded = decoder_embedding(decoder_inputs)

    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    decoder_states = [state_h, state_c]

    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_lstm_outputs)

    return decoder_outputs, decoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense


def build_training_model(encoder_inputs, decoder_inputs, decoder_outputs) -> Model:
    """
    Combine encoder and decoder into a single training model.

    Compile with:
      optimizer=Adam(LEARNING_RATE)
      loss='sparse_categorical_crossentropy'
      metrics=['accuracy']

    sparse_categorical_crossentropy because targets are integer indices,
    not one-hot vectors. This is more memory efficient for character-level
    models with large vocabularies.

    Returns: compiled Keras Model
    """
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='training_model')
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_inference_models(encoder_inputs, encoder_states,
                            decoder_inputs, decoder_embedding,
                            decoder_lstm, decoder_dense,
                            vocab_size: int) -> tuple:
    """
    Build separate encoder and decoder models for inference.

    The inference decoder is stateful — it maintains the LSTM state
    between calls, updating it at each generation step. This is the key
    difference from the training model.

    Returns:
        (inference_encoder_model, inference_decoder_model)
    """
    inference_encoder = Model(encoder_inputs, encoder_states, name='inference_encoder')

    decoder_state_input_h = Input(shape=(LATENT_DIM,), name='decoder_state_h')
    decoder_state_input_c = Input(shape=(LATENT_DIM,), name='decoder_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_lstm_outputs, state_h, state_c = decoder_lstm(
        decoder_embedded, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_lstm_outputs)

    inference_decoder = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name='inference_decoder'
    )

    return inference_encoder, inference_decoder
