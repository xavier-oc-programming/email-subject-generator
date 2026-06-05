from pathlib import Path

# Data
DATA_DIR = Path('data')
MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')

# Preprocessing
MAX_BODY_LEN = 200
MAX_SUBJECT_LEN = 60
MIN_SUBJECT_LEN = 3
START_TOKEN = '<'
END_TOKEN = '>'

# Model architecture
LATENT_DIM = 256
# the dimensionality of the encoder hidden state and the
# decoder hidden state. 256 gives sufficient capacity for short
# subject line generation without being too large for CPU training.

EMBEDDING_DIM = 64
# each character is embedded into a 64-dimensional vector
# before being fed to the LSTM. This is more efficient than one-hot
# encoding and allows the model to learn character similarity.

# Training
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001

# Inference
TEMPERATURE = 0.7
# temperature controls randomness during generation.
# temperature < 1.0: more conservative, repeats common patterns
# temperature = 1.0: sample directly from model distribution
# temperature > 1.0: more random, more creative, less coherent
# 0.7 is a good default for professional subject lines.

NUM_SUGGESTIONS = 5
# generate this many subject lines per request by
# running inference NUM_SUGGESTIONS times with slight temperature
# variation. Gives the user options to choose from.

# Azure
AZURE_APP_NAME = 'email-subject-gen-xoc'
AZURE_RESOURCE_GROUP = 'email-subject-gen-rg'
AZURE_LOCATION = 'westeurope'
