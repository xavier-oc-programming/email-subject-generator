import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    MODEL_DIR, PLOTS_DIR, MAX_BODY_LEN, MAX_SUBJECT_LEN,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, NUM_SUGGESTIONS
)
from preprocess import (
    load_and_clean_data, build_vocabulary, encode_sequences,
    save_vocab
)
from model import (
    build_encoder, build_decoder, build_training_model, build_inference_models
)
from generator import generate_subject


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_ran = len(history['loss'])
    epochs_x = range(1, epochs_ran + 1)

    ax1.plot(epochs_x, history['loss'], label='Train Loss')
    ax1.plot(epochs_x, history['val_loss'], label='Val Loss')
    best_epoch = int(np.argmin(history['val_loss'])) + 1
    ax1.axvline(best_epoch, color='red', linestyle='--', label=f'Best epoch ({best_epoch})')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs_x, history['accuracy'], label='Train Accuracy')
    ax2.plot(epochs_x, history['val_accuracy'], label='Val Accuracy')
    ax2.axvline(best_epoch, color='red', linestyle='--', label=f'Best epoch ({best_epoch})')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_length_distribution(lengths, title, xlabel, save_path):
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=40, color='steelblue', edgecolor='white')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_sample_generations(bodies, subjects, save_path, n=10):
    samples = list(zip(bodies[:n], subjects[:n]))
    generated = []
    for body, true_subject in samples:
        gen = generate_subject(body)
        generated.append(gen)

    fig, ax = plt.subplots(figsize=(16, n * 0.8 + 1))
    ax.axis('off')

    table_data = [['Body (first 80 chars)', 'True Subject', 'Generated Subject']]
    for (body, true_sub), gen in zip(samples, generated):
        # strip start/end tokens from true subject for display
        true_display = true_sub.strip('<>').strip()
        table_data.append([body[:80] + '...', true_display, gen])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2])
    plt.title('Sample Generations', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading and cleaning data...")
    bodies, subjects = load_and_clean_data()

    print("\nBuilding vocabulary...")
    all_texts = bodies + subjects
    char_to_idx, idx_to_char = build_vocabulary(all_texts)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    save_vocab(char_to_idx, idx_to_char, str(MODEL_DIR / 'vocab.json'))

    print("\nEncoding sequences...")
    body_encoded = encode_sequences(bodies, char_to_idx, MAX_BODY_LEN, padding='post')

    # +2 for START_TOKEN and END_TOKEN already included in subjects
    subject_encoded = encode_sequences(subjects, char_to_idx, MAX_SUBJECT_LEN + 2, padding='post')

    # Teacher forcing: decoder input is subject shifted right by one position.
    # At each step the model sees the correct previous character and predicts the next one.
    decoder_input = subject_encoded[:, :-1]
    decoder_target = subject_encoded[:, 1:]

    print(f"Body encoded shape: {body_encoded.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder target shape: {decoder_target.shape}")

    print("\nBuilding models...")
    encoder_model, encoder_inputs, encoder_states = build_encoder(vocab_size)
    decoder_outputs, _, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = build_decoder(vocab_size, encoder_states)
    training_model = build_training_model(encoder_inputs, decoder_inputs, decoder_outputs)
    training_model.summary()

    inference_encoder, inference_decoder = build_inference_models(
        encoder_inputs, encoder_states,
        decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense,
        vocab_size
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_DIR / 'training_model_best.keras'),
                        save_best_only=True, monitor='val_loss', verbose=1)
    ]

    print("\nTraining...")
    history = training_model.fit(
        [body_encoded, decoder_input],
        decoder_target,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    print("\nSaving models...")
    training_model.save(str(MODEL_DIR / 'training_model_best.keras'))
    inference_encoder.save(str(MODEL_DIR / 'inference_encoder.keras'))
    inference_decoder.save(str(MODEL_DIR / 'inference_decoder.keras'))

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f)

    model_config = {
        'vocab_size': vocab_size,
        'latent_dim': 256,
        'embedding_dim': 64,
        'max_body_len': MAX_BODY_LEN,
        'max_subject_len': MAX_SUBJECT_LEN,
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'training_samples': len(bodies),
    }
    with open(MODEL_DIR / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    print("\nGenerating plots...")
    plot_training_curves(history_dict, PLOTS_DIR / '01_training_curves.png')

    subject_lengths = [len(s) - 2 for s in subjects]  # exclude start/end tokens
    plot_length_distribution(
        subject_lengths,
        'Subject Line Length Distribution',
        'Length (characters)',
        PLOTS_DIR / '02_subject_length_distribution.png'
    )

    body_lengths = [len(b) for b in bodies]
    plot_length_distribution(
        body_lengths,
        'Email Body Length Distribution (after truncation)',
        'Length (characters)',
        PLOTS_DIR / '03_body_length_distribution.png'
    )

    plot_sample_generations(bodies, subjects, PLOTS_DIR / '04_sample_generations.png')

    print("\nTraining complete.")
    print(f"Best val_loss: {model_config['best_val_loss']:.4f}")
    print(f"Best val_accuracy: {model_config['best_val_accuracy']:.4f}")


if __name__ == '__main__':
    main()
