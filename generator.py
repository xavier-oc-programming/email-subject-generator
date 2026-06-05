import numpy as np
import tensorflow as tf
from pathlib import Path
from config import (
    TEMPERATURE, NUM_SUGGESTIONS, MAX_BODY_LEN, MAX_SUBJECT_LEN,
    START_TOKEN, END_TOKEN, MODEL_DIR
)
from preprocess import load_vocab, encode_sequences

_encoder_model = None
_decoder_model = None
_char_to_idx = None
_idx_to_char = None
_vocab_loaded = False


def _load_models():
    global _encoder_model, _decoder_model, _char_to_idx, _idx_to_char, _vocab_loaded

    if _vocab_loaded:
        return

    vocab_path = MODEL_DIR / 'vocab.json'
    encoder_path = MODEL_DIR / 'inference_encoder.keras'
    decoder_path = MODEL_DIR / 'inference_decoder.keras'

    if not (vocab_path.exists() and encoder_path.exists() and decoder_path.exists()):
        raise FileNotFoundError("Model files not found. Run train.py first.")

    _char_to_idx, _idx_to_char = load_vocab(str(vocab_path))
    _encoder_model = tf.keras.models.load_model(str(encoder_path))
    _decoder_model = tf.keras.models.load_model(str(decoder_path))
    _vocab_loaded = True


def generate_subject(
    body_text: str,
    temperature: float = TEMPERATURE,
    max_len: int = MAX_SUBJECT_LEN
) -> str:
    """
    Generate a subject line from an email body.

    Steps:
    1. Preprocess body_text (lowercase, truncate, encode)
    2. Run through inference encoder to get context vector [h, c]
    3. Initialize decoder with START_TOKEN and context vector
    4. Loop:
       a. Get next character probabilities from decoder
       b. Apply temperature scaling
       c. Decode index to character
       d. Append to generated sequence
       e. If END_TOKEN or max_len reached: stop
       f. Update decoder state with new [h, c]
    5. Strip START_TOKEN, END_TOKEN, and padding
    6. Return generated subject line
    """
    _load_models()

    # Step 1: preprocess
    body_text = body_text[:MAX_BODY_LEN].lower().strip()
    body_encoded = encode_sequences([body_text], _char_to_idx, MAX_BODY_LEN, padding='post')

    # Step 2: encode body to get context vector
    states = _encoder_model.predict(body_encoded, verbose=0)
    state_h, state_c = states[0:1], states[1:2]

    # Step 3: initialize decoder with START_TOKEN
    start_idx = _char_to_idx.get(START_TOKEN, 1)
    target_seq = np.array([[start_idx]])

    generated = []

    # Step 4: generation loop
    for _ in range(max_len):
        # a. Get next character probabilities
        output = _decoder_model.predict([target_seq, state_h, state_c], verbose=0)
        char_probs = output[0][0, 0, :]
        new_state_h = output[1]
        new_state_c = output[2]

        # b. Temperature scaling:
        # Dividing log probabilities by temperature before softmax
        # (equivalently raising probabilities to 1/temperature) sharpens
        # the distribution at low temperature (more deterministic) and
        # flattens it at high temperature (more random). This is the
        # standard sampling trick for text generation.
        char_probs = char_probs.astype(np.float64)
        char_probs = np.power(char_probs, 1.0 / temperature)
        char_probs = char_probs / char_probs.sum()
        next_char_idx = np.random.choice(len(char_probs), p=char_probs)

        # c. Decode index to character
        next_char = _idx_to_char.get(next_char_idx, '')

        # d. Append to generated sequence
        generated.append(next_char)

        # e. Stop if END_TOKEN or max_len
        if next_char == END_TOKEN:
            break

        # f. Update decoder state
        target_seq = np.array([[next_char_idx]])
        state_h = new_state_h
        state_c = new_state_c

    # Step 5: strip tokens and join
    result = ''.join(generated)
    result = result.replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
    return result


def generate_multiple(
    body_text: str,
    n: int = NUM_SUGGESTIONS,
    temperature_range: tuple = (0.5, 0.9)
) -> list[dict]:
    """
    Generate n subject line suggestions with varying temperatures.
    Deduplicates results.

    Returns:
        List of dicts: [{"subject": str, "temperature": float, "style": str}]

    style labels:
      temperature < 0.6: "conservative"
      0.6 <= temperature <= 0.8: "balanced"
      temperature > 0.8: "creative"
    """
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n)

    results = []
    seen = set()

    for temp in temperatures:
        temp = float(round(temp, 2))
        subject = generate_subject(body_text, temperature=temp)

        if subject and subject not in seen:
            seen.add(subject)

            if temp < 0.6:
                style = 'conservative'
            elif temp <= 0.8:
                style = 'balanced'
            else:
                style = 'creative'

            results.append({
                'subject': subject,
                'temperature': temp,
                'style': style
            })

    return results
