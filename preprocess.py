import json
import string
import numpy as np
from config import (
    MAX_BODY_LEN, MAX_SUBJECT_LEN, MIN_SUBJECT_LEN,
    START_TOKEN, END_TOKEN, MODEL_DIR
)


def load_and_clean_data() -> tuple[list, list]:
    """
    Load AESLC dataset and return cleaned (body, subject) pairs.

    Steps:
    1. Load via HuggingFace datasets
    2. Combine train, validation, test splits
    3. Extract email_body and subject_line fields
    4. Truncate body to MAX_BODY_LEN characters
    5. Lowercase and strip whitespace
    6. Filter by length constraints
    7. Add START_TOKEN and END_TOKEN to subjects
    8. Print: total pairs loaded, vocabulary sizes, length distributions

    Returns:
        (bodies, subjects) — two parallel lists of cleaned strings
    """
    from datasets import load_dataset
    dataset = load_dataset('aeslc')

    all_bodies = []
    all_subjects = []

    for split in ['train', 'validation', 'test']:
        for example in dataset[split]:
            body = example['email_body']
            subject = example['subject_line']

            if body is None or subject is None:
                continue

            body = body[:MAX_BODY_LEN].lower().strip()
            subject = subject.lower().strip()

            # Keep only printable ASCII
            printable = set(string.printable)
            body = ''.join(c for c in body if c in printable)
            subject = ''.join(c for c in subject if c in printable)

            if len(body) < 20:
                continue
            if len(subject) < MIN_SUBJECT_LEN:
                continue
            if len(subject) > MAX_SUBJECT_LEN:
                continue

            subject = START_TOKEN + subject + END_TOKEN

            all_bodies.append(body)
            all_subjects.append(subject)

    body_lengths = [len(b) for b in all_bodies]
    subject_lengths = [len(s) for s in all_subjects]

    print(f"Total pairs loaded: {len(all_bodies)}")
    print(f"Body length  — min: {min(body_lengths)}, max: {max(body_lengths)}, mean: {np.mean(body_lengths):.1f}")
    print(f"Subject length — min: {min(subject_lengths)}, max: {max(subject_lengths)}, mean: {np.mean(subject_lengths):.1f}")

    body_vocab = set(''.join(all_bodies))
    subject_vocab = set(''.join(all_subjects))
    print(f"Body vocabulary size: {len(body_vocab)}")
    print(f"Subject vocabulary size: {len(subject_vocab)}")

    return all_bodies, all_subjects


def build_vocabulary(texts: list[str]) -> tuple[dict, dict]:
    """
    Build character-level vocabulary from a list of strings.

    Returns:
        (char_to_idx, idx_to_char) — bidirectional character mappings
        Includes special tokens: PAD (index 0), UNK (index 1),
        START_TOKEN, END_TOKEN
    """
    chars = set()
    for text in texts:
        chars.update(text)

    # Reserve 0 for PAD, 1 for UNK
    sorted_chars = ['<PAD>', '<UNK>'] + sorted(chars)
    char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
    idx_to_char = {i: c for i, c in enumerate(sorted_chars)}

    return char_to_idx, idx_to_char


def encode_sequences(
    texts: list[str],
    char_to_idx: dict,
    max_len: int,
    padding: str = 'post'
) -> np.ndarray:
    """
    Encode a list of strings as integer sequences, padded to max_len.

    Returns:
        numpy array of shape (len(texts), max_len)
    """
    unk_idx = char_to_idx.get('<UNK>', 1)
    pad_idx = char_to_idx.get('<PAD>', 0)

    encoded = np.full((len(texts), max_len), pad_idx, dtype=np.int32)

    for i, text in enumerate(texts):
        seq = [char_to_idx.get(c, unk_idx) for c in text[:max_len]]
        if padding == 'post':
            encoded[i, :len(seq)] = seq
        else:
            # pre-padding
            start = max_len - len(seq)
            encoded[i, start:] = seq

    return encoded


def save_vocab(char_to_idx: dict, idx_to_char: dict,
               path: str = 'models/vocab.json') -> None:
    """Save vocabulary mappings to JSON."""
    MODEL_DIR.mkdir(exist_ok=True)
    data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': {str(k): v for k, v in idx_to_char.items()}
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Vocabulary saved to {path}")


def load_vocab(path: str = 'models/vocab.json') -> tuple[dict, dict]:
    """Load vocabulary mappings from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    char_to_idx = data['char_to_idx']
    idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
    return char_to_idx, idx_to_char
