import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from config import (
    NUM_SUGGESTIONS, MAX_BODY_LEN,
    T5_PREFIX, T5_MAX_INPUT_LEN, T5_MAX_TARGET_LEN, T5_MODEL_DIR,
)

_model = None
_tokenizer = None
_loaded = False


def _load_models():
    global _model, _tokenizer, _loaded

    if _loaded:
        return

    if not T5_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"T5 model not found at {T5_MODEL_DIR}. Run train_t5.py first."
        )

    _tokenizer = AutoTokenizer.from_pretrained(str(T5_MODEL_DIR))
    _model = T5ForConditionalGeneration.from_pretrained(str(T5_MODEL_DIR))
    _model.eval()
    _loaded = True


def generate_subject(
    body_text: str,
    temperature: float = 0.7,
    max_len: int = T5_MAX_TARGET_LEN,
) -> str:
    _load_models()

    input_text = T5_PREFIX + body_text[:MAX_BODY_LEN]
    inputs = _tokenizer(
        input_text,
        return_tensors='pt',
        max_length=T5_MAX_INPUT_LEN,
        truncation=True,
    )

    with torch.no_grad():
        if temperature <= 0.6:
            # conservative: beam search — deterministic, picks the most likely sequence
            outputs = _model.generate(
                inputs['input_ids'],
                max_new_tokens=max_len,
                num_beams=4,
                early_stopping=True,
            )
        else:
            # balanced / creative: temperature sampling
            outputs = _model.generate(
                inputs['input_ids'],
                max_new_tokens=max_len,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
            )

    subject = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return subject.strip()


def generate_multiple(
    body_text: str,
    n: int = NUM_SUGGESTIONS,
    temperature_range: tuple = (0.5, 0.9),
) -> list[dict]:
    """
    Generate n subject line suggestions across a temperature range.

    Returns:
        List of dicts: [{"subject": str, "temperature": float, "style": str}]
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
                'style': style,
            })

    return results
