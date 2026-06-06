"""
Fine-tune T5-small on the AESLC email subject corpus.

Run once from the terminal (not via notebook — takes 30-90 min on CPU):
    python train_t5.py

Saves to models/t5/ — the directory generator.py loads from.
"""

import gc
import json
import os
from pathlib import Path

# Must be set before importing torch or transformers on macOS arm64
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate framework

import datasets as _ds
_ds.disable_caching()
from datasets import load_dataset

# transformers/torch are imported inside main() after datasets filtering is complete.
# Importing torch before datasets' multiprocessing initializes its thread pool in a
# state that conflicts with SentencePiece pthreads on macOS arm64.

from config import (
    MODEL_DIR, MAX_BODY_LEN,
    T5_MODEL_NAME, T5_PREFIX, T5_MAX_INPUT_LEN, T5_MAX_TARGET_LEN, T5_MODEL_DIR,
)

T5_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter(dataset):
    def keep(example):
        body = (example['email_body'] or '').strip()
        subj = (example['subject_line'] or '').strip()
        return len(body) >= 20 and 3 <= len(subj) <= 60

    return {
        split: dataset[split].filter(keep, num_proc=1)
        for split in dataset.keys()
    }


def build_preprocess_fn(tokenizer):
    def preprocess(examples):
        inputs = [T5_PREFIX + body[:MAX_BODY_LEN] for body in examples['email_body']]
        targets = examples['subject_line']

        model_inputs = tokenizer(
            inputs,
            max_length=T5_MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=T5_MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )

        # -100 tells the trainer to ignore padding in the loss
        label_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels['input_ids']
        ]
        model_inputs['labels'] = label_ids
        return model_inputs

    return preprocess


def print_samples(model, tokenizer, dataset, n=5):
    print('\n--- Sample generations ---')
    for i in range(min(n, len(dataset))):
        body = dataset[i]['email_body'][:MAX_BODY_LEN]
        gold = dataset[i]['subject_line']

        inputs = tokenizer(
            T5_PREFIX + body,
            return_tensors='pt',
            max_length=T5_MAX_INPUT_LEN,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=T5_MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'  body   : {body[:80]}')
        print(f'  gold   : {gold}')
        print(f'  t5     : {pred}')
        print()


def main():
    print('Loading AESLC dataset...')
    raw = load_dataset('aeslc')
    data = load_and_filter(raw)
    del raw; gc.collect()  # release datasets state before torch/transformers initialize threads
    print(f"  train={len(data['train'])}  validation={len(data['validation'])}  test={len(data['test'])}")

    # T5Tokenizer (SentencePiece) must be loaded before torch initializes its thread pool.
    # Importing T5ForConditionalGeneration triggers torch init, which conflicts with
    # SentencePiece pthreads on macOS arm64 if loaded after.
    print(f'Loading {T5_MODEL_NAME}...')
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)

    from transformers import (
        T5ForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    import torch
    torch.set_num_threads(1)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)

    print('Tokenizing...')
    preprocess = build_preprocess_fn(tokenizer)
    tokenized = {
        split: data[split].map(
            preprocess, batched=True,
            remove_columns=data[split].column_names,
            num_proc=1,
        )
        for split in data.keys()
    }

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(T5_MODEL_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=False,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        processing_class=tokenizer,
        data_collator=collator,
    )

    print('Fine-tuning T5-small (this takes 30–90 min on CPU)...')
    trainer.train()

    print(f'Saving model to {T5_MODEL_DIR}...')
    trainer.save_model(str(T5_MODEL_DIR))
    tokenizer.save_pretrained(str(T5_MODEL_DIR))

    import shutil
    for ckpt in T5_MODEL_DIR.glob('checkpoint-*'):
        shutil.rmtree(ckpt)
    (T5_MODEL_DIR / 'training_args.bin').unlink(missing_ok=True)

    config = {
        'model_name': T5_MODEL_NAME,
        'prefix': T5_PREFIX,
        'max_input_len': T5_MAX_INPUT_LEN,
        'max_target_len': T5_MAX_TARGET_LEN,
        'epochs': int(training_args.num_train_epochs),
        'train_size': len(data['train']),
    }
    with open(MODEL_DIR / 't5_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print_samples(model, tokenizer, data['test'])
    print('Done. Run the FastAPI server to use T5 for inference.')


if __name__ == '__main__':
    main()
