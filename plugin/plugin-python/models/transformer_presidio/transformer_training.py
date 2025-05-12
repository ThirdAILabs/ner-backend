import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import Trainer, TrainingArguments, default_data_collator
from ..model_interface import Sample


class HFNERDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.stride = stride
        # build sliding-window examples
        self.examples: List[tuple[List[str], List[str]]] = []
        for sample in samples:
            tokens, labels = sample.tokens, sample.labels
            n = len(tokens)
            if self.stride and n > max_length:
                for start in range(0, n, self.stride):
                    end = min(start + max_length, n)
                    self.examples.append((tokens[start:end], labels[start:end]))
            else:
                self.examples.append((tokens, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, labels = self.examples[idx]
        # tokenize and pad to max_length
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        # create label ids, pad labels to -100
        label_ids = [-100] * self.max_length
        for i, word_idx in enumerate(word_ids or []):
            if word_idx is None or i >= self.max_length:
                continue
            label_ids[i] = self.label2id.get(
                labels[word_idx], self.label2id.get("O", 0)
            )
        # convert to torch tensors
        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
        return item


def train_transformer(
    model,
    tokenizer,
    samples: List[Sample],
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-4,
    max_length: int = 256,
    stride: int = 10,
):
    """
    Finetunes a HuggingFace token classification model.
    Prints training and evaluation loss.
    """
    # use existing label mapping in model.config
    if not hasattr(model.config, "label2id") or not model.config.label2id:
        raise ValueError("Model config missing label2id mapping; cannot align labels")
    label2id = model.config.label2id
    # ensure id2label exists
    if not hasattr(model.config, "id2label") or not model.config.id2label:
        model.config.id2label = {v: k for k, v in label2id.items()}

    # prepare training dataset with sliding windows
    train_dataset = HFNERDataset(samples, tokenizer, label2id, max_length, stride)

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        report_to=[],  # disable external logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    train_output = trainer.train()

    print(f"Train loss: {train_output.metrics.get('train_loss')}", flush=True)

    return train_output.metrics
