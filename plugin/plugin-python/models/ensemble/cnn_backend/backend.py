from typing import List

import torch
import os
from transformers import AutoTokenizer

from .impl import CNNNERModelSentenceTokenized


class CNNModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        tag2idx: dict,
        tokenizer_path: str = None,
    ):
        # Try loading from local path first if provided, otherwise use pretrained
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tag2idx = tag2idx

        self.model = CNNNERModelSentenceTokenized(
            tokenizer=self.tokenizer,
            tag2idx=self.tag2idx,
        )

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_batch(self, texts: List[str]):
        predictions = self.model.predict_batch(texts)
        return [tags for _, tags in predictions]

    def predict(self, text: str):
        predictions = self.model.predict_batch([text])
        return predictions[0]

    def finetune(
        self,
        raw_samples,
        epochs: int = 1,
        lr: float = 3e-4,
        batch_size: int = 16,
    ):
        self.model.finetune(
            raw_samples,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

    def save(self, dir: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(dir, "cnn_model.pth"))
