from typing import List

import torch
from transformers import AutoTokenizer

from .impl import CNNNERModelSentenceTokenized


class CNNModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        tag2idx: dict,
    ):
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
