from typing import List

import torch
import os
from transformers import AutoTokenizer

from .impl import CNNNERModelSentenceTokenized
from .convert_model_to_onnx import export

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

    def save(self, dir: str, export_onnx: bool = False) -> None:
        cnn_model_path = os.path.join(dir, "cnn_model.pth")
        torch.save(self.model.state_dict(), cnn_model_path)
        if export_onnx:
            onnx_dir = os.path.join(dir, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            export(
                self.model,
                self.tag2idx,
                onnx_dir,
            )

