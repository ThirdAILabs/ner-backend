from time import time
from .cnn_backend.backend import CNNModel
from ..model_interface import Model, Predictions, Entities
from .utils import build_tag_vocab, clean_text
from typing import List


class CnnNerExtractor(Model):
    def __init__(self, model_path: str):
        self.model = CNNModel(
            model_path=model_path,
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            tag2idx=build_tag_vocab(),
        )

    def predict(self, text: str) -> Predictions:
        start = time()
        text = clean_text(text)
        tokens = text.split()
        preds = self.model.predict(text)

        if len(preds) != len(tokens):
            raise ValueError(
                f"Token count ({len(tokens)}) and prediction count ({len(preds)}) differ."
            )

        elapsed_ms = round((time() - start) * 1000, 2)

        # build entity list with character offsets
        offset = 0
        entities: List[Entities] = []
        for tok, tag in zip(tokens, preds):
            idx = text.find(tok, offset)
            if idx == -1:
                idx = offset
            offset = idx + len(tok)

            if tag == "O":
                continue

            entities.append(
                Entities(
                    text=tok,
                    label=tag,
                    score=1.0,
                    start=idx,
                    end=idx + len(tok),
                )
            )

        return Predictions(entities=entities, elapsed_ms=elapsed_ms)
