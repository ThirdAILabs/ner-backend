from .cnn_backend.backend import CNNModel
from ..model_interface import Model, SentencePredictions, BatchPredictions, Entities
from .utils import build_tag_vocab, clean_text
from typing import List


class CnnNerExtractor(Model):
    def __init__(self, model_path: str):
        self.model = CNNModel(
            model_path=model_path,
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            tag2idx=build_tag_vocab(),
        )

    def _process_prediction(self, text: str, pred: List[str]) -> SentencePredictions:
        tokens = text.split()
        if len(tokens) != len(pred):
            raise ValueError(
                f"Token count ({len(tokens)}) and prediction count ({len(pred)}) differ."
            )
        entities: List[Entities] = []
        for tok, tag in zip(tokens, pred):
            idx = text.find(tok)
            if idx == -1:
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
        return SentencePredictions(entities=entities)

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        texts = [clean_text(text) for text in texts]
        preds = self.model.predict_batch(texts)

        sentence_predictions = [
            self._process_prediction(text, pred) for text, pred in zip(texts, preds)
        ]

        return BatchPredictions(predictions=sentence_predictions)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]
