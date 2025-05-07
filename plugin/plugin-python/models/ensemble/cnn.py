from typing import List

from ..model_interface import BatchPredictions, Entity, Model, SentencePredictions
from .cnn_backend.backend import CNNModel
from ..utils import build_tag_vocab, clean_text


class CnnNerExtractor(Model):
    def __init__(self, model_path: str):
        self.model = CNNModel(
            model_path=model_path,
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            tag2idx=build_tag_vocab(),
        )

        self.batch_size = 100

    def _process_prediction(self, text: str, pred: List[str]) -> SentencePredictions:
        tokens = text.split()
        if len(tokens) != len(pred):
            raise ValueError(
                f"Token count ({len(tokens)}) and prediction count ({len(pred)}) differ."
            )
        entities: List[Entity] = []
        for tok, tag in zip(tokens, pred):
            idx = text.find(tok)
            if idx == -1:
                continue
            if tag == "O":
                continue
            entities.append(
                Entity(
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

        predictions = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_predictions = self.model.predict_batch(batch_texts)
            sentence_predictions = [
                self._process_prediction(text, pred)
                for text, pred in zip(batch_texts, batch_predictions)
            ]
            predictions.extend(sentence_predictions)

        return BatchPredictions(predictions=predictions)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]
