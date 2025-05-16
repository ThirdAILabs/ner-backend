from typing import List
import os

from ..model_interface import BatchPredictions, Entity, Model, SentencePredictions
from .cnn_backend.backend import CNNModel
from ..utils import build_tag_vocab, clean_text_with_spans


class CnnNerExtractor(Model):
    def __init__(self, model_path: str):
        self.model = CNNModel(
            model_path=model_path,
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            tag2idx=build_tag_vocab(),
        )

        self.batch_size = 100

    def _process_prediction(
        self,
        original_text,
        spans,
        tags,
    ) -> SentencePredictions:
        if len(spans) != len(tags):
            raise ValueError(f"{len(spans)} spans vs {len(tags)} tags")
        entities = []
        for (start, end), tag in zip(spans, tags):
            if tag == "O":
                continue

            entities.append(
                Entity(
                    text=original_text[start:end],
                    label=tag,
                    score=1.0,
                    start=start,
                    end=end,
                )
            )
        return SentencePredictions(entities=entities)

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            cleaned_and_spans = [clean_text_with_spans(t) for t in batch]
            cleaned_texts, spans_list = zip(*cleaned_and_spans)

            batch_tags = self.model.predict_batch(list(cleaned_texts))
            for orig, spans, tags in zip(batch, spans_list, batch_tags):
                results.append(self._process_prediction(orig, spans, tags))

        return BatchPredictions(predictions=results)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]

    def finetune(
        self,
        prompt,
        tags,
        samples,
    ):
        raw_samples = []
        for sample in samples:
            tokens = sample.tokens
            labels = sample.labels
            raw_samples.append((tokens, labels))

        self.model.finetune(
            raw_samples,
            epochs=1,
            lr=3e-4,
            batch_size=16,
        )

        return True

    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        self.model.save(dir)
