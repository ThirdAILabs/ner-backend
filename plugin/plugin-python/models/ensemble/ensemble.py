from collections import Counter
from typing import List

from ..model_interface import BatchPredictions, Entity, Model, SentencePredictions
from .cnn_backend.backend import CNNModel
from .pytorch_embedding_backend.backend import EmbeddingBagWrappedNerModel
from .udt_backend.backend import UDTModel
from ..utils import build_tag_vocab, clean_text


class EnsembleModel(Model):
    def __init__(self, load_config: dict):
        self.models = []
        for model_name, arguments in load_config.items():
            if model_name == "cnn":
                self.models.append(
                    CNNModel(
                        model_path=arguments["model_path"],
                        tokenizer_name="Qwen/Qwen2.5-0.5B",
                        tag2idx=build_tag_vocab(),
                    )
                )
            elif model_name == "embedding_bag":
                self.models.append(
                    EmbeddingBagWrappedNerModel(
                        checkpoint_path=arguments["checkpoint_path"],
                        tag2idx=build_tag_vocab(),
                    )
                )
            elif model_name == "udt":
                self.models.append(UDTModel(model_path=arguments["model_path"]))

            else:
                raise ValueError(f"Model {model_name} not found")

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]

    def _process_prediction(
        self, text: str, preds: List[List[str]]
    ) -> SentencePredictions:
        tokens = text.split()
        for i in range(len(preds)):
            if len(preds[i]) != len(tokens):
                raise ValueError(
                    f"Model Number {i} has different length of predictions. "
                    f"Initial length: {len(tokens)}, Current length: {len(preds[i])}"
                )

        tags = []
        L = len(preds[0])
        for i in range(L):
            choices = [preds[m][i] for m in range(len(self.models))]
            tag = Counter(choices).most_common(1)[0][0]
            tags.append(tag)

        offset = 0
        entities: List[Entity] = []
        for tok, tag in zip(tokens, tags):
            idx = text.find(tok, offset)
            if idx == -1:
                idx = offset
            offset = idx + len(tok)
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
        batched_predictions = [model.predict_batch(texts) for model in self.models]

        single_sample_predictions = []
        for index in range(len(batched_predictions[0])):
            # take the i-th prediction of each model
            single_sample_predictions.append(
                self._process_prediction(
                    texts[index], [pred[index] for pred in batched_predictions]
                )
            )

        return BatchPredictions(predictions=single_sample_predictions)
