from collections import Counter
from .cnn_backend.backend import CNNModel
from .pytorch_embedding_backend.backend import EmbeddingBagWrappedNerModel
from .udt_backend.backend import UDTModel

from ..model_interface import Model, Predictions, Entities
from .utils import build_tag_vocab, clean_text
from typing import List
from time import time


class EnsembleModel(Model):
    def __init__(self, load_config: dict):
        self.models = []
        for model_name, arguments in load_config.items():
            if model_name == "cnn":
                self.models.append(
                    CNNModel(
                        model_path=arguments["model_path"],
                        tokenizer_name=arguments["tokenizer_name"],
                        tag2idx=build_tag_vocab(),
                        embedding_path=arguments["embedding_path"],
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

    def predict(self, text: str) -> Predictions:
        start = time()
        text = clean_text(text)
        tokens = text.split()
        preds = [model.predict(text) for model in self.models]

        initial_len = len(tokens)
        for i, pred in enumerate(preds):
            if len(pred) != initial_len:
                raise ValueError(
                    f"Model Number {i} has different length of predictions. "
                    f"Initial length: {initial_len}, Current length: {len(pred)}"
                )

        elapsed_ms = round((time() - start) * 1000, 2)

        # perform majority voting
        L = len(preds[0])
        tags = []
        for i in range(L):
            choices = [preds[m][i] for m in range(len(self.models))]
            tag = Counter(choices).most_common(1)[0][0]
            tags.append(tag)

        # generate offsets and entities
        offset = 0
        entities: List[Entities] = []
        for tok, tag in zip(tokens, tags):
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
