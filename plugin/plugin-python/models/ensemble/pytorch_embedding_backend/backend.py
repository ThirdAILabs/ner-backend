from .impl import EmbeddingBagNERModel, HASH_DIMENSION, run_ner_inference
import torch
from typing import List


class EmbeddingBagWrappedNerModel:
    def __init__(
        self,
        checkpoint_path: str,
        tag2idx: dict,
        pad_token_idx: int = HASH_DIMENSION,
    ):
        self.tag2idx = tag2idx
        self.model = EmbeddingBagNERModel(
            embedding_bag=None,
            tag2idx=tag2idx,
            pad_token_idx=pad_token_idx,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text: str):

        features = run_ner_inference(text.split())
        seqs = [[[int(tok[i]) for i in range(len(tok))] for tok in features]]
        lengths = [len(seq) for seq in seqs]

        preds = self.model.predict_sequence(seqs, lengths)
        return preds[0]

    def predict_batch(self, texts: List[str]):
        return [self.predict(text) for text in texts]
