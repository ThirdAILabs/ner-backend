from transformers import AutoTokenizer
import torch
import torch.nn as nn
from .impl import CNNNERModelSentenceTokenized


class CNNModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        tag2idx: dict,
        embedding_path: str = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tag2idx = tag2idx

        if embedding_path:
            pretrained_emb = nn.Embedding.from_pretrained(torch.load(embedding_path))
        else:
            pretrained_emb = None

        self.model = CNNNERModelSentenceTokenized(
            tokenizer=self.tokenizer,
            tag2idx=self.tag2idx,
            pretrained_embedding=pretrained_emb,
        )

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str):
        _, pred_tags = self.model.predict(text)
        return [tag for tag in pred_tags]
