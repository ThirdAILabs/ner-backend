import torch
import torch.nn as nn

from thirdai import data, dataset

HASH_DIMENSION = 50000

TAGS = [
    data.transformations.NERLearnedTag(name)
    for name in [
        "ADDRESS",
        "CARD_NUMBER",
        "COMPANY",
        "CREDIT_SCORE",
        "DATE",
        "EMAIL",
        "ETHNICITY",
        "GENDER",
        "ID_NUMBER",
        "LICENSE_PLATE",
        "LOCATION",
        "NAME",
        "PHONENUMBER",
        "SERVICE_CODE",
        "SEXUAL_ORIENTATION",
        "SSN",
        "URL",
        "VIN",
        "O",
    ]
]


def run_ner_inference(text: str):
    tag_tracker = data.transformations.NerTagTracker(tags=TAGS, ignored_tags=set())
    ner_tf = data.transformations.NerTokenizerUnigram(
        tokens_column="source",
        featurized_sentence_column="featurized_sentence",
        target_column=None,
        dyadic_num_intervals=3,
        target_word_tokenizers=[dataset.NaiveSplitTokenizer(" ")],
        feature_enhancement_config=data.transformations.NerFeatureConfig(
            True, True, True, True, True, True, True
        ),
        tag_tracker=tag_tracker,
    )
    text_tf = data.transformations.Text(
        input_column="featurized_sentence",
        output_indices="featurized_sentence",
        output_values=None,
        tokenizer=dataset.NaiveSplitTokenizer(" "),
        encoder=dataset.NGramEncoder(1),
        lowercase=True,
        dim=HASH_DIMENSION,
    )
    pipeline = data.transformations.Pipeline(transformations=[ner_tf, text_tf])
    cols = data.ColumnMap({"source": data.columns.StringArrayColumn([text])})
    return pipeline(cols)["featurized_sentence"]


class EmbeddingBagNERModel(nn.Module):
    def __init__(self, embedding_bag, tag2idx, pad_token_idx=HASH_DIMENSION):
        super().__init__()
        if embedding_bag is None:
            self.embedding_bag = nn.EmbeddingBag(
                num_embeddings=HASH_DIMENSION + 1,
                embedding_dim=512,
                padding_idx=pad_token_idx,
                sparse=True,
            )
            self.embedding_bias = nn.Parameter(torch.zeros(512))
            nn.init.normal_(self.embedding_bag.weight, 0.0, 0.01)
            with torch.no_grad():
                self.embedding_bag.weight[pad_token_idx].zero_()
        else:
            self.embedding_bag = embedding_bag
        D = self.embedding_bag.embedding_dim
        self.fc = nn.Linear(D, len(tag2idx))
        nn.init.normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.activation = nn.LeakyReLU()
        self.idx_to_tag = {i: t for t, i in tag2idx.items()}
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, batch_sequences, lengths):
        device = next(self.fc.parameters()).device
        B, T = len(batch_sequences), max(lengths)
        L = max(len(tok) for seq in batch_sequences for tok in seq)
        padded = torch.full((B, T, L), HASH_DIMENSION, dtype=torch.long, device=device)
        for i, seq in enumerate(batch_sequences):
            for j, tok in enumerate(seq):
                padded[i, j, : len(tok)] = torch.tensor(tok, device=device)
        flat = padded.view(B * T, L)
        emb = self.embedding_bag(flat) + getattr(self, "embedding_bias", 0)
        emb = emb.view(B, T, -1)
        return self.fc(self.activation(emb))

    def predict_sequence(self, seqs, lengths):
        logits = self.forward(seqs, lengths)
        preds = logits.argmax(dim=-1)
        all_tags = []
        for i, L in enumerate(lengths):
            all_tags.append([self.idx_to_tag[int(idx)] for idx in preds[i, :L]])
        return all_tags
