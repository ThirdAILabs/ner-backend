from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader

import sys


def manual_word_ids(text: str, offsets: list[tuple[int, int]]) -> list[int | None]:
    word_ids = []
    current_word = -1
    last_end = -1
    for start, end in offsets:
        if start == end == 0:
            word_ids.append(None)
        else:
            if (start == 0 or text[start].isspace()) and start >= last_end:
                current_word += 1
            word_ids.append(current_word)
        last_end = end
    return word_ids


def aggregate_predictions(pred_tags, subword_lens):
    aggregated_pred = []
    pointer = 0
    for length in subword_lens:
        best_tag = None
        for j in range(length):
            if pred_tags[pointer + j] != "O":
                best_tag = pred_tags[pointer + j]
                break
            else:
                best_tag = "O"
        aggregated_pred.append(best_tag)
        pointer += length
    return aggregated_pred


class _FinetuneDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[int], List[int]]], pad_token_id: int):
        self.samples = samples
        self.pad_id = pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, batch):
        tokens_batch, labels_batch = zip(*batch)
        max_len = max(len(t) for t in tokens_batch)
        toks = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labs = torch.zeros((len(batch), max_len), dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, (toks_i, labs_i) in enumerate(zip(tokens_batch, labels_batch)):
            L = len(toks_i)
            toks[i, :L] = torch.tensor(toks_i, dtype=torch.long)
            labs[i, :L] = torch.tensor(labs_i, dtype=torch.long)
            mask[i, :L] = 1
        return toks, labs, mask


class CNNNERModelSentenceTokenized(nn.Module):
    def __init__(
        self,
        tokenizer,
        tag2idx,
        pretrained_embedding=None,
        conv_channels=128,
        num_blocks=3,
        kernel_size=3,
        dropout=0.1,
        freeze_embedding=False,
    ):
        super(CNNNERModelSentenceTokenized, self).__init__()
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
            self.embed_dim = self.embedding.weight.shape[1]
        else:
            # hardcoding the embedding dimensions
            self.embedding = nn.Embedding(151936, 896)
            self.embed_dim = 896

        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        self.conv_channels = conv_channels
        if self.embed_dim != conv_channels:
            self.proj = nn.Conv1d(self.embed_dim, conv_channels, kernel_size=1)
        else:
            self.proj = None

        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.num_blocks = num_blocks

        self.branch1_convs = nn.ModuleList()
        self.branch2_convs = nn.ModuleList()
        self.batch_norms_1 = nn.ModuleList()
        self.batch_norms_2 = nn.ModuleList()

        for b in range(num_blocks):
            dil = 2**b
            for _ in range(2):
                self.branch1_convs.append(
                    nn.Conv1d(
                        conv_channels,
                        conv_channels,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                        dilation=1,
                    )
                )
                self.branch2_convs.append(
                    nn.Conv1d(
                        conv_channels,
                        conv_channels,
                        kernel_size,
                        padding=(kernel_size - 1) * dil // 2,
                        dilation=dil,
                    )
                )
                self.batch_norms_1.append(nn.BatchNorm1d(conv_channels))
                self.batch_norms_2.append(nn.BatchNorm1d(conv_channels))

        num_tags = len(tag2idx)
        self.hidden2tag = nn.Linear(conv_channels, num_tags)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

        self.crf = CRF(num_tags, batch_first=True)

        self.idx_to_tag = {idx: tag for tag, idx in tag2idx.items()}

    def forward(self, tokens):
        embeds = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        x = embeds.transpose(1, 2)  # (batch, embed_dim, seq_len)
        if self.proj is not None:
            x = self.proj(x)

        out_branch1 = x
        out_branch2 = x

        for b in range(self.num_blocks):
            i1 = 2 * b
            i2 = 2 * b + 1

            y1 = self.branch1_convs[i1](out_branch1)
            y1 = self.batch_norms_1[i1](y1)
            y1 = F.leaky_relu(y1)
            if self.dropout_rate and self.training:
                y1 = F.dropout(y1, p=self.dropout_rate)
            y1 = self.branch1_convs[i2](y1)
            y1 = self.batch_norms_1[i2](y1)
            y1 = F.leaky_relu(y1)
            if self.dropout_rate and self.training:
                y1 = F.dropout(y1, p=self.dropout_rate)
            out_branch1 = x + y1

            z1 = self.branch2_convs[i1](out_branch2)
            z1 = self.batch_norms_2[i1](z1)
            z1 = F.leaky_relu(z1)
            if self.dropout_rate and self.training:
                z1 = F.dropout(z1, p=self.dropout_rate)
            z1 = self.branch2_convs[i2](z1)
            z1 = self.batch_norms_2[i2](z1)
            z1 = F.leaky_relu(z1)
            if self.dropout_rate and self.training:
                z1 = F.dropout(z1, p=self.dropout_rate)
            out_branch2 = x + z1

        out = out_branch1 + out_branch2  # (batch, conv_channels, seq_len)
        out = out.transpose(1, 2)  # (batch, seq_len, conv_channels)
        emissions = self.hidden2tag(out)
        return emissions

    def compute_loss(self, tokens, labels):
        mask = tokens != self.tokenizer.pad_token_id
        ll = self.crf(self(tokens), labels, mask=mask, reduction="sum")
        return -ll / tokens.size(0)

    def predict_batch(self, texts: List[str]):
        """
        Tokenize & score `texts` in mini‐batches of size `batch_size`.
        Returns a flat list of (words, word_preds) for each input text.
        """
        results = []
        batch = texts
        # lowercase if needed
        txts = [
            t.lower() if getattr(self.tokenizer, "do_lower_case", False) else t
            for t in batch
        ]
        # batch‐tokenize with padding
        enc = self.tokenizer(
            txts,
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(next(self.parameters()).device)
        offsets_batch = enc["offset_mapping"]
        mask = input_ids != self.tokenizer.pad_token_id

        # forward & CRF‐decode entire mini‐batch
        emissions = self.forward(input_ids)  # (B, L, C)
        paths_batch = self.crf.decode(emissions, mask=mask)  # List[B × L]

        # map to tag‐strings
        sub_tags_batch = [
            [self.idx_to_tag[idx] for idx in path] for path in paths_batch
        ]

        # for each sample in the mini‐batch, re‐aggregate to per‐word
        for orig_text, offsets, sub_tags in zip(batch, offsets_batch, sub_tags_batch):
            word_ids = manual_word_ids(orig_text, offsets)
            words = orig_text.split()
            word_preds = ["O"] * len(words)
            for wid, tag in zip(word_ids, sub_tags):
                if wid is None:
                    continue
                if word_preds[wid] == "O" and tag != "O":
                    word_preds[wid] = tag
            results.append((words, word_preds))

        print("Results:", results, flush=True, file=sys.stderr)
        return results

    def finetune(
        self,
        raw_samples: List[Tuple[List[str], List[str]]],
        epochs: int = 5,
        lr: float = 3e-4,
        batch_size: int = 16,
    ) -> None:
        processed = []
        for tokens, tags in raw_samples:
            text = " ".join(tokens)
            enc = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            wids = manual_word_ids(text, offsets)
            lab_ids = [
                self.tag2idx.get(tags[w], 0) if w is not None else 0 for w in wids
            ]
            processed.append((ids, lab_ids))

        ds = _FinetuneDataset(processed, self.tokenizer.pad_token_id)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for toks, labs, mask in loader:
                opt.zero_grad()
                loss = self.compute_loss(toks, labs)
                loss.backward()
                opt.step()
                total += loss.item()
            print(
                f"Ep {ep}/{epochs}, loss={total/len(loader):.4f}",
                flush=True,
                file=sys.stderr,
            )
