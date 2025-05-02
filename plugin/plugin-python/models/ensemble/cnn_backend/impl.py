import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF


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
            self.embedding = nn.Embedding(
                tokenizer.vocab_size, 100, padding_idx=tokenizer.pad_token_id
            )
            self.embed_dim = 100

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

        # Build two separate lists for standard and dilated convolution blocks.
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

        # Linear layer for projecting to the tag space.
        num_tags = len(tag2idx)
        self.hidden2tag = nn.Linear(conv_channels, num_tags)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

        # CRF layer for sequence tagging.
        self.crf = CRF(num_tags, batch_first=True)

        # Inverse mapping of tags.
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

            # Branch 1: standard convolution.
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

            # Branch 2: dilated convolution.
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

    def predict(self, text: str):
        txt = text.lower() if getattr(self.tokenizer, "do_lower_case", False) else text
        enc = self.tokenizer(txt, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0)
        offsets = enc["offset_mapping"]

        word_ids = manual_word_ids(txt, offsets)
        mask = input_ids != self.tokenizer.pad_token_id
        emissions = self.forward(input_ids)
        paths = self.crf.decode(emissions, mask=mask)[0]
        sub_tags = [self.idx_to_tag[idx] for idx in paths]

        word_preds = []
        max_word = max(w for w in word_ids if w is not None)
        for w in range(max_word + 1):
            toks = [sub_tags[i] for i, wid in enumerate(word_ids) if wid == w]
            chosen = next((t for t in toks if t != "O"), "O")
            word_preds.append(chosen)

        words = txt.split()
        assert len(words) == len(
            word_preds
        ), f"Mismatch {len(words)} words vs {len(word_preds)} preds"
        return words, word_preds
