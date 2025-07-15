import torch
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
import os
from transformers import AutoTokenizer
from typing import List, Any, Tuple, Optional
from ..model_interface import (
    BatchPredictions,
    Entity,
    Sample,
    Model,
    SentencePredictions,
)
import sys
from ..utils import clean_text_with_spans
import json
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def decrypt_model(enc_path: str, dec_path: str, key_b64: str):
    try:
        key = base64.b64decode(key_b64)
    except Exception as e:
        raise ValueError(f"Invalid MODEL_KEY: {e}")

    if len(key) != 32:
        raise ValueError("MODEL_KEY must be 32 bytes")

    try:
        with open(enc_path, "rb") as f:
            ct = f.read()
    except Exception as e:
        raise IOError(f"Error reading encrypted model: {e}")

    if len(ct) < 12:
        raise ValueError("Ciphertext too short")

    nonce, ciphertext = ct[:12], ct[12:]

    try:
        cipher = AESGCM(key)
        plaintext = cipher.decrypt(nonce=nonce, data=ciphertext, associated_data=None)
        with open(dec_path, "wb") as f:
            f.write(plaintext)
    except Exception as e:
        raise ValueError(f"AES-GCM decrypt failed: {e}")


def encrypt_model(dec_path: str, enc_path: str, key_b64: str):
    try:
        key = base64.b64decode(key_b64)
    except Exception as e:
        raise ValueError(f"Invalid MODEL_KEY: {e}")

    if len(key) != 32:
        raise ValueError("MODEL_KEY must be 32 bytes")

    try:
        with open(dec_path, "rb") as f:
            plaintext = f.read()
    except Exception as e:
        raise IOError(f"Error reading plaintext model: {e}")
    finally:
        os.remove(dec_path)

    try:
        nonce = os.urandom(12)
        cipher = AESGCM(key)
        ciphertext = cipher.encrypt(nonce=nonce, data=plaintext, associated_data=None)

        with open(enc_path, "wb") as f:
            f.write(nonce + ciphertext)
    except Exception as e:
        raise ValueError(f"AES-GCM encrypt failed: {e}")


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


class FinetuneDataset(Dataset):
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


IDX_TO_TAG = [
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
    "O",
    "PHONENUMBER",
    "SERVICE_CODE",
    "SEXUAL_ORIENTATION",
    "SSN",
    "URL",
    "VIN",
]

TAG_TO_IDX = {t: i for i, t in enumerate(IDX_TO_TAG)}


class CnnModel(Model):
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        tokenizer_path: str = None,
        key_b64: Optional[str] = None,
    ):
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        enc_model_path = os.path.join(model_path, "model.pt.enc")
        dec_model_path = os.path.join(model_path, "model.pt")
        if key_b64:
            decrypt_model(
                enc_path=enc_model_path, dec_path=dec_model_path, key_b64=key_b64
            )
        self.model = torch.jit.load(dec_model_path)
        self.model.eval()

        if key_b64:
            os.remove(dec_model_path)

        self.crf = CRF(num_tags=len(IDX_TO_TAG), batch_first=True)
        state_dict = torch.load(
            os.path.join(model_path, "crf.pth"),
            map_location=torch.device("cpu"),
        )
        self.crf.load_state_dict(state_dict)
        self.crf.eval()

        self.batch_size = 16

        self.key_b64 = key_b64

    def preprocess_text(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        mask = input_ids != self.tokenizer.pad_token_id

        return input_ids, offsets, mask

    def postprocess_tags(
        self,
        orig_texts: List[str],
        cleaned_texts: List[str],
        paths: List[List[int]],
        offsets_batch,
        spans_batch: List[List[Tuple[int, int]]],
    ) -> List[List[Entity]]:
        sub_tags_batch = [[IDX_TO_TAG[idx] for idx in path] for path in paths]

        results = []
        for orig_text, cleaned_text, offsets, sub_tags, spans in zip(
            orig_texts, cleaned_texts, offsets_batch, sub_tags_batch, spans_batch
        ):
            word_ids = manual_word_ids(cleaned_text, offsets)
            word_preds = ["O"] * len(orig_text.split())
            for wid, tag in zip(word_ids, sub_tags):
                if wid is None:
                    continue
                if word_preds[wid] == "O" and tag != "O":
                    word_preds[wid] = tag

            entities = [
                Entity(
                    text=orig_text[spans[i][0] : spans[i][1]],
                    label=tag,
                    score=1.0,
                    start=spans[i][0],
                    end=spans[i][1],
                )
                for i, tag in enumerate(word_preds)
                if tag != "O"
            ]

            results.append(SentencePredictions(entities=entities))

        return results

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        self.model.eval()
        self.crf.eval()

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            cleaned_and_spans = [clean_text_with_spans(t) for t in batch]
            cleaned_texts, spans_batch = zip(*cleaned_and_spans)

            input_ids, offsets_batch, mask = self.preprocess_text(list(cleaned_texts))
            emissions = self.model(input_ids)
            paths_batch = self.crf.decode(emissions, mask=mask)

            results.extend(
                self.postprocess_tags(
                    orig_texts=batch,
                    cleaned_texts=cleaned_texts,
                    paths=paths_batch,
                    offsets_batch=offsets_batch,
                    spans_batch=spans_batch,
                )
            )

        return BatchPredictions(predictions=results)

    def finetune(
        self,
        prompt: str,
        tags: List[Any],
        samples: List[Sample],
        lr: float = 3e-4,
        epochs: int = 5,
        batch_size: int = 16,
    ) -> None:
        default_tag = TAG_TO_IDX["O"]

        processed = []
        for sample in samples:
            text = " ".join(sample.tokens)
            enc = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            wids = manual_word_ids(text, offsets)
            lab_ids = [
                (
                    TAG_TO_IDX.get(sample.labels[w], default_tag)
                    if w is not None
                    else default_tag
                )
                for w in wids
            ]
            processed.append((ids, lab_ids))

        ds = FinetuneDataset(processed, self.tokenizer.pad_token_id)

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate
        )

        self.model.train()
        self.crf.train()

        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.crf.parameters()), lr=lr
        )

        for ep in range(1, epochs + 1):
            total = 0.0
            for toks, labels, mask in loader:
                opt.zero_grad()

                emissions = self.model(toks)
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
                loss.backward()

                opt.step()
                total += loss.item()
            print(
                f"Ep {ep}/{epochs}, loss={total/len(loader):.4f}",
                flush=True,
                file=sys.stderr,
            )

        self.model.eval()
        self.crf.eval()

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

        dec_torch_model_path = os.path.join(save_dir, "model.pt")
        enc_torch_model_path = os.path.join(save_dir, "model.pt.enc")
        torch.jit.save(self.model, dec_torch_model_path)
        if self.key_b64:
            encrypt_model(
                dec_path=dec_torch_model_path,
                enc_path=enc_torch_model_path,
                key_b64=self.key_b64,
            )

        torch.save(self.crf.state_dict(), os.path.join(save_dir, "crf.pth"))

        dummy_input = torch.zeros(1, 128, dtype=torch.long, device=torch.device("cpu"))

        dec_onnx_model_path = os.path.join(save_dir, "model.onnx")
        enc_onnx_model_path = os.path.join(save_dir, "model.onnx.enc")
        torch.onnx.export(
            self.model,
            dummy_input,
            dec_onnx_model_path,
            input_names=["input_ids"],
            output_names=["emissions"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "emissions": {0: "batch", 1: "seq"},
            },
            opset_version=13,
            do_constant_folding=True,
        )
        if self.key_b64:
            encrypt_model(
                dec_path=dec_onnx_model_path,
                enc_path=enc_onnx_model_path,
                key_b64=self.key_b64,
            )

        transitions = self.crf.transitions.detach().cpu().numpy().tolist()

        start_probs = self.crf.start_transitions.detach().cpu().numpy().tolist()
        end_probs = self.crf.end_transitions.detach().cpu().numpy().tolist()

        with open(
            os.path.join(save_dir, "transitions.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "Transitions": transitions,
                    "StartProbs": start_probs,
                    "EndProbs": end_probs,
                },
                f,
                indent=2,
            )
