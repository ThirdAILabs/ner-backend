import string
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import torch
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer


def aggregate_max(predictions: List[Tuple], pred_index: int, act_index: int):
    pred_dict = defaultdict(float)
    for pred in predictions:
        pred_dict[pred[pred_index]] += pred[act_index]

    max_pred = max(pred_dict, key=pred_dict.get)
    return max_pred, pred_dict[max_pred] / len(predictions)


def aggregate_first(predictions: List[Tuple], pred_index: int, act_index: int):
    return predictions[0][pred_index], predictions[0][act_index]


def aggregate_average(predictions: List[Tuple], pred_index: int, act_index: int):
    pred_dict = defaultdict(float)
    for pred in predictions:
        pred_dict[pred[pred_index]] += pred[act_index]

    max_pred = max(pred_dict, key=pred_dict.get)
    return max_pred, pred_dict[max_pred] / len(predictions)


def aggregate_majority(predictions: List[Tuple], pred_index: int, act_index: int):
    pred_dict = defaultdict(int)
    for pred in predictions:
        pred_dict[pred[pred_index]] += 1

    max_pred = max(pred_dict, key=pred_dict.get)
    pred_value = pred_dict[max_pred]

    aggregated_max = defaultdict(float)
    for pred in predictions:
        if pred_dict[pred[pred_index]] >= pred_value:
            aggregated_max[pred[pred_index]] += pred[act_index]

    max_pred = max(aggregated_max, key=aggregated_max.get)
    return max_pred, aggregated_max[max_pred] / len(predictions)


def aggregate_first_non_o(predictions: List[Tuple], pred_index: int, act_index: int):
    for pred in predictions:
        if pred[pred_index] != 0:
            return pred[pred_index], pred[act_index]
    return 0, 0.1


def punctuation_filter(token_ids: List[int], tokenizer: AutoTokenizer):
    include_tokens = []
    for token_id in token_ids:
        token_string = tokenizer._convert_id_to_token(token_id)
        if all(char in string.punctuation for char in token_string):
            include_tokens.append(False)
        else:
            include_tokens.append(True)
    return include_tokens


class TokenPrediction(BaseModel):
    token_id: int
    preds: list[int]
    activations: list[float]

    def find_max(self, threshold):
        if (
            self.preds[0] == 0
            and self.activations[0] < threshold
            and len(self.preds) > 1
        ):
            return self.preds[1], (self.activations[1] + 1e-6) / (
                sum(self.activations[1:]) + 1e-6
            )
        return self.preds[0], self.activations[0]


class WordPrediction(BaseModel):
    word_id: int
    tokens_predictions: list[TokenPrediction] = []

    @property
    def token_ids(self):
        return [
            token_prediction.token_id for token_prediction in self.tokens_predictions
        ]

    def add(self, token_prediction: TokenPrediction):
        self.tokens_predictions.append(token_prediction)

    def _get_tokenwise_predictions(self, threshold):
        tokenwise_predictions = []
        for token_prediction in self.tokens_predictions:
            pred, act = token_prediction.find_max(threshold)
            tokenwise_predictions.append((pred, act))
        return tokenwise_predictions

    def aggregate(
        self,
        strategy: str,
        threshold: float,
        tokenizer: AutoTokenizer,
        filter_funcs: List[Callable],
    ):
        tokenwise_predictions = self._get_tokenwise_predictions(threshold)
        include_token = [True for _ in range(len(tokenwise_predictions))]
        for filter_func in filter_funcs:
            diff_filter = filter_func(self.token_ids, tokenizer)
            include_token = [
                include_token[i] and diff_filter[i] for i in range(len(include_token))
            ]

        filtered_tokenwise_predictions = []
        for i in range(len(tokenwise_predictions)):
            if include_token[i]:
                filtered_tokenwise_predictions.append(tokenwise_predictions[i])

        aggregate_func = None
        if strategy == "max_activation":
            aggregate_func = aggregate_max
        elif strategy == "first":
            aggregate_func = aggregate_first
        elif strategy == "average":
            aggregate_func = aggregate_average
        elif strategy == "majority":
            aggregate_func = aggregate_majority
        elif strategy == "first_non_o":
            aggregate_func = aggregate_first_non_o

        if aggregate_func is None:
            raise ValueError(f"Invalid strategy: {strategy}")

        if not filtered_tokenwise_predictions:
            return 0, 0.1

        # in token prediction, label is index 0, activation is index 1
        pred_index = 0
        act_index = 1
        label, confidence = aggregate_func(
            filtered_tokenwise_predictions,
            pred_index,
            act_index,
        )
        return label, confidence


def normal_predictions(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits


def convert_encoded_to_words(offset_mapping, tokenizer, tokens):
    mapping = {}
    unique_tokens = 0

    end_offset = -1
    cls_id, sep_id, pad_id = (
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    )
    for i, offset in enumerate(offset_mapping):
        if tokens[i] in {cls_id, sep_id, pad_id}:
            continue
        start = offset[0]
        end = offset[1]
        if start > end_offset:
            # if start is greater than end_offset, then we have a new word
            mapping[i] = unique_tokens
            unique_tokens += 1
        elif start == end_offset:
            # if start is equal to end_offset and the token start is not a space, the the word is the same
            # else we have the same word
            if tokenizer._convert_id_to_token(tokens[i])[0] != "Ġ":
                mapping[i] = unique_tokens - 1
            else:
                mapping[i] = unique_tokens
                unique_tokens += 1
        end_offset = end

    # this stores the span of the word across subtokens
    spans = defaultdict(list)

    for subtoken_index, word_index in mapping.items():
        spans[word_index].append(offset_mapping[subtoken_index][0])
        spans[word_index].append(offset_mapping[subtoken_index][1])

    # combine word spans
    combined_word_spans = {}
    for word_index, all_positions in spans.items():
        combined_word_spans[word_index] = (min(all_positions), max(all_positions))

    return mapping, combined_word_spans


def process_span(span: Tuple[int, int], text: str):
    start = span[0]
    end = span[1] - 1

    while text[start].isspace():
        start += 1
    while text[end].isspace():
        end -= 1

    return start, end + 1


def _merge_single_text_chunks(
    input_ids_list: List[List[int]],
    offset_mappings: List[List[Tuple[int, int]]],
    top_k_ids: List[List[List[int]]],
    top_k_acts: List[List[List[float]]],
) -> Tuple[
    List[int],  # flat token ids
    List[Tuple[int, int]],  # flat offset mapping
    List[List[int]],  # flat top-k predictions
    List[List[float]],  # flat top-k activations
]:
    """
    Merge overlapping chunks (single sample):
     - drop special tokens (span == (0,0))
     - drop duplicates in overlapped regions (keep first)
    """
    seen_spans = set()
    flat_ids, flat_offs, flat_preds, flat_acts = [], [], [], []

    for ids, offs, preds, acts in zip(
        input_ids_list, offset_mappings, top_k_ids, top_k_acts
    ):
        for tok, span, p, a in zip(ids, offs, preds, acts):
            # 1) drop special tokens
            if span[0] == span[1] == 0:
                continue
            key = (span[0], span[1])
            # 2) drop duplicates
            if key in seen_spans:
                continue
            seen_spans.add(key)

            flat_ids.append(tok)
            flat_offs.append(key)
            flat_preds.append(p)
            flat_acts.append(a)

    return flat_ids, flat_offs, flat_preds, flat_acts


def transform_predictions(
    tokens: List[int],
    offset_mapping: List[Tuple[int, int]],
    predictions: List[List[int]],
    activations: List[List[float]],
    tokenizer: AutoTokenizer,
    aggregation_strategy="max_activation",
    threshold=0.5,
    filter_funcs: List[Callable] = [],
):
    mapping, word_spans = convert_encoded_to_words(offset_mapping, tokenizer, tokens)

    word_predictions: Dict[int, WordPrediction] = {}
    for idx in set(mapping.values()):
        word_predictions[idx] = WordPrediction(word_id=idx)

    for index, (token, preds, acts) in enumerate(zip(tokens, predictions, activations)):
        if index not in mapping:
            continue
        word_id = mapping[index]
        word_predictions[word_id].add(
            TokenPrediction(token_id=token, preds=preds, activations=acts)
        )

    aggregated_results = []
    sorted_word_ids = sorted(word_predictions.keys())

    for word_id in sorted_word_ids:
        word_prediction = word_predictions[word_id]
        label, confidence = word_prediction.aggregate(
            strategy=aggregation_strategy,
            threshold=threshold,
            tokenizer=tokenizer,
            filter_funcs=filter_funcs,
        )
        aggregated_results.append((label, confidence))

    return aggregated_results, word_spans


def predict_batch(
    texts: List[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    aggregation_strategy: str = "max_activation",
    threshold: float = 0.5,
    filter_funcs: List[Callable] = [],
    max_length: int = 256,
    stride: int = 4,
    top_k: int = 3,
) -> List[List[Tuple[str, str, float, Tuple[int, int]]]]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offset_mappings = encoded.pop("offset_mapping").tolist()
    sample_mapping = encoded.pop("overflow_to_sample_mapping").tolist()
    input_ids_list = encoded["input_ids"].cpu().tolist()
    inputs = {k: v.to(model.device) for k, v in encoded.items()}

    # Forward pass once for all chunks
    logits = normal_predictions(model, inputs)
    probs = torch.softmax(logits, dim=-1)
    topk_acts, topk_ids = torch.topk(probs, k=top_k, dim=-1)
    topk_ids = topk_ids.cpu().tolist()
    topk_acts = topk_acts.cpu().tolist()

    # Group chunks by original text sample
    num_texts = len(texts)
    chunks_per_sample: Dict[int, List[int]] = {i: [] for i in range(num_texts)}
    for chunk_idx, sample_idx in enumerate(sample_mapping):
        chunks_per_sample[sample_idx].append(chunk_idx)

    batch_results: List[List[Tuple[str, str, float, Tuple[int, int]]]] = []
    # Process each text sample
    for i, text in enumerate(texts):
        chunk_indices = chunks_per_sample.get(i, [])
        ids_chunks = [input_ids_list[j] for j in chunk_indices]
        offs_chunks = [offset_mappings[j] for j in chunk_indices]
        preds_chunks = [topk_ids[j] for j in chunk_indices]
        acts_chunks = [topk_acts[j] for j in chunk_indices]

        # Merge overlapping chunks for this sample
        flat_ids, flat_offs, flat_preds, flat_acts = _merge_single_text_chunks(
            ids_chunks, offs_chunks, preds_chunks, acts_chunks
        )
        # Aggregate token predictions into word-level predictions
        agg, word_spans = transform_predictions(
            tokens=flat_ids,
            offset_mapping=flat_offs,
            predictions=flat_preds,
            activations=flat_acts,
            tokenizer=tokenizer,
            aggregation_strategy=aggregation_strategy,
            threshold=threshold,
            filter_funcs=filter_funcs,
        )

        words = text.split()
        assert len(words) == len(agg), f"Got {len(words)} words vs {len(agg)} preds"
        sample_results: List[Tuple[str, str, float, Tuple[int, int]]] = []
        for idx, (label_id, conf) in enumerate(agg):
            w = words[idx]
            span = word_spans[idx]
            start, end = process_span(span, text)
            sample_results.append(
                (w, model.config.id2label[label_id], conf, (start, end))
            )
        batch_results.append(sample_results)

    return batch_results
