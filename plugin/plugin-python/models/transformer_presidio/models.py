import contextlib
import io
from typing import Any, Dict, List

from transformers import AutoModelForTokenClassification, AutoTokenizer
from presidio_analyzer import RecognizerResult

from ..model_interface import Entities, Model, SentencePredictions, BatchPredictions
from .make_analyzer import get_analyzer, analyze_text_batch
from .transformer_inference import predict_batch, punctuation_filter
from ..utils import clean_text


def suppress_output():

    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    )


def merge_predictions(
    hf_preds: SentencePredictions, pres_preds: SentencePredictions, full_text: str
) -> SentencePredictions:
    """
    Merge two Predictions objects (HF + Presidio) according to:
      1. disjoint spans → keep both
      2. overlapping + same label → expand to cover full span
      3. overlapping + different label → pick Presidio
    """
    # annotate source
    all_items: List[Dict[str, Any]] = []
    for e in hf_preds.entities:
        all_items.append({"ent": e, "src": "hf"})
    for e in pres_preds.entities:
        all_items.append({"ent": e, "src": "presidio"})

    # sort by start offset
    all_items.sort(key=lambda x: x["ent"].start)

    # cluster by overlap connectivity
    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = []
    cluster_end = -1
    for item in all_items:
        s, e = item["ent"].start, item["ent"].end
        if current_cluster and s <= cluster_end:
            current_cluster.append(item)
            cluster_end = max(cluster_end, e)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [item]
            cluster_end = e
    if current_cluster:
        clusters.append(current_cluster)

    merged: List[Entities] = []
    # process each cluster
    for cluster in clusters:
        pres = [x["ent"] for x in cluster if x["src"] == "presidio"]
        hf = [x["ent"] for x in cluster if x["src"] == "hf"]

        if pres:
            # rule: prefer Presidio
            pres_labels = {e.label for e in pres}
            if len(pres_labels) == 1:
                # same tag → expand to cover both HF + Presidio
                label = pres[0].label
                starts = [e.start for e in pres + hf]
                ends = [e.end for e in pres + hf]
                st, ed = min(starts), max(ends)
                merged.append(
                    Entities(
                        text=full_text[st:ed],
                        label=label,
                        # you could max or avg the scores
                        score=max(e.score for e in pres + hf),
                        start=st,
                        end=ed,
                    )
                )
            else:
                # conflicting Presidio tags → pick one by longest span (or highest score)
                choice = max(pres, key=lambda e: (e.end - e.start, e.score))
                merged.append(choice)
        else:
            # HF‐only cluster (no Presidio overlap) → keep all HF
            merged.extend(hf)

    return SentencePredictions(entities=merged)


class HuggingFaceModel(Model):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.aggregation_strategy = "max_activation"
        self.threshold = 0.8
        self.filter_funcs = [punctuation_filter]
        self.max_length = 256
        self.stride = 8
        self.top_k = 3

        self.skipped_entites = set({"O", "CARD_NUMBER"})

    def _process_prediction(
        self, text: str, entities: List[str]
    ) -> SentencePredictions:
        tokens = text.split()
        if len(entities) != len(tokens):
            raise ValueError("Model has different length of predictions.")
        predictions = []
        for entity in entities:
            w, label, conf, span = entity
            if label in self.skipped_entites:
                continue
            predictions.append(
                Entities(
                    text=w,
                    label=label,
                    score=conf,
                    start=span[0],
                    end=span[1],
                )
            )
        return SentencePredictions(entities=predictions)

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        texts = [clean_text(text) for text in texts]
        batch_entities = predict_batch(
            texts,
            self.model,
            self.tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            threshold=self.threshold,
            filter_funcs=self.filter_funcs,
            max_length=self.max_length,
            stride=self.stride,
            top_k=self.top_k,
        )

        sentence_predictions = [
            self._process_prediction(text, entities)
            for text, entities in zip(texts, batch_entities)
        ]

        return BatchPredictions(predictions=sentence_predictions)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]


class PresidioWrappedNerModel(Model):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.analyzer = get_analyzer()

    def _process_prediction(
        self, text: str, entities: List[RecognizerResult]
    ) -> SentencePredictions:
        predictions = []
        for entity in entities:
            predictions.append(
                Entities(
                    text=text[entity.start : entity.end],
                    label=entity.entity_type,
                    score=entity.score,
                    start=entity.start,
                    end=entity.end,
                )
            )
        return SentencePredictions(entities=predictions)

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        batch_entities = analyze_text_batch(texts, self.analyzer, self.threshold)
        sentence_predictions = [
            self._process_prediction(text, entities)
            for text, entities in zip(texts, batch_entities)
        ]
        return BatchPredictions(predictions=sentence_predictions)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]


class CombinedNERModel(Model):
    """
    Wraps an HF model and a Presidio model and merges their outputs.
    """

    def __init__(self, model_path, threshold):
        self.threshold = threshold
        with suppress_output()[0], suppress_output()[1]:
            self.hf = HuggingFaceModel(model_path)
            self.pres = PresidioWrappedNerModel(threshold)

    def _process_predictions(self, text, hf_out, pres_out):
        merged = merge_predictions(hf_out, pres_out, text)
        return merged

    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        hf_out = self.hf.predict_batch(texts)
        pres_out = self.pres.predict_batch(texts)
        # merge
        sentence_predictions = [
            self._process_predictions(text, hf_out, pres_out)
            for text, hf_out, pres_out in zip(texts, hf_out, pres_out)
        ]
        return BatchPredictions(predictions=sentence_predictions)

    def predict(self, text: str) -> SentencePredictions:
        return self.predict_batch([text]).predictions[0]
