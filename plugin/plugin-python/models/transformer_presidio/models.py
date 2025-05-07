import contextlib
import io
import time
from typing import Any, Dict, List

from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..model_interface import Entities, Model, Predictions
from .make_analyzer import analyze_text, get_analyzer
from .transformer_inference import predict_on_text, punctuation_filter


def suppress_output():

    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    )


def merge_predictions(
    hf_preds: Predictions, pres_preds: Predictions, full_text: str
) -> Predictions:
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

    # recompute elapsed
    total_time = round(hf_preds.elapsed_ms + pres_preds.elapsed_ms, 2)
    return Predictions(entities=merged, elapsed_ms=total_time)


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

    def predict(self, text: str) -> Predictions:
        start_time = time.time()

        entities = predict_on_text(
            text,
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            threshold=self.threshold,
            filter_funcs=self.filter_funcs,
            max_length=self.max_length,
            top_k=self.top_k,
            stride=self.stride,
        )

        end_time = time.time()
        elapsed_ms = round((end_time - start_time) * 1000, 2)

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

        return Predictions(entities=predictions, elapsed_ms=elapsed_ms)


class PresidioWrappedNerModel(Model):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.analyzer = get_analyzer()

    def predict(self, text: str) -> Predictions:
        start_time = time.time()

        results = analyze_text(text, self.analyzer, self.threshold)

        end_time = time.time()
        elapsed_ms = round((end_time - start_time) * 1000, 2)

        predictions = []
        for result in results:
            predictions.append(
                Entities(
                    text=text[result.start : result.end],
                    label=result.entity_type,
                    score=result.score,
                    start=result.start,
                    end=result.end,
                )
            )

        return Predictions(entities=predictions, elapsed_ms=elapsed_ms)

    def tag_text(self, text, predictions: Predictions, verbose=False):
        tokens = text.split()
        tags = [["O", 1] for _ in tokens]

        results = [res for res in predictions.entities if res.score >= self.threshold]

        # Calculate the character start index of each token
        token_positions = []
        start = 0
        for token in tokens:
            start = text.find(token, start)
            token_positions.append((start, start + len(token)))
            start += len(token)

        if verbose:
            print(f"{text=}")
            print(f"{'---'*10} Detected Entities {'---'*10} ")
            for ent in results:
                entity = ent.label
                chunk = text[ent.start : ent.end]
                score = ent.score
                print(f"{entity=}, {chunk=}, {score=}")

        for ent in results:
            transformed_type = ent.label

            for i, (token_start, token_end) in enumerate(token_positions):
                # Check for overlap
                if (token_start >= ent.start and token_start <= ent.end) or (
                    ent.start >= token_start and ent.start <= token_end
                ):
                    # If the token is already tagged with a different entity, decide on a strategy
                    current_tag = tags[i][0]
                    if current_tag == "O":
                        tags[i] = [(transformed_type, ent.score)]
                    else:
                        if ent.score > tags[i][0][1]:
                            tags[i] = [(transformed_type, ent.score)]

        if verbose:
            print(f"{tags=}")
        return tags

    def get_tokenized_predictions(self, text, verbose=False):
        predictions = self.predict(text)
        tags = self.tag_text(text, predictions, verbose)
        return tags


class CombinedNERModel(Model):
    """
    Wraps an HF model and a Presidio model and merges their outputs.
    """

    def __init__(self, model_path, threshold):
        self.threshold = threshold
        with suppress_output()[0], suppress_output()[1]:
            self.hf = HuggingFaceModel(model_path)
            self.pres = PresidioWrappedNerModel(threshold)

    def predict(self, text: str) -> Predictions:
        # get both
        hf_out = self.hf.predict(text)
        pres_out = self.pres.predict(text)
        # merge
        merged = merge_predictions(hf_out, pres_out, text)
        return merged

    def get_tokenized_predictions(self, text, verbose=False):
        predictions = self.predict(text)
        tags = self.pres.tag_text(text, predictions, verbose)
        return tags
