import os
import warnings
from typing import List

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    RecognizerResult,
    BatchAnalyzerEngine,
)

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_analyzer() -> AnalyzerEngine:
    """Create an analyzer with only pattern-based recognizers for the defined entities."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    # Remove the ML-based Spacy recognizer
    registry.remove_recognizer("SpacyRecognizer")
    return AnalyzerEngine(registry=registry, supported_languages=["en"])


def get_batch_analyzer() -> BatchAnalyzerEngine:
    analyzer = get_analyzer()
    return BatchAnalyzerEngine(analyzer_engine=analyzer)


entities_map = {
    "LOCATION": "ADDRESS",
    "US_DRIVER_LICENSE": "VIN",
    "PHONE_NUMBER": "PHONENUMBER",
    "DATE_TIME": "DATE",
    "EMAIL_ADDRESS": "EMAIL",
    "CREDIT_CARD": "CARD_NUMBER",
    "US_SSN": "SSN",
    "URL": "URL",
    "US_PASSPORT": "ID_NUMBER",
    "US_ITIN": "ID_NUMBER",
    "US_BANK_NUMBER": "ID_NUMBER",
    "IN_PAN": "ID_NUMBER",
    "IN_AADHAAR": "ID_NUMBER",
    "IN_VEHICLE_REGISTRATION": "VIN",
}
entities = list(entities_map.keys())


def transform_existing_tags(text, res: RecognizerResult):
    if res.entity_type not in entities_map:
        raise Exception(
            f"Invalid entity {res.entity_type} found for text {text}. {res=}"
        )

    res.entity_type = entities_map[res.entity_type]


def analyze_text(
    text: str, analyzer: AnalyzerEngine, threshold: float
) -> List[RecognizerResult]:
    """Analyze text for the given entities and remap tags using entities_map."""
    results = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
        score_threshold=threshold,
    )
    for res in results:
        transform_existing_tags(text, res)
    return results


def analyze_text_batch(
    texts: List[str], analyzer: BatchAnalyzerEngine, threshold: float
) -> List[List[RecognizerResult]]:
    results = analyzer.analyze_iterator(
        texts=texts,
        entities=entities,
        language="en",
        score_threshold=threshold,
    )

    for text, text_results in zip(texts, results):
        for res in text_results:
            transform_existing_tags(text, res)

    return results
