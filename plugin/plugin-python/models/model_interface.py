from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class SentencePredictions(BaseModel):
    entities: List[Entity]

    def to_go(self) -> List[Dict[str, Any]]:
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end,
            }
            for entity in self.entities
        ]


class BatchPredictions(BaseModel):
    predictions: List[SentencePredictions]

    def to_go(self) -> List[Dict[str, Any]]:
        representations = [
            {"entities": sentence.to_go()} for sentence in self.predictions
        ]
        return representations


class TagInfo(BaseModel):
    """
    Information about a label/tag, includes description and examples.
    """

    name: str
    description: str
    examples: List[str]


class Sample(BaseModel):
    """
    A single training example, defined by tokens and corresponding labels.
    """

    tokens: List[str]
    labels: List[str]


class Model(ABC):

    @abstractmethod
    def predict(self, text: str) -> SentencePredictions:
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        pass

    @abstractmethod
    def finetune(
        self,
        prompt: str,
        tags: List[TagInfo],
        samples: List[Sample],
    ) -> None:
        pass
