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
        return [sentence.to_go() for sentence in self.predictions]


class Model(ABC):

    @abstractmethod
    def predict(self, text: str) -> SentencePredictions:
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> BatchPredictions:
        pass
