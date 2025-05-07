from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class Entities(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class SentencePredictions(BaseModel):
    entities: List[Entities]

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
    def predict(self, texts: List[str]) -> BatchPredictions:
        pass
