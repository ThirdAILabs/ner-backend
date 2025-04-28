from typing import List, Dict, Any
from pydantic import BaseModel
from abc import ABC, abstractmethod


class Entities(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class Predictions(BaseModel):
    entities: List[Entities]
    elapsed_ms: float

    def to_go(self) -> List[Dict[str, Any]]:
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end
            }
            for entity in self.entities
        ]
    
    
class Model(ABC):

    @abstractmethod
    def predict(self, text: str) -> Predictions:
        pass