from typing import List

from thirdai import bolt


class UDTModel:
    def __init__(self, model_path: str):
        self.model = bolt.UniversalDeepTransformer.load(model_path)

    def predict(self, text: str) -> List[List[str]]:
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]):
        results = self.model.predict_batch(
            [{"source": text} for text in texts], top_k=1
        )
        tags = []
        for res in results:
            tags.append([r[0][0] for r in res])
        return tags
