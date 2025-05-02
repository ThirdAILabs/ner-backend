from thirdai import bolt

from typing import List


class UDTModel:
    def __init__(self, name: str, model_path: str):
        self.model_path = model_path
        self.model = bolt.UniversalDeepTransformer.load(model_path)

    def predict(self, text: str) -> List[List[str]]:
        cleaned_chars: List[str] = []
        index_mapping: List[int] = []
        for idx, char in enumerate(text):
            if ord(char) < 128:
                cleaned_chars.append(char)
                index_mapping.append(idx)
        cleaned_text = "".join(cleaned_chars)

        results = self.model.predict({"source": cleaned_text}, top_k=1)
        print(results)
        return [result[0][0] for result in results]
