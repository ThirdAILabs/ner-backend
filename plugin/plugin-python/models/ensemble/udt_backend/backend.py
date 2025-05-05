from thirdai import bolt
import thirdai

from typing import List

# TODO(pratik): remove this dependency on license
thirdai.licensing.activate("EUKX-7FXA-4WAX-XMYT-VWWY-MJPR-XTC3-7F9V")


class UDTModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = bolt.UniversalDeepTransformer.load(model_path)

    def predict(self, text: str) -> List[List[str]]:
        results = self.model.predict({"source": text}, top_k=1)
        return [result[0][0] for result in results]
