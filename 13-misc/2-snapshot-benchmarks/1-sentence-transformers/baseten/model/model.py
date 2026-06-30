"""Baseten (Truss) variant of the sentence-transformers snapshot benchmark.

Baseten has no GPU/memory snapshot, so there is no checkpoint trigger — every
scale-from-zero cold start pays the full model-load cost. Weights are cached on
a Baseten volume (model_cache); we block on that download, then load from the
local cache path in `load()`. Inference happens in `predict()`.
"""

from sentence_transformers import SentenceTransformer

# model_cache volume_folder mount path.
CACHE_DIR = "/app/model_cache/bge-small-en"


class Model:
    def __init__(self, **kwargs):
        self._data_resolver = kwargs.get("lazy_data_resolver")
        self._model = None

    def load(self):
        if self._data_resolver:
            self._data_resolver.block_until_download_complete()
        self._model = SentenceTransformer(CACHE_DIR, device="cuda")

    def predict(self, request: dict):
        sentences = request.get("sentences") or ["what is the meaning of life?"]
        embeddings = self._model.encode(sentences, normalize_embeddings=True)
        return {"embeddings": embeddings.tolist()}
