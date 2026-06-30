"""Modal port of the sentence-transformers snapshot benchmark.

Mirror of ../main.py (Cerebrium memory checkpointing) using Modal's
`enable_memory_snapshot` + experimental `enable_gpu_snapshot`. The expensive
cold-start work (loading the embedding model onto the GPU) runs inside
`@modal.enter(snap=True)` so it is captured in the snapshot; subsequent cold
starts restore from the snapshot instead of reloading.

Deploy:  modal deploy app.py
"""

import modal

MODEL_ID = "BAAI/bge-small-en-v1.5"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers",
    "torch",
)

app = modal.App("snapshot-bench-sentence-transformers", image=image)


@app.cls(
    gpu="A10G",
    memory=16384,
    scaledown_window=2,  # scale to zero quickly so the next call is a cold start
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_ID, device="cuda")

    @modal.method()
    def infer(self, sentences: list[str] | None = None):
        if sentences is None:
            sentences = ["what is the meaning of life?"]
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        return {"embeddings": embeddings.tolist()}
