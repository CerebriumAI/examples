"""High-throughput embedding service for Cerebrium
Supports multiple models for embeddings, reranking, and classification.
"""

from fastapi import FastAPI, Body
from infinity_emb import AsyncEngineArray, EngineArgs
import numpy as np

app = FastAPI(title="High-Throughput Embedding Service")


def embeddings_to_list(embeddings: list) -> list:
    """Convert list of numpy arrays to list of lists."""
    return [e.tolist() for e in embeddings]


class InfinityModel:
    def __init__(self):
        self.model_ids = [
            "jinaai/jina-clip-v1",
            "michaelfeil/bge-small-en-v1.5", 
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "philschmid/tiny-bert-sst2-distilled"
        ]
        self.engine_array = None
    
    def _get_array(self):
        return AsyncEngineArray.from_args([
            EngineArgs(model_name_or_path=model, model_warmup=False)
            for model in self.model_ids
        ])
    
    async def setup(self):
        print(f"Setting up models: {self.model_ids}")
        self.engine_array = self._get_array()
        await self.engine_array.astart()
        print("All models loaded successfully!")


model = InfinityModel()


@app.on_event("startup")
async def startup_event():
    """Initialize models on container startup"""
    await model.setup()


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness endpoint to report model initialization state."""
    is_ready = model.engine_array is not None
    return {"ready": is_ready}


@app.post("/embed")
async def embed(sentences: list[str] = Body(...), model_index: int = Body(1)):
    """Generate embeddings using the specified model."""
    engine = model.engine_array[model_index]
    embeddings, usage = await engine.embed(sentences=sentences)
    
    return {
        "embeddings": embeddings_to_list(embeddings),
        "usage": usage,
        "model": model.model_ids[model_index]
    }


@app.post("/image_embed")
async def image_embed(image_urls: list[str] = Body(...), model_index: int = Body(0)):
    """Generate embeddings for images using CLIP model."""
    engine = model.engine_array[model_index]
    embeddings, usage = await engine.image_embed(images=image_urls)
    
    return {
        "embeddings": embeddings_to_list(embeddings),
        "usage": usage,
        "model": model.model_ids[model_index]
    }


@app.post("/rerank")
async def rerank(query: str = Body(...), docs: list[str] = Body(...), model_index: int = Body(2)):
    """Rerank documents based on query relevance."""
    engine = model.engine_array[model_index]
    rankings, usage = await engine.rerank(query=query, docs=docs)
    
    return {
        "rankings": rankings,
        "usage": usage,
        "model": model.model_ids[model_index]
    }


@app.post("/classify")
async def classify(sentences: list[str] = Body(...), model_index: int = Body(3)):
    """Classify text sentiment."""
    engine = model.engine_array[model_index]
    classes, usage = await engine.classify(sentences=sentences)
    
    return {
        "classifications": classes,
        "usage": usage,
        "model": model.model_ids[model_index]
    }
