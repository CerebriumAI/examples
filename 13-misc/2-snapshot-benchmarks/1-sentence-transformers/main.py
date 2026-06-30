from sentence_transformers import SentenceTransformer
import urllib.request
import http.client
# --- Cold-start work that we want the snapshot to capture ----------------------
# Loading the embedding model onto the GPU is the expensive part of a cold start.
# We run it at module import time so it executes once, before the checkpoint.
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")


try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    urllib.request.urlopen(req)
    print("Checkpointed successfully")    
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw remote
    pass


def run(sentences: list[str] = None):
    if sentences is None:
        sentences = ["what is the meaning of life?"]
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}
