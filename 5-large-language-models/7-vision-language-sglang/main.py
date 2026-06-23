import asyncio
import base64
import http.client
import io
import json
import os
import urllib.request
from contextlib import asynccontextmanager

import sglang as sgl
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sglang import function

os.environ.setdefault("HF_HOME", "/persistent-storage/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
MODEL_DIR = f"/persistent-storage/models/{model_path}"
CHECKPOINT_URL = "http://169.254.169.253:8234/checkpoint"


def _ensure_model_downloaded() -> str:
    from pathlib import Path
    from huggingface_hub import login, snapshot_download

    model_dir = Path(MODEL_DIR)
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"[init] using cached weights at {model_dir}", flush=True)
        return str(model_dir)

    print(f"[init] downloading {model_path} to {model_dir}", flush=True)
    login(token=os.environ.get("HF_TOKEN"))
    snapshot_download(model_path, local_dir=str(model_dir))
    return str(model_dir)


def _trigger_snapshot() -> None:
    print("[init] requesting GPU snapshot", flush=True)
    try:
        req = urllib.request.Request(CHECKPOINT_URL, method="POST")
        urllib.request.urlopen(req, timeout=300)
        print("[init] snapshot complete", flush=True)
    except http.client.RemoteDisconnected:
        # TCP connections disconnect on restore and throw remote
        print("[init] snapshot complete (RemoteDisconnected)", flush=True)
    except Exception as exc:
        print(f"[init] snapshot failed: {type(exc).__name__}: {exc}", flush=True)


def _initialize_runtime() -> None:
    print("[init] starting", flush=True)
    print("[init] building SGLang runtime", flush=True)
    local_model_path = _ensure_model_downloaded()
    runtime = sgl.Runtime(
        model_path=local_model_path,
        enable_multimodal=True,
        mem_fraction_static=0.8,
        tp_size=1,
        attention_backend="flashinfer",
    )
    runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
        "qwen2-vl"
    )
    sgl.set_default_backend(runtime)
    print("[init] SGLang runtime ready", flush=True)
    _trigger_snapshot()
    print("[init] handler ready", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_initialize_runtime)
    yield


app = FastAPI(title="Vision Language SGLang API", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    image_base64: str
    ad_description: str
    dimensions: list


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


def process_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data))


@function
def analyze_ad(s, image, ad_description, dimensions):
    s += sgl.system("Evaluate an advertisement about an company's description.")
    s += sgl.user(sgl.image(image) + "Company Description: " + ad_description)
    s += sgl.assistant("Sure!")

    s += sgl.user("Is the company description related to the image?")
    s += sgl.assistant(sgl.select("related", choices=["yes", "no"]))
    if s["related"] == "no":
        return

    forks = s.fork(len(dimensions))
    for i, (f, dim) in enumerate(zip(forks, dimensions)):
        f += sgl.user("Evaluate based on the following dimension: " +
                      dim + ". End your judgment with the word 'END'")
        # Use unique slot names per dimension to avoid collisions
        f += sgl.assistant("Judgment: " + sgl.gen(f"judgment_{i}", stop="END"))

    s += sgl.user("Provide a one-sentence synthesis of the overall evaluation, then we will output JSON.")
    s += sgl.assistant(sgl.gen("summary_one_liner", stop="."))

    schema = r'^\{"summary": ".{1,400}", "grade": "[ABCD][+\-]?"\}$'
    s += sgl.user("Return only a 3 line parapgrah JSON object with keys summary and grade (A, B, C, D, +, -), where summary briefly synthesizes the above judgments.")
    s += sgl.assistant(sgl.gen("output", regex=schema))


@app.post("/analyze")
def analyze_advertisement(req: AnalyzeRequest):
    try:
        image = process_image(req.image_base64)
        state = analyze_ad.run(image, req.ad_description, req.dimensions)
        try:
            print(state)
            output = state["output"]
        except KeyError:
            output = None
        if isinstance(output, str):
            start = output.find("{")
            end = output.rfind("}") + 1
            if start != -1 and end > start:
                return {
                    "success": True,
                    "analysis": json.loads(output[start:end]),
                    "dimensions_evaluated": req.dimensions
                }
        return {
            "success": True,
            "analysis": output,
            "dimensions_evaluated": req.dimensions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
