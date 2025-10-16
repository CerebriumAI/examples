import sglang as sgl
from sglang import function
from PIL import Image
from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor
from pydantic import BaseModel
import base64
import io
import json

app = FastAPI(title="Vision Language SGLang API")
model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
processor = AutoProcessor.from_pretrained(model_path)

class AnalyzeRequest(BaseModel):
    image_base64: str
    ad_description: str
    dimensions: list

@app.on_event("startup")
def _startup_warmup():
    # Initialize engine on main thread during app startup
    runtime = sgl.Runtime(
        model_path=model_path,
        enable_multimodal=True,
        mem_fraction_static=0.8,
        tp_size=1,
        attention_backend="flashinfer",
    )
    runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
        "qwen2-vl"
    )
    sgl.set_default_backend(runtime)


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }

def process_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data))

# dimensions = ["Effectiveness", "Clarity", "Appeal", "Credibility"]

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

    # Brief one-line synthesis (separate slot to avoid clobbering)
    s += sgl.user("Provide a one-sentence synthesis of the overall evaluation, then we will output JSON.")
    s += sgl.assistant(sgl.gen("summary_one_liner", stop="."))

    # Constrain output to a simple JSON object; allow punctuation in summary
    schema = r'^\{"summary": ".{1,400}", "grade": "[ABCD][+\-]?"\}$'
    s += sgl.user("Return only a 3 line parapgrah JSON object with keys summary and grade (A, B, C, D, +, -), where summary briefly synthesizes the above judgments.")
    # Emit plain JSON (no code fence) so it appears in logs as-is
    s += sgl.assistant(sgl.gen("output", regex=schema))


@app.post("/analyze")
def analyze_advertisement(req: AnalyzeRequest):
    try:
        image = process_image(req.image_base64)
        # Ensure engine exists before running program (sets default engine)
        state = analyze_ad.run(image, req.ad_description, req.dimensions)
        # The program emitted fenced JSON; extract and parse it if present
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
        # Fallback: return raw
        return {
            "success": True,
            "analysis": output,
            "dimensions_evaluated": req.dimensions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

