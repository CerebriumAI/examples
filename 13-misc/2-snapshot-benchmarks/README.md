# Snapshot / Checkpoint Benchmarks

A small suite of identical ML workloads deployed three ways — **Cerebrium**,
**Modal**, and **Baseten** — so the same model can be compared across platforms,
with a focus on cold-start behaviour and each platform's snapshot/checkpoint
support.

Each numbered folder is one workload and contains all three deployments:

```
<workload>/
├── main.py            # Cerebrium handler
├── cerebrium.toml     # Cerebrium deployment config
├── modal/
│   └── app.py         # Modal deployment
└── baseten/
    ├── config.yaml    # Baseten (Truss) config
    └── model/
        └── model.py   # Baseten handler
```

## Workloads

| Folder | Workload | Model |
| --- | --- | --- |
| `1-sentence-transformers/` | Text embeddings | `BAAI/bge-small-en-v1.5` |
| `2-chatterbox-tts/` | Text-to-speech | `ChatterboxTTS` |
| `3-parakeet/` | Speech-to-text | `nvidia/parakeet-tdt-0.6b-v2` |
| `4-vit-compile/` | Image classification (`torch.compile`) | `google/vit-base-patch16-224` |
| `5-whisperx/` | Speech-to-text | `whisperx` large-v2 |
| `6-qwen-vl-sglang/` | Vision-language (SGLang) | `Qwen/Qwen2-VL-7B-Instruct` |
| `7-vllm/` | LLM serving (vLLM) | `Qwen/Qwen2.5-0.5B-Instruct` |

## How each platform handles cold starts

- **Cerebrium** — memory + GPU checkpointing. The model is loaded onto the GPU at
  import time, then a checkpoint is triggered; subsequent scale-from-zero cold
  starts restore from the checkpoint. Requires the v2 container runtime
  (`container_runtime = "v2"` in `cerebrium.toml`).
- **Modal** — `enable_memory_snapshot=True` plus the experimental
  `enable_gpu_snapshot`. The model is loaded inside `@modal.enter(snap=True)` so
  the load is captured in the snapshot and later cold starts restore from it.
- **Baseten** — no snapshotting; relies on caching model weights and container
  images. Weights are cached on a volume (`model_cache`) so cold start measures
  model load rather than download.

## Deploy

Cerebrium (from a workload folder):

```bash
cd 1-sentence-transformers
cerebrium deploy
```

Modal:

```bash
modal deploy 1-sentence-transformers/modal/app.py
```

Baseten (from the `baseten/` folder, using the Truss CLI):

```bash
cd 1-sentence-transformers/baseten
truss push
```
