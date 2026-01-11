# Positional Mirror-Shift Cipher Fine-Tuning

This project implements a fine-tuning pipeline to teach a Large Language Model (LLM) a positional mirror-shift cipher. The objective is to evaluate whether a fine-tuned smaller model can outperform or match a large model on positional ciphers.

---

## Baseline and Model Selection

The project uses gpt-5-nano as the baseline for zero-shot performance comparison. While GPT-5-nano represents the frontier of general reasoning, positional ciphers often require specific pattern recognition that general-purpose models may struggle with at scale.

We have selected llama-3.1-8b-instruct for fine-tuning. This model offers a balance of parameter count and reasoning ability. By using the Unsloth library, we can perform 4-bit quantization and Parameter-Efficient Fine-Tuning (PEFT) via LoRA, making it possible to train on GPU hardware such as the Ampere A10.

---

## Project Structure

| File              | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `generate.py`     | Generates synthetic training and testing data with step-by-step Chain-of-Thought analysis. |
| `evalLLM.py`      | Benchmarks the gpt-5-nano baseline by running tests across various string lengths. |
| `main.py`         | Core training and evaluation logic deployed on Cerebrium. Handles 4-bit training (LoRA) and batch inference for evaluation. |
| `cerebrium.toml`  | Configuration for deployment |
| `plot.py`         | Processes results from the baseline and fine-tuned model to plot accuracy across cipher length |

---

## Workflow

1. **Baseline Evaluation**: Run `evalLLM.py` to determine how well GPT-5-nano handles the cipher rules without fine-tuning.
2. **Data Generation**: Run `generate.py` locally to create `dataset.jsonl` (training) and `testset.json` (evaluation).
3. **Cloud Preparation**: Deploy the environment to Cerebrium using the configuration in `cerebrium.toml`.
4. **Data Transfer**: Move local datasets to Cerebrium persistent storage.
5. **Training**: Trigger the `train` function in `main.py` via the Cerebrium API.
6. **Model Evaluation**: Trigger the `evaluate` function in `main.py` to test the fine-tuned adapter.
7. **Result Retrieval**: Download the results to your local machine.
8. **Visualization**: Run `plot.py` to compare the two models.

---

## Technical Requirements and Warnings

> **[IMPORTANT] Compatibility Warning**  
> Before running the training pipeline, verify that the latest version of **Unsloth** is compatible with the specific **CUDA** and **PyTorch** versions. Unsloth is highly optimized for specific kernels; version mismatches between `bitsandbytes`, `torch`, and `cuda-toolkit` can lead to runtime errors or significant performance degradation.

---

## Cipher Logic

The model is trained to follow these rules based on a global index \( i \):

- **Even Index**: Mirror the character  
  \( a \rightarrow z,\; b \rightarrow y \)
- **Odd Index**: Shift the character forward by 3  
  \( a \rightarrow d,\; z \rightarrow c \)
- **Non-Alphabetic**: Index increments, but the character remains unchanged.

---

## Cerebrium Storage and Transfer

Cerebrium provides a persistent volume at `/persistent-storage` to ensure models and datasets persist across container restarts.

### Upload Local Data

To pass your locally generated datasets to cloud storage:

```bash
cerebrium cp ./results/dataset.jsonl dataset.jsonl
cerebrium cp ./results/testset.json testset.json
```

### Download Results

To retrieve the fine-tuned model’s performance CSV for local plotting:

```bash
cerebrium download finetune_results.csv ./results/finetune_results.csv
```
