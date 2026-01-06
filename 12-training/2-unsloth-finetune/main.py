
STORAGE_PATH = "/persistent-storage"
DATASET_FILE = f"{STORAGE_PATH}/dataset.jsonl"
ADAPTER_PATH = f"{STORAGE_PATH}/cipher-lora-adapter"
CHECKPOINT_DIR = f"{STORAGE_PATH}/checkpoints"

MODEL_NAME = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

def train(mx_steps: int = 100):
    # lazy import
    import os
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    try:
        mx_steps = int(mx_steps)
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid parameter type") from e

    # we use 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )

    # PEFT / LoRA setup
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, # simulates larger batch size w/o memory spike
            gradient_checkpointing = True,
            warmup_steps = 5,
            max_steps = int(mx_steps),
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = CHECKPOINT_DIR,
            save_strategy = "steps",
            save_steps = 50,
            save_total_limit = 1,
        ),
    )

    resume_from = None
    if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        print("Resuming from latest checkpoint...")
        resume_from = True 

    trainer.train(resume_from_checkpoint=resume_from)
    
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Weights saved to {ADAPTER_PATH}")
    
    return {"status": "success", "message": "Training complete or checkpoint saved."}

    

def evaluate(input_filename: str = "testset.json", output_filename: str = "finetune_results.csv"):
    # lazy import
    import os
    import json
    import csv
    from unsloth import FastLanguageModel 

    input_path = os.path.join(STORAGE_PATH, input_filename)
    output_path = os.path.join(STORAGE_PATH, output_filename)

    if not os.path.exists(input_path):
        return {"error": f"File {input_filename} not found in {STORAGE_PATH}"}

    with open(input_path, 'r') as f:
        data = json.load(f)

    raw_list = data['plain_texts']
    cipher_truth_list = data['ciphered_texts']
    
    load_path = ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else MODEL_NAME

    # 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = load_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    instruction = (
        "Apply the positional mirror-shift cipher to the input text using a GLOBAL index. "
        "The index starts at 0 for the first character and increments for EVERY character (including spaces and symbols). "
        "Rules: If the global index is even, mirror the letter ($a \\to z, A \\to Z, b \\to y, B \\to Y, c \\to x$, etc.). "
        "If the global index is odd, shift the letter forward by 3 (a \\to d, A \\to D, b \\to e, B \\to E, z \\to c$). "
        "Non-alphabetic characters do not change but still consume an index count."
    )

    batch_size = 8
    results_for_csv = []
    
    print(f"Starting evaluation on {len(raw_list)} items...")
    
    # batch inference
    for i in range(0, len(raw_list), batch_size):
        print(f"Processing batch {len(raw_list)//(i+batch_size)}")
        chunk_input = raw_list[i : i + batch_size]
        chunk_truth = cipher_truth_list[i : i + batch_size]
        
        prompts = [f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n" for text in chunk_input]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, full_text in enumerate(decoded):
            response_section = full_text.split("### Response:\n")[-1].strip()
            
            if "Final Cipher:" in response_section:
                model_cipher = response_section.split("Final Cipher:")[-1].strip()
            else:
                model_cipher = response_section.split("\n")[-1].strip()

            truth = chunk_truth[j]
            is_correct = (model_cipher == truth)

            results_for_csv.append({
                "length": len(chunk_input[j]),
                "original_text": chunk_input[j],
                "true_cipher": truth,
                "model_cipher": model_cipher,
                "is_correct": is_correct
            })

    keys = ["length", "original_text", "true_cipher", "model_cipher", "is_correct"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_for_csv)

    accuracy = sum(1 for x in results_for_csv if x['is_correct']) / len(results_for_csv) * 100
    print(f"Evaluation complete. Accuracy: {accuracy:.2f}%")

    return {
        "status": "completed",
        "accuracy": f"{accuracy:.2f}%",
        "output_file": output_path
    }
