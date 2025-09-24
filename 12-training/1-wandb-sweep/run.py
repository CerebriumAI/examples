import wandb
import requests
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

CEREBRIUM_API_KEY = os.getenv("CEREBRIUM_API_KEY")
ENDPOINT_URL = "https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxx/wandb-sweep/train_model?async=true"


def train_with_params(params: Dict[str, Any]):
    """
    Send training parameters to Cerebrium endpoint
    """
    headers = {
        "Authorization": f"Bearer {CEREBRIUM_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(ENDPOINT_URL, json={"params": params}, headers=headers)
    if response.status_code != 202:
        raise Exception(f"Training failed: {response.text}")

    return response.json()


# Define the sweep configuration
sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform",
            "min": -10,  # exp(-10) ≈ 4.54e-5
            "max": -7,  # exp(-7) ≈ 9.12e-4
        },
        "batch_size": {"values": [1, 2, 4]},
        "gradient_accumulation_steps": {"values": [2, 4, 8]},
        "lora_r": {"values": [8, 16, 32]},
        "lora_alpha": {"values": [16, 32, 64]},
        "lora_dropout": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "max_seq_length": {"values": [512, 1024]},
    },
}


def main():
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="Llama-3.2-Customer-Support")

    def sweep_iteration():
        # Initialize a new W&B run
        with wandb.init() as run:
            # Get the parameters for this run
            params = wandb.config.as_dict()

            # Add any fixed parameters
            params.update(
                {
                    "wandb_project": "Llama-3.2-Customer-Support",
                    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
                    "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                    "run_name": f"sweep-{run.id}",
                    "epochs": 1,
                    "test_size": 0.1,
                }
            )

            # Call the Cerebrium endpoint with these parameters
            try:
                result = train_with_params(params)
                print(f"Training completed: {result}")
            except Exception as e:
                print(f"Training failed: {str(e)}")
                run.finish(exit_code=1)

    # Run the sweep
    wandb.agent(sweep_id, function=sweep_iteration, count=10)  # Run 10 experiments


if __name__ == "__main__":
    main()
