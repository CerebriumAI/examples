# Firstly, import the conduit from cerebrium
from cerebrium import Conduit, model_type, hardware

YOUR_API_KEY = "Your private api key from the dashboard" 
your_name_for_your_deployment = "my-hf-gpt-neo-125m"

c = Conduit(
    name="hf-gpt",
    api_key=YOUR_API_KEY,
    flow=[
        (
            model_type.HUGGINGFACE_PIPELINE,
            {
                "task": "text-generation",
                "model": "EleutherAI/gpt-neo-125M",
                "max_new_tokens": 100,
            },
        ),
    ],
    # and you can chose your hardware as follows:
    hardware=hardware.TURING_5000,
    cpu=2,
    memory=8,
)


print("Deploying your conduit...")
c.deploy() # Deploy your conduit on Cerebrium
