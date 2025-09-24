import os
import base64
from io import BytesIO
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
from livekit import api
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["HF_TRANSFER"] = "1"
login(token=os.environ["HF_TOKEN"])

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    attn_implementation="sdpa",  # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=False,
    init_tts=False,
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-o-2_6", trust_remote_code=True
)


async def run(
    images: list,
    question: str,
    conversation_history: list = None,
    max_tokens: int = 7,
    temperature: float = 0.7,
):
    if conversation_history is None:
        conversation_history = []

    pil_images = []
    for img_base64 in images:
        try:
            image_bytes = base64.b64decode(img_base64)
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue

    # Create message with multiple images
    msgs = [
        *conversation_history,  # Include previous messages
        {"role": "user", "content": [*pil_images, question]},
    ]

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        stream=True,
        # max_new_tokens=max_tokens,
        temperature=temperature,  # Add temperature to control randomness
        top_p=0.9,  # Add top_p to filter unlikely tokens
    )
    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end="")
        yield new_text


def create_token(room_name: str = "my-room"):
    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity("identity")
        .with_name("my name")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
    )
    return {"token": token.to_jwt()}
