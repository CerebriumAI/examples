import PIL
import torch
import base64
import io
from transformers import ViTFeatureExtractor, ViTForImageClassification
from fastapi import FastAPI, WebSocket

app = FastAPI()

model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Wait for the client to send data
        data = await websocket.receive_json()
        
        # Extract parameters from the received data
        frame_data = data.get("frame_data", "")
        run_id = data.get("run_id", None)
        
        # Process the image
        result = await run(
            frame_data=frame_data,
            run_id=run_id
        )
        
        # Send the processed image back to the client
        await websocket.send_json(result)
    
    except Exception as e:
        # Handle any errors
        await websocket.send_json({"error": str(e)})
    
    finally:
        # Close the connection
        await websocket.close()

async def run(frame_data: str, run_id=None):

    # Decode base64 string to bytes
    image_bytes = base64.b64decode(frame_data)
    
    # Create PIL Image from bytes
    image = PIL.Image.open(io.BytesIO(image_bytes))
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    inputs = transforms(image, return_tensors='pt')
    output = model(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)

    # Get the predicted age range label
    age_ranges = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    return {"predicted_age": age_ranges[preds.item()]}