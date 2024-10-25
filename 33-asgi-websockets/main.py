from fastapi import FastAPI, WebSocket

app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Web socket endpoint
@app.websocket("/your-websocket-function-name")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello, WebSocket!")
    await websocket.close()