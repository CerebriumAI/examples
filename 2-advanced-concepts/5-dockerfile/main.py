from fastapi import FastAPI

app = FastAPI()

@app.post("/hello")
def hello():
    return {"message": "Hello Cerebrium!"}

@app.get("/health")
def health():
    return "OK"