from http.client import HTTPException
import os
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request
from fastapi.security import APIKeyHeader
from react_agent.state import State
from react_agent.graph import builder

load_dotenv()

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/generate")
async def generate(state: State, request: Request):

    graph = builder.compile()
    graph.name = "LangGraphDeployDemo"
    print(state)
    config = {
        "configurable": {"thread_id": "3"},
        "messages": state.messages,  # Add this line
    }
    try:
        result = await graph.ainvoke(config)
        return {"success": True, "result": result}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
