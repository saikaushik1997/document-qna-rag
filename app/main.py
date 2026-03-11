# server starts here
from fastapi import FastAPI
from app.api.routes.query import router

app = FastAPI(
    title="Document QnA RAG",
    description="Upload documents and ask questions about them",
    version="1.0.0"
)

app.include_router(router)

# simple health check endpoint to ping at.
@app.get("/health")
async def health():
    return {"status": "ok"}