from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ingestion import ingest_file
from app.core.retrieval import retrieve
from app.core.generation import generate, create_chat_history
from app.models.schemas import IngestResponse, QueryRequest, QueryResponse, SourceChunk

router = APIRouter(prefix="/api/v1", tags=["query"])

# in-memory chat history store, keyed by session_id
# resets on server restart.
# one chat history per document.
chat_histories = {}

# Accept a PDF or DOCX upload, chunk and index it in Pinecone. 
# Returns metadata about the ingestion.
@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    file_bytes = await file.read()
    result = ingest_file(file_bytes, file.filename)
    return IngestResponse(**result)

# Accept a question and namespace, retrieve relevant chunks, generate an answer.
# Return above answer with source citations(metadata).
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # get or create chat history for this session
    if request.namespace not in chat_histories:
        chat_histories[request.namespace] = create_chat_history()
    chat_history = chat_histories[request.namespace]

    chunks = retrieve(request.question, request.namespace)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found for this query")

    answer = generate(request.question, chunks, chat_history)

    sources = [SourceChunk(**chunk) for chunk in chunks]

    return QueryResponse(answer=answer, sources=sources)