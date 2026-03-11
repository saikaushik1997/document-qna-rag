from pydantic import BaseModel

# Model for Object returned after a document is successfully ingested.
class IngestResponse(BaseModel):
    filename: str
    namespace: str
    chunks_indexed: int
    pages: int

# Model for incoming query from the user.
class QueryRequest(BaseModel):
    question: str
    namespace: str
    streaming: bool = False

# Model for a single retrieved chunk with metadata - used in the QueryResponse model
class SourceChunk(BaseModel):
    text: str
    filename: str
    page: float
    score: float

# Model for Object returned after a query is processed.
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]