# app/core/ingestion.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from app.config import settings
import tempfile
import os

# Initialize and return the embeddings model
def _get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
    )

# Initialize Pinecone client and return the index itself - currently using free tier of Pinecone - so just the 1 index.
def _get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index)

# Loads the data from the tmp file created - so it can then be chunked and embedded.
def _load_document(file_path: str, file_type: str):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return loader.load()

# Chunks the documents to prevent the whole doc being loaded into the context and overloading the LLM model's limits. 
def _chunk_documents(docs, strategy: str = "recursive"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

# Insert/Update to the Pinecone index.
def _upsert_to_pinecone(chunks, filename: str, index):
    embeddings = _get_embeddings()
    namespace = filename.replace(" ", "_").lower()

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        vectors.append({
            "id": f"{namespace}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "filename": filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_index": i
            }
        })

    # upsert in batches of 100 - safe default, upsert API has limits
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)

    return namespace

def ingest_file(file_bytes: bytes, filename: str, strategy: str = "recursive") -> dict:
    # creating a temp file to store documents in since we get bytes input when a file is uploaded and we need a file for the loader
    file_type = filename.split(".")[-1].lower()

    # write to temp file — loaders need file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # core ingesting/indexing pipeline - load, chunk, upsert(update/insert to vector store)
    try:
        docs = _load_document(tmp_path, file_type)
        chunks = _chunk_documents(docs, strategy)
        index = _get_index()
        namespace = _upsert_to_pinecone(chunks, filename, index)
    finally:
        os.unlink(tmp_path)  # clean up temp file

    return {
        "filename": filename,
        "namespace": namespace,
        "chunks_indexed": len(chunks),
        "pages": len(docs)
    }