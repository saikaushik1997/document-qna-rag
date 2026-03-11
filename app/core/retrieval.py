from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import cohere
from app.config import settings

# Initialize and return the OpenAI embeddings model - to embed the query.
def _get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
    )

# Initialize Pinecone client and return the configured index - for the search.
def _get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index)

# Initialize and return the Cohere client - reranker to improve search results.
def _get_cohere_client():
    return cohere.Client(api_key=settings.co_api_key)

# Embed the user query using the OpenAI embeddings model.
def _embed_query(query: str) -> list[float]:
    embeddings = _get_embeddings()
    return embeddings.embed_query(query)

# Run similarity search against Pinecone - return raw matches with metadata intact. Running dense search here.
def _search_pinecone(query_vector: list[float], namespace: str, top_k: int) -> list[dict]:
    index = _get_index()
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return results["matches"]

# Rerank Pinecone matches using Cohere reranker model - much smaller model, not as costly as an inference call.
# Also extracts text from metadata, reranks, then maps relevance scores back to original match objects. 
def _rerank(query: str, matches: list[dict]) -> list[dict]:
    co = _get_cohere_client()
    documents = [m["metadata"]["text"] for m in matches]

    reranked = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",
        top_n=settings.top_k
    )

    # map reranked indices back to original matches
    return [matches[r.index] for r in reranked.results]

# Core retrieval logic. Embeds the query, searches Pinecone. 
# Then it reranks results with Cohere before returning top chunks. 
# Each returned chunk includes text, filename, and page for citations.
def retrieve(query: str, namespace: str, top_k: int = None) -> list[dict]:
    top_k = top_k or settings.top_k

    # Over-fetching - fetch more than needed before reranking — reranker needs candidates to work with.
    # Since we only need top_k to be fed to the actual prompt.
    query_vector = _embed_query(query)
    matches = _search_pinecone(query_vector, namespace, top_k=top_k * 3)
    reranked = _rerank(query, matches)

    # this gets attached to the prompt
    return [
        {
            "text": m["metadata"]["text"],
            "filename": m["metadata"]["filename"],
            "page": m["metadata"]["page"],
            "score": m["score"]
        }
        for m in reranked
    ]