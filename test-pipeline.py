# test_pipeline.py - just running sanity checks
import sys
sys.path.append(".")

from app.core.ingestion import ingest_file
from app.core.retrieval import retrieve
from app.core.generation import generate, create_chat_history

# --- test ingestion ---
print("Testing ingestion...")
with open("test.pdf", "rb") as f:
    result = ingest_file(f.read(), "test.pdf")
print(result)

# --- test retrieval ---
print("\nTesting retrieval...")
chunks = retrieve("what is this document about?", namespace="test.pdf")
for c in chunks:
    print(c["filename"], "| page", c["page"])
    print(c["text"][:100])
    print("---")

# --- test generation ---
print("\nTesting generation...")
chat_history = create_chat_history()
answer = generate("what is this document about?", chunks, chat_history)
print(answer)