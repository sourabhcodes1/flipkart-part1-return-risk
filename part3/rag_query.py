from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = BASE_DIR / "knowledge_base"

INDEX_PATH = KNOWLEDGE_BASE / "policy_index.faiss"
CHUNKS_PATH = KNOWLEDGE_BASE / "chunks.json"


# -----------------------------
# Load RAG components
# -----------------------------
print("Loading knowledge base...")

index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Knowledge base loaded: {len(chunks)} chunks")
print(f"Embedding dimension: {index.d}")


# -----------------------------
# Search function
# -----------------------------
def search_policy(query, top_k=3):
    """
    Search the policy knowledge base
    and return the most relevant chunks.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        results.append({
            "score": float(distance),
            "text": chunks[idx]["text"],
            "source": chunks[idx].get("source", "unknown")
        })

    return results


# -----------------------------
# Interactive testing
# -----------------------------
if __name__ == "__main__":

    print("\nRAG POLICY SEARCH READY")
    print("Type a question.")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("Question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            continue

        results = search_policy(query)

        print("\nRelevant policy information:\n")

        for i, result in enumerate(results, start=1):

            print(f"--- Result {i} ---")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(result["text"])
            print()