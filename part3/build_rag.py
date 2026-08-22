import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from policy_kb import POLICY_DOCUMENTS


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

KB_DIR = BASE_DIR / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = KB_DIR / "policy_index.faiss"
CHUNKS_PATH = KB_DIR / "chunks.json"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Sentence chunking
# --------------------------------------------------

def sentence_chunk(text):
    """
    Split a policy document into individual sentences.
    Each sentence becomes one RAG chunk.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# Build RAG index
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BUILDING FLIPKART POLICY RAG INDEX")
    print("=" * 60)

    # ----------------------------------------------
    # 1. Create chunks
    # ----------------------------------------------

    chunks = []

    for policy in POLICY_DOCUMENTS:

        sentences = sentence_chunk(policy["text"])

        for sentence in sentences:

            chunks.append({
                "document_id": policy["id"],
                "title": policy["title"],
                "text": sentence
            })

    print()
    print("Policy documents:", len(POLICY_DOCUMENTS))
    print("Total chunks:", len(chunks))

    # ----------------------------------------------
    # 2. Load embedding model
    # ----------------------------------------------

    print()
    print("Loading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # ----------------------------------------------
    # 3. Generate embeddings
    # ----------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print()
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # ----------------------------------------------
    # 4. Build FAISS index
    # ----------------------------------------------

    dimension = embeddings.shape[1]

    print()
    print("Embedding dimension:", dimension)

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    # ----------------------------------------------
    # 5. Save FAISS index
    # ----------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # ----------------------------------------------
    # 6. Save chunk metadata
    # ----------------------------------------------

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ----------------------------------------------
    # 7. Final verification
    # ----------------------------------------------

    print()
    print("=" * 60)
    print("RAG BUILD COMPLETE")
    print("=" * 60)

    print("Documents :", len(POLICY_DOCUMENTS))
    print("Chunks    :", len(chunks))
    print("Dimension :", dimension)
    print("FAISS     :", INDEX_PATH)
    print("Metadata  :", CHUNKS_PATH)

    print()
    print("Files created successfully:")
    print(INDEX_PATH)
    print(CHUNKS_PATH)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()