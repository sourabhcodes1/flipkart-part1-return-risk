from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
KB = BASE / "knowledge_base"
TRANSCRIPTS = BASE / "transcripts"
TRANSCRIPTS.mkdir(parents=True, exist_ok=True)

# ---------- Load actual RAG index ----------
index = faiss.read_index(str(KB / "policy_index.faiss"))
with open(KB / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- 8 actual test conversations ----------
queries = [
    "What is the return policy for apparel?",
    "What is the return policy for electronics?",
    "How long can footwear be returned after delivery?",
    "When will I receive my COD refund?",
    "Can I exchange a product?",
    "What happens if my delivery is delayed?",
    "Is reverse pickup available for returns?",
    "What condition should a returned product be in?"
]

# ---------- Generate 8 transcript files ----------
for i, question in enumerate(queries, 1):
    from answer_generator import generate_answer
    try:
        answer = generate_answer(question)
    except Exception as e:
        answer = f"Error while generating answer: {e}"

    text = (
        f"# Test Conversation {i}\n\n"
        f"**User:** {question}\n\n"
        f"**Assistant:** {answer}\n"
    )

    (TRANSCRIPTS / f"conversation_{i:02d}.md").write_text(
        text, encoding="utf-8"
    )

# ---------- Retrieval evaluation ----------
eval_queries = [
    ("What is the return policy for apparel?", "POL001"),
    ("What is the return policy for footwear?", "POL002"),
    ("What is the return policy for electronics?", "POL003"),
    ("When is a COD refund processed?", "POL005"),
    ("What is the exchange policy?", "POL013"),
]

lines = [
    "# Part 3 Retrieval Evaluation",
    "",
    "Evaluation is performed at document level.",
    "Top-3 retrieved chunks are mapped to their parent document IDs and deduplicated before scoring.",
    ""
]

precision_values = []
recall_values = []

for n, (query, gold) in enumerate(eval_queries, 1):
    emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, ids = index.search(emb, 3)

    retrieved_docs = []
    for idx in ids[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        doc_id = chunks[idx]["document_id"]
        if doc_id not in retrieved_docs:
            retrieved_docs.append(doc_id)

    relevant = 1 if gold in retrieved_docs else 0

    precision = relevant / 3
    recall = relevant / 1

    precision_values.append(precision)
    recall_values.append(recall)

    lines.extend([
        f"## Query {n}",
        f"**Query:** {query}",
        f"**Gold document:** {gold}",
        f"**Retrieved document IDs (top-3, deduplicated):** {retrieved_docs}",
        f"**Relevant retrieved documents:** {relevant}",
        f"**Precision@3:** {relevant}/3 = {precision:.4f}",
        f"**Recall@3:** {relevant}/1 = {recall:.4f}",
        ""
    ])

avg_p = sum(precision_values) / len(precision_values)
avg_r = sum(recall_values) / len(recall_values)

lines.extend([
    "## Final averages",
    f"**Average Precision@3:** ({' + '.join(f'{x:.4f}' for x in precision_values)}) / {len(precision_values)} = {avg_p:.4f}",
    f"**Average Recall@3:** ({' + '.join(f'{x:.4f}' for x in recall_values)}) / {len(recall_values)} = {avg_r:.4f}",
    ""
])

(KB.parent / "retrieval_evaluation.md").write_text(
    "\n".join(lines), encoding="utf-8"
)

print("=" * 60)
print("SUBMISSION ARTIFACTS CREATED")
print("=" * 60)
print("8 transcripts created in part3/transcripts/")
print("retrieval_evaluation.md created")
print(f"Average Precision@3: {avg_p:.4f}")
print(f"Average Recall@3:    {avg_r:.4f}")
print("=" * 60)
