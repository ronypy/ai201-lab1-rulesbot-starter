"""
inspect_retriever.py — a learning tool for the retrieval step of RAG.

Run this to *see* what `retrieve()` in retriever.py actually does: take a
question, embed it, and pull back the rule chunks whose meaning is closest.
This is the "R" in RAG — the search that grounds the LLM's answer.

    python inspect_retriever.py                      # runs a few built-in demo questions
    python inspect_retriever.py "How do I win at Catan?"   # ask your own question

What it does first:
  1. Ingests every doc in ./docs into the ChromaDB vector store (idempotent —
     re-adding the same chunk_ids just overwrites, so it's safe to re-run).
  2. Runs retrieve() for each question and prints the matches with distances.

Note: the FIRST run downloads the sentence-transformers embedding model
(~80MB, 30–60s). Later runs use the local cache. No GROQ_API_KEY needed —
retrieve() never touches the LLM.
"""

import sys

from ingest import load_documents, chunk_document
from retriever import embed_and_store, retrieve, get_collection

# A few questions that show retrieval working across different games.
DEMO_QUERIES = [
    "What happens when you roll a 7?",
    "What happens when you land on Go in Monopoly?",
    "How does the robber work?",
    "When can I play a Reverse card?",
]


def ingest_all():
    """Load, chunk, and embed every document so the store has data to search."""
    print("Ingesting documents into the vector store...")
    for doc in load_documents():
        embed_and_store(chunk_document(doc["text"], doc["game"]))
    print(f"Vector store now holds {get_collection().count()} chunks.\n")


def show_results(query):
    """Run one query through retrieve() and print the matches readably."""
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retrieve(query)
    if not results:
        print("  (no results — is the vector store empty?)")
        return

    # Lower cosine distance = more similar. Results come back best-first.
    for rank, r in enumerate(results, start=1):
        snippet = " ".join(r["text"].split())[:150]
        print(f"\n  #{rank}  [{r['game']}]  distance={r['distance']:.4f}")
        print(f"      {snippet}...")
    print(
        "\n  (distance: 0 = identical meaning, higher = less related. These are"
        "\n   the chunks generate_response() will hand to the LLM as context.)"
    )


def main():
    ingest_all()

    # Use the user's question if they passed one, otherwise the demo set.
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUERIES

    for query in queries:
        show_results(query)
        print()


if __name__ == "__main__":
    main()
