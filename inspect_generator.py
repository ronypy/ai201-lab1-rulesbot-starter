"""
inspect_generator.py — a learning tool for the generation step of RAG.

Run this to *see* the full RAG chain end to end: a question goes in, retrieve()
finds the closest rule chunks, and generate_response() turns them into a
grounded answer that cites its game. This is the "G" in RAG.

    python inspect_generator.py                       # built-in demo questions
    python inspect_generator.py "How do I win at Catan?"   # ask your own

For each question it prints:
  1. the chunks retrieve() pulled (so you can see what the LLM was given)
  2. the answer generate_response() produced from ONLY those chunks

It also runs a small GROUNDING PROBE: questions whose answers are NOT in the
rule books. A trustworthy RulesBot should say "the rules don't cover that"
instead of answering from the model's own board-game knowledge. Watch those
closely — they're the real test of whether grounding works.

Requirements:
  - GROQ_API_KEY must be set (generate_response() calls the Groq LLM).
    Put it in a .env file or export it before running.
  - First run downloads the embedding model (~80MB) for retrieval.
"""

import sys

from config import GROQ_API_KEY
from ingest import load_documents, chunk_document
from retriever import embed_and_store, retrieve, get_collection
from generator import generate_response

# Questions whose answers ARE in the loaded rules — should get grounded answers.
DEMO_QUERIES = [
    "What happens when you roll a 7?",
    "How do I win at Catan?",
    "What happens when you roll a 7 in Catan?",
    "When can I play a Reverse card in Uno?",
]

# Questions whose answers are NOT in the rules. A grounded bot should decline,
# not answer from pretraining. This is the trustworthiness test.
GROUNDING_PROBES = [
    "What is the capital of France?",                  # totally off-topic
    "What are the official tournament rules for Chess?",  # game we never loaded
    "How much does the game of Catan cost to buy?",    # in-game, but not in the rules
]


def ingest_all():
    """Load, chunk, and embed every document so retrieval has data to search."""
    print("Ingesting documents into the vector store...")
    for doc in load_documents():
        embed_and_store(chunk_document(doc["text"], doc["game"]))
    print(f"Vector store now holds {get_collection().count()} chunks.\n")


def ask(query):
    """Run one question through the full retrieve -> generate chain and print both stages."""
    print("=" * 72)
    print(f"QUESTION: {query}")
    print("=" * 72)

    chunks = retrieve(query)

    print("Retrieved context (what the LLM was allowed to use):")
    if not chunks:
        print("  (nothing retrieved)")
    for i, c in enumerate(chunks, start=1):
        snippet = " ".join(c["text"].split())[:90]
        print(f"  {i}. [{c['game']}] distance={c['distance']:.4f}  {snippet}...")

    answer = generate_response(query, chunks)
    print("\nRulesBot answer:")
    for line in answer.splitlines():
        print(f"  {line}")
    print()


def main():
    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set. generate_response() needs it to call the LLM.\n"
            "Set it in a .env file or `export GROQ_API_KEY=...` and re-run."
        )

    ingest_all()

    if len(sys.argv) > 1:
        # User asked their own question — just answer that one.
        ask(" ".join(sys.argv[1:]))
        return

    print("########## GROUNDED QUESTIONS (answers ARE in the rules) ##########\n")
    for q in DEMO_QUERIES:
        ask(q)

    print("########## GROUNDING PROBES (answers are NOT in the rules) ##########")
    print("# A trustworthy bot declines these instead of answering from memory.\n")
    for q in GROUNDING_PROBES:
        ask(q)


if __name__ == "__main__":
    main()
