"""
inspect_chunks.py — a learning tool for the chunking step of RAG.

Run this to *see* what `chunk_document()` in ingest.py actually produces:
how a rule document gets sliced into overlapping pieces before they ever
touch an embedding model or a vector store.

    python inspect_chunks.py            # inspects the first game (Catan)
    python inspect_chunks.py monopoly   # inspects a specific game

No API key, no network, no ChromaDB needed — ingest.py only reads ./docs.
"""

import sys

from ingest import load_documents, chunk_document


def pick_document(documents, requested):
    """Return the document matching `requested` (by game or filename), or the first one."""
    if not requested:
        return documents[0]

    needle = requested.lower().replace(".txt", "")
    for doc in documents:
        if needle in doc["game"].lower() or needle in doc["filename"].lower():
            return doc

    available = ", ".join(d["game"] for d in documents)
    raise SystemExit(f"No document matches '{requested}'. Available games: {available}")


def show_chunks(doc):
    """Print a detailed, readable breakdown of how one document gets chunked."""
    text = doc["text"]
    chunks = chunk_document(text, doc["game"])

    print("=" * 70)
    print(f"DOCUMENT: {doc['game']}  (file: {doc['filename']})")
    print("=" * 70)
    print(f"Raw document length : {len(text)} characters")
    print(f"Chunks produced     : {len(chunks)}")
    print(
        "Strategy            : 300-char window, 50-char overlap, 50-char minimum\n"
        "                      (so each chunk starts 250 chars after the previous one)"
    )

    # --- The first few chunks, in full, so you can read real chunk content ---
    preview_count = min(3, len(chunks))
    print(f"\n--- First {preview_count} chunk(s) in full ---")
    for chunk in chunks[:preview_count]:
        print(f"\n[{chunk['chunk_id']}]  ({len(chunk['text'])} chars)")
        print("┌" + "─" * 68)
        for line in chunk["text"].splitlines() or [""]:
            print("│ " + line)
        print("└" + "─" * 68)

    # --- The overlap demonstration: the whole point of a "sliding window" ---
    if len(chunks) >= 2:
        overlap = 50
        tail = chunks[0]["text"][-overlap:]
        head = chunks[1]["text"][:overlap]
        print("\n--- Overlap demonstration (why chunks share text) ---")
        print("A rule that lands on a chunk boundary could be split in half.")
        print("Overlap copies the last ~50 chars of one chunk into the start of")
        print("the next, so the rule stays retrievable intact. Compare:\n")
        print(f"  last  {overlap} chars of {chunks[0]['chunk_id']}: ...{tail!r}")
        print(f"  first {overlap} chars of {chunks[1]['chunk_id']}: {head!r}...")
        print(
            "\n  (The two strings won't match exactly because .strip() trims "
            "whitespace\n   at chunk edges, but you'll see the same words repeat.)"
        )


def show_all_summary(documents):
    """One line per game: how document size maps to chunk count."""
    print("\n" + "=" * 70)
    print("SUMMARY — all documents (size → chunk count)")
    print("=" * 70)
    for doc in documents:
        n = len(chunk_document(doc["text"], doc["game"]))
        print(f"  {doc['game']:<18} {len(doc['text']):>6} chars  ->  {n:>3} chunks")


def main():
    requested = sys.argv[1] if len(sys.argv) > 1 else None

    documents = load_documents()
    if not documents:
        raise SystemExit("No documents found in ./docs — nothing to chunk.")

    doc = pick_document(documents, requested)
    show_chunks(doc)
    show_all_summary(documents)

    print(
        "\nTip: try `python inspect_chunks.py <game>` to inspect a different "
        "file,\n     e.g. monopoly, risk, uno, clue, codenames, pandemic, "
        "ticket_to_ride."
    )


if __name__ == "__main__":
    main()
