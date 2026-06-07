"""
test_ingest.py — plain-assert checks for the chunking contract in ingest.py.

These checks double as documentation: each one states a rule that
`chunk_document()` promises to keep, then proves it on the real docs.

Run it two ways:

    python test_ingest.py     # runs every check, prints ✓ lines + a summary
    pytest test_ingest.py     # if you install pytest later (same test_* funcs)

The bare `assert` statements work in both — pytest discovers `test_*`
functions automatically, and the `__main__` block at the bottom runs them
without pytest installed.
"""

from ingest import load_documents, chunk_document

# Constants must match the values inside chunk_document() in ingest.py.
CHUNK_SIZE = 300
MIN_LENGTH = 50
EXPECTED_DOC_COUNT = 8


def _all_chunks():
    """Chunk every document and return a flat list of all chunks."""
    chunks = []
    for doc in load_documents():
        chunks.extend(chunk_document(doc["text"], doc["game"]))
    return chunks


def test_load_documents_returns_expected_count():
    """load_documents() should find all 8 board-game rule files."""
    documents = load_documents()
    assert len(documents) == EXPECTED_DOC_COUNT, (
        f"expected {EXPECTED_DOC_COUNT} docs, got {len(documents)}"
    )
    # Every document carries the three keys the rest of the pipeline relies on.
    for doc in documents:
        assert set(doc.keys()) == {"game", "filename", "text"}
        assert doc["text"].strip(), f"{doc['filename']} is empty"


def test_chunk_lengths_are_within_bounds():
    """Every chunk is at least MIN_LENGTH and at most CHUNK_SIZE characters."""
    for chunk in _all_chunks():
        length = len(chunk["text"])
        assert length >= MIN_LENGTH, f"{chunk['chunk_id']} too short: {length}"
        assert length <= CHUNK_SIZE, f"{chunk['chunk_id']} too long: {length}"


def test_chunk_text_is_non_empty():
    """No chunk should be empty or whitespace-only."""
    for chunk in _all_chunks():
        assert chunk["text"].strip(), f"{chunk['chunk_id']} is blank"


def test_chunk_ids_are_unique():
    """chunk_id must uniquely identify a chunk across the whole corpus."""
    ids = [chunk["chunk_id"] for chunk in _all_chunks()]
    assert len(ids) == len(set(ids)), "duplicate chunk_id found"


def test_chunk_ids_are_prefixed_with_game_name():
    """A chunk_id looks like '<game>_<n>', e.g. 'catan_0' for the Catan doc."""
    for doc in load_documents():
        prefix = doc["game"].lower().replace(" ", "_")
        for chunk in chunk_document(doc["text"], doc["game"]):
            assert chunk["game"] == doc["game"]
            assert chunk["chunk_id"].startswith(prefix + "_"), (
                f"{chunk['chunk_id']} should start with '{prefix}_'"
            )


def _run_all():
    """Run each test_* function in this module without needing pytest."""
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        fn()
        # A short, readable label so you can see what each check proved.
        print(f"  ✓ {name.replace('test_', '').replace('_', ' ')}")
    print(f"\nAll {len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
