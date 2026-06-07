# Spec: `retrieve()`

**File:** `retriever.py`
**Status:** Spec incomplete — fill in all blank fields before implementing

---

## Purpose

Given a user's natural language query, find the most relevant chunks from the vector store using semantic similarity search. Return them ranked by relevance so that `generate_response()` can use them as context.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The user's natural language question |
| `n_results` | `int` | Maximum number of chunks to return (default: `N_RESULTS` from `config.py`) |

**Output:** `list[dict]`

Each dict in the returned list must contain exactly these keys:

| Key | Type | Description |
|-----|------|-------------|
| `"text"` | `str` | The chunk text |
| `"game"` | `str` | The game name this chunk came from |
| `"distance"` | `float` | Cosine distance score — lower means more similar to the query |

Results should be ordered from most to least relevant (lowest to highest distance). Returns an empty list `[]` if the collection contains no documents.

---

## Design Decisions

*Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours.*

---

### Query approach

*Describe how you will use `_collection.query()` to find relevant chunks. What arguments will you pass, and why?*

```
I call _collection.query() with three arguments:

  - query_texts=[query]
      The query is wrapped in a list because ChromaDB supports batching
      multiple queries at once. I only have one question, so it's a
      single-element list. ChromaDB embeds this text with the SAME
      sentence-transformers model used at ingestion, so the question and
      the stored chunks live in the same 384-dim vector space and can be
      compared by distance.

  - n_results=n_results
      How many of the nearest chunks to return. Defaults to N_RESULTS (3)
      from config.py, but callers can override it.

  - include=["documents", "metadatas", "distances"]
      I ask for exactly the three pieces I need to build my return dicts:
      the chunk text (documents), the game name (metadatas), and the
      similarity score (distances). I do NOT include "embeddings" — the
      raw vectors are large and useless to the caller, so leaving them out
      keeps the payload small.

```

---

### Return structure

*Sketch out what one item in your return list looks like as a concrete example. Where does each field come from in the query results?*

```
One returned item looks like:

  {
    "text": "He moves the robber to any terrain hex and steals one random
             resource card from a player with a settlement or city adjacent
             to that hex. TRADING You may tra...",
    "game": "Catan",
    "distance": 0.4630,
  }

Where each field comes from in the query results (after indexing [0]):
  - "text"     <- results["documents"][0][i]   (the chunk text itself)
  - "game"     <- results["metadatas"][0][i]["game"]
                  (pulled out of the metadata dict I stored in embed_and_store)
  - "distance" <- results["distances"][0][i]   (cosine distance, lower = closer)

I zip the three parallel lists together so index i lines up across all
three — documents[i], metadatas[i], and distances[i] all describe the
same chunk.

```

---

### Handling the nested result structure

*`_collection.query()` returns nested lists. Describe what index you need to access to get the actual list of results for a single query, and why the nesting exists.*

```
_collection.query() returns each field as a LIST OF LISTS — one inner list
per query string in query_texts. The shape is:

  results["documents"]  ==  [ [chunk, chunk, chunk] ]
                              ^outer = per-query   ^inner = the n_results

The nesting exists because query() is built to handle a BATCH of queries.
If I'd passed query_texts=["A", "B"], I'd get two inner lists back, one
per question. Since I send exactly one query, I take index [0] of each
field to unwrap the single query's results:

  documents = results["documents"][0]
  metadatas = results["metadatas"][0]
  distances = results["distances"][0]

Without the [0], I'd be iterating over query-groups instead of over the
actual chunks.

```

---

### Relevance threshold

*Will you filter out results above a certain distance score, or return all `n_results` regardless of how relevant they are? What are the tradeoffs of each approach?*

```
Decision: return all n_results regardless of distance — no threshold filter.

Reasoning / tradeoffs:
  - No threshold (what I chose): simple, predictable, and never returns an
    empty list when data exists. The downside is that weak matches still
    come through. Example from my test: querying "How do I win at Catan?"
    returned a Risk "WINNING" chunk at distance 0.61 as result #2 — only
    loosely related, but n_results=3 forced a third pick.
  - With a threshold (e.g. drop anything above ~0.7): cleaner context, fewer
    irrelevant chunks reaching the LLM. But cosine distances are not
    absolute — a "good" cutoff varies by query and corpus, so a fixed
    threshold risks dropping real answers or keeping junk. It's brittle to
    tune.

I keep retrieve() purely about ranking and push relevance judgment to
generate_response() (Milestone 3): I'd rather give the LLM the top chunks
plus their distances and prompt it to answer only from genuinely relevant
context (and say "I don't know" otherwise) than hard-filter here and risk
discarding a correct chunk.

```

---

### Edge cases

*How does your implementation behave when: (a) the collection is empty, (b) the query matches no chunks well, (c) the query matches chunks from multiple games?*

```
(a) Collection is empty:
    I guard with `if _collection.count() == 0: return []` before querying.
    This avoids calling query() on an empty store and matches the contract
    (empty list when there are no documents).

(b) Query matches no chunks well:
    query() ALWAYS returns the n_results nearest chunks — there is no
    "nothing matched" case. So I still get 3 results, just with high
    distance scores. retrieve() returns them as-is; the high distances are
    the signal that they're weak. (Generation will decide whether they're
    good enough to answer from.)

(c) Query matches chunks from multiple games:
    Totally fine and expected. The results are ranked purely by distance,
    so a single query can return a mix of games. I saw this with
    "How do I win at Catan?" returning both Catan and Risk chunks. The
    "game" field on each dict lets the caller (and the user) see exactly
    which game each chunk came from.

```

---

## Implementation Notes

*Fill this in after implementing, before moving to Milestone 3.*

**Test query and top result returned:**

```
Query: What happens when you roll a 7?
Top result game: Catan
Distance score: 0.4664
Does it make sense? yes. But, the n-results are not same as showed in the Tinker.
```

**One thing about the query results that surprised you:**

```
Semantic search returned the RIGHT chunk without keyword overlap. "How does
the robber work?" pulled the robber rule even though I phrased it as a
casual question, not the document's wording. The flip side surprised me too:
querying about Catan also surfaced a Risk "WINNING" chunk, because the model
groups all "how to win" text together by meaning regardless of game. It made
me realize retrieval ranks by semantic closeness, not by topic correctness —
so a low rank doesn't guarantee the chunk is actually about what I asked.

```
