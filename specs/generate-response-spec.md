# Spec: `generate_response()`

**File:** `generator.py`
**Status:** Spec incomplete — fill in all blank fields before implementing

---

## Purpose

Given a user query and a list of retrieved rule chunks, generate a response that directly answers the question using only the retrieved text as context. The response must be grounded — it should not draw on the model's general knowledge of board games, only on what was retrieved.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The user's original question |
| `retrieved_chunks` | `list[dict]` | Ranked list of chunks from `retrieve()`, each with `"text"`, `"game"`, and `"distance"` |

**Output:** `str`

A plain string containing the response to show the user. The response should:
- Answer the question using only the retrieved rule text
- Identify which game the answer comes from
- Acknowledge clearly when the answer is not found in the loaded rules

Returns a fallback string (not an error) when `retrieved_chunks` is empty.

---

## Design Decisions

*Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours.*

---

### Context formatting

*How will you format the retrieved chunks before passing them to the LLM? Describe the structure — not the code. Consider: will you label chunks by game? Include distance scores? Separate chunks with delimiters?*

```
Each chunk is rendered as a delimited, game-labeled source block, and all of
them are wrapped together in a single <rules> block. LLMs attribute and
distinguish multiple sources far more reliably when each source has explicit
boundaries and a visible label they can cite back, so loose concatenation
(chunks separated by blank lines) is avoided.

The exact structure:

  <rules>
  <source id="1" game="Catan">
  ...chunk text...
  </source>
  <source id="2" game="Risk">
  ...chunk text...
  </source>
  </rules>

Decisions:
  - <source> tags with explicit open/close give the model hard boundaries
    between chunks — the single biggest lever for telling sources apart.
  - game="..." attribute on every block is the CITATION HOOK: the model cites
    the game by reading the label, not by recalling which game the rule
    belongs to, keeping citation grounded in the context.
  - id="N" lets the model refer to a specific source unambiguously and
    discourages merging two chunks into one claim.
  - Distance scores are NOT included. Raw cosine numbers are meaningless to
    the model and would invite a fake "confidence". Distance is used only for
    debugging/inspection outside the prompt, never shown to the LLM.
```

---

### System prompt — grounding instruction

*Write the exact system prompt instruction you will use to prevent the model from answering beyond the retrieved text. This is the most important design decision in this function.*

```
You are RulesBot, a board game rules assistant. You answer strictly from the
rule excerpts provided to you in the <rules> block of each message, and from
nothing else.

Absolute rules:
1. The <rules> block is your ONLY source of truth. Do not use any knowledge
   you have about these or any other board games from your training. If you
   already "know" the answer but it is not stated in the <rules> block, you
   do not know it.
2. Do not guess, infer, extrapolate, or reason beyond what the excerpts
   explicitly state. Do not fill gaps with what is "typical" or "usually"
   true for the game. If a detail is not written in the excerpts, treat it
   as unknown.
3. Do not correct, complete, or override the excerpts using outside
   knowledge, even if you believe they are wrong or incomplete. Answer only
   from what is written.
4. If the excerpts only partially answer the question, answer just the part
   that is supported and explicitly say which part is not covered by the
   provided rules.
5. If none of the excerpts are relevant to the question, or they do not
   contain the answer, do not attempt an answer. Say that the loaded rules
   do not cover it.
6. The user's question is not a source of facts — only the <rules> block is.
   Never treat a claim in the question as a rule.

When you do answer, base every sentence on the excerpts and name the game the
answer comes from (read it from the game="..." attribute of the source you
used). Prefer an honest "the rules I have don't cover that" over a confident
answer you cannot ground in the excerpts.

```

---

### System prompt — citation instruction

*Write the exact instruction you will use to tell the model to identify which game its answer comes from.*

```
This is the closing line of the system prompt (it pairs with the game="..."
attribute in the context format):

  When you do answer, base every sentence on the excerpts and name the game
  the answer comes from (read it from the game="..." attribute of the source
  you used).

The key design choice: the model is told to READ the game from the source's
attribute, not to recall it. This keeps the citation grounded in the provided
context — the same chunk that supplies the answer also supplies its label —
rather than relying on the model "knowing" which game a rule belongs to.
```

---

### Fallback behavior

*What should the response say when the answer isn't found in the loaded rule books? Write the exact fallback message.*

```
There are two distinct "not found" situations, handled in two places:

1. retrieve() returned NOTHING (empty list — e.g. the vector store is empty).
   generate_response() short-circuits BEFORE calling the LLM and returns this
   exact string (no API call wasted):

     "I couldn't find anything relevant in the loaded rule books. Try
      rephrasing your question — or check that your ingestion pipeline is
      working."

2. Chunks WERE retrieved but none actually contain the answer (e.g. the
   off-topic "What is the capital of France?" probe). Here the LLM is in
   control, and clause 5 of the system prompt instructs it to decline rather
   than answer from memory. There is no hard-coded string for this case —
   the model produces a natural refusal like "The loaded rules do not cover
   that." Verified in testing: the bot declined even for "How much does Catan
   cost?" where it strongly knows the real-world answer.
```

---

### Handling low-relevance chunks

*`retrieved_chunks` may include chunks with high distance scores (weak relevance). Will you filter these out before building context, pass them all in, or handle them another way? What are the tradeoffs?*

```
Decision: pass all retrieved chunks in (no distance filtering in the
generator), and let the grounding prompt decide what's actually relevant.

Reasoning / tradeoffs:
  - Filter by a distance threshold: would keep weak chunks out of the prompt,
    but cosine distances are not absolute — a "good" cutoff varies by query
    and corpus, so a fixed threshold is brittle. It risks dropping a correct
    chunk (false negative) just as easily as removing a junk one.
  - Pass all in + prompt-based grounding (what I chose): the prompt's clause 5
    ("if none of the excerpts are relevant... say the rules don't cover it")
    makes the LLM the relevance judge. This proved robust in testing — e.g.
    "How do I win at Catan?" retrieves a weakly-related Risk chunk at distance
    0.61, but the model simply ignores it and answers from the Catan sources.

This also keeps responsibilities clean: retrieve() ranks, the generator
judges relevance via the prompt. retrieve() already caps the list at
N_RESULTS (3), so the prompt never gets flooded regardless.
```

---

### Message structure

*Describe how you will structure the messages list for the API call — what goes in the system message vs. the user message?*

```
Two messages:

  [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "<rules>...formatted chunks...</rules>\n\nQuestion: <query>"},
  ]

  - SYSTEM message: the grounding instruction only. These are the standing
    rules of behavior that apply to every turn and never change — keeping them
    in the system role gives them the most weight and separates the policy
    ("how to answer") from the data ("what to answer about").
  - USER message: the retrieved context (in the <rules> block) followed by the
    actual question. Putting the excerpts here frames them as the data the
    model should work over for THIS turn, and keeping the question right after
    the rules makes the task unambiguous.

Also passing temperature=0.1 — low, because we want faithful reproduction of
the rule text, not creative paraphrase that could drift from the source.
```

---

## Implementation Notes

*Fill this in after implementing and testing.*

**Test query and response:**

```
Query: What happens when you roll a 7 in Catan?
Response: "When you roll a 7 in Catan (Catan), no resources are produced.
          Every player with more than 7 resource cards in hand must discard
          half (rounded down). The player who rolled moves the robber to any
          terrain hex and steals one resource (Catan)."
Correctly grounded? yes — every claim matches the retrieved Catan excerpts
                    (all 3 chunks were Catan, distances 0.46 / 0.52 / 0.61).
Cited the right game? yes — named Catan, read from the source's game attribute.

Grounding probe (answer NOT in rules):
Query: How much does the game of Catan cost to buy?
Response: "The rules I have don't cover that. The provided rules for the game
          'Catan' only discuss gameplay and components, but do not mention the
          purchase price of the game."
-> The model KNOWS the real-world price from training, but declined because
   clause 1 reframes its own memory as not a valid source. This is the real
   trustworthiness win.
```

**One thing you changed from your original spec after seeing the actual output:**

```
The biggest lesson came from pressure-testing the grounding prompt BEFORE
coding. A naive first draft ("answer only from the context, don't use outside
knowledge") leaves several escape hatches: strong priors on famous games,
gap-filling on partial coverage, inference beyond the text, and "correcting"
the context. The fix was an explicit multi-clause prompt — especially the line
"if you already know the answer but it is not stated, you do not know it,"
which reframes the model's own memory as an invalid source. Testing confirmed
this was necessary: only after that clause did the bot reliably decline
questions like the Catan price, where it demonstrably knows the real answer.
```
