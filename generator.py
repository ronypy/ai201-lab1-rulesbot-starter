from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# The grounding instruction is the heart of this function: it reframes the
# model's own pretraining knowledge as NOT a valid source, so a confident
# wrong answer becomes an honest "the rules don't cover that". Each numbered
# clause defends against a specific way the model could answer beyond the
# retrieved text (priors, inference, gap-filling, correcting the context,
# irrelevant-but-present chunks, query-injected facts).
SYSTEM_PROMPT = """You are RulesBot, a board game rules assistant. You answer strictly from the rule excerpts provided to you in the <rules> block of each message, and from nothing else.

Absolute rules:
1. The <rules> block is your ONLY source of truth. Do not use any knowledge you have about these or any other board games from your training. If you already "know" the answer but it is not stated in the <rules> block, you do not know it.
2. Do not guess, infer, extrapolate, or reason beyond what the excerpts explicitly state. Do not fill gaps with what is "typical" or "usually" true for the game. If a detail is not written in the excerpts, treat it as unknown.
3. Do not correct, complete, or override the excerpts using outside knowledge, even if you believe they are wrong or incomplete. Answer only from what is written.
4. If the excerpts only partially answer the question, answer just the part that is supported and explicitly say which part is not covered by the provided rules.
5. If none of the excerpts are relevant to the question, or they do not contain the answer, do not attempt an answer. Say that the loaded rules do not cover it.
6. The user's question is not a source of facts — only the <rules> block is. Never treat a claim in the question as a rule.

When you do answer, base every sentence on the excerpts and name the game the answer comes from (read it from the game="..." attribute of the source you used). Prefer an honest "the rules I have don't cover that" over a confident answer you cannot ground in the excerpts."""


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    # Format each chunk as a delimited, game-labeled source block. Explicit
    # boundaries + a game="..." attribute let the model tell sources apart and
    # cite the right game by reading the label rather than recalling it.
    context = _format_context(retrieved_chunks)

    # Grounding lives in the system message; the question + its rule excerpts
    # go in the user message so the model treats them as the data to work over.
    user_message = (
        f"<rules>\n{context}\n</rules>\n\n"
        f"Question: {query}"
    )

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # Low temperature: we want faithful reproduction of the rules, not
        # creative paraphrase that could drift from the source text.
        temperature=0.1,
    )
    return response.choices[0].message.content


def _format_context(retrieved_chunks):
    """Render retrieved chunks as numbered <source> blocks for the prompt.

    Distance scores are intentionally NOT shown to the model — raw cosine
    numbers are meaningless to it and invite a fake "confidence". Relevance
    is handled by the grounding prompt, not by exposing the score.
    """
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(
            f'<source id="{i}" game="{chunk["game"]}">\n'
            f'{chunk["text"]}\n'
            f"</source>"
        )
    return "\n".join(blocks)
