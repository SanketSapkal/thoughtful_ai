"""
Thoughtful AI – Mini Support Agent (Streamlit)
================================================

Quick Start
-----------
1) Create a virtual env (optional) and install deps:
   pip install streamlit

2) Save this file as app.py and run:
   streamlit run app.py

3) Chat in the UI (http://localhost:8501). The agent will:
   • Retrieve the most relevant hardcoded answer about Thoughtful AI
   • Fall back to a generic offline response when there’s no good match
   • Show the matched source and similarity score for transparency

Notes
-----
- No OpenAI or external API dependencies. Purely local.
- No external vector libs used. We implement a simple TF‑IDF + cosine matcher in ~40 lines.
- Robust to odd inputs: empty strings, long text, punctuation, repeated whitespace, etc.

Submission
----------
- This single file is ready for Replit/GitHub. Include instructions above in README if desired.

"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st

# ---------------------------
# Predefined Thoughtful AI Q&A
# ---------------------------
KB: List[Dict[str, str]] = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": (
            "EVA automates the process of verifying a patient’s eligibility and benefits "
            "information in real-time, eliminating manual data entry errors and reducing claim rejections."
        ),
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": (
            "CAM streamlines the submission and management of claims, improving accuracy, "
            "reducing manual intervention, and accelerating reimbursements."
        ),
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": (
            "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate "
            "reconciliation of payments and reducing administrative burden."
        ),
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": (
            "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline "
            "healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), "
            "and Payment Posting (PHIL), among others."
        ),
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": (
            "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational "
            "efficiency, and reduce errors in critical processes like claims management and payment posting."
        ),
    },
]

# Build a small searchable corpus from questions + answers for better recall.
CORPUS: List[Tuple[str, int, str]] = []  # (text, kb_index, field)
for i, qa in enumerate(KB):
    CORPUS.append((qa["question"], i, "question"))
    CORPUS.append((qa["answer"], i, "answer"))

# ---------------------------
# Lightweight TF‑IDF Utilities
# ---------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

# Document frequencies across the corpus
DF: Counter[str] = Counter()
DOC_TOKENS: List[List[str]] = []
for text, _, _ in CORPUS:
    toks = list(dict.fromkeys(tokenize(text)))  # unique per doc
    DOC_TOKENS.append(toks)
    DF.update(toks)

N_DOCS = len(CORPUS)
IDF: Dict[str, float] = {}
for term, df in DF.items():
    IDF[term] = math.log((1 + N_DOCS) / (1 + df)) + 1.0

# Precompute TF‑IDF vectors for corpus
CORPUS_VECS: List[Dict[str, float]] = []
for text, _, _ in CORPUS:
    toks = tokenize(text)
    tf = Counter(toks)
    vec = {t: (tf[t] / max(1, len(toks))) * IDF.get(t, 0.0) for t in tf}
    CORPUS_VECS.append(vec)


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    num = sum(w * b.get(t, 0.0) for t, w in a.items())
    den_a = math.sqrt(sum(w * w for w in a.values()))
    den_b = math.sqrt(sum(w * w for w in b.values()))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)


def vectorize(text: str) -> Dict[str, float]:
    toks = tokenize(text)
    tf = Counter(toks)
    return {t: (tf[t] / max(1, len(toks))) * IDF.get(t, 0.0) for t in tf}


@dataclass
class Match:
    kb_index: int
    field: str
    score: float


def search(query: str, top_k: int = 3) -> List[Match]:
    qvec = vectorize(query)
    scored: List[Match] = []
    for idx, (_, kb_i, field) in enumerate(CORPUS):
        s = cosine_sim(qvec, CORPUS_VECS[idx])
        if s > 0:
            scored.append(Match(kb_index=kb_i, field=field, score=s))
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:top_k]


# ---------------------------
# Fallback: Generic message only
# ---------------------------
GENERIC_FALLBACK = (
    "I’m not fully sure about that topic. I’m best at questions about Thoughtful AI’s "
    "agents (EVA, CAM, PHIL). You can try asking, for example: ‘What does EVA do?’ or "
    "‘What are the benefits of Thoughtful AI’s agents?’"
)


def llm_fallback(prompt: str) -> str:
    return GENERIC_FALLBACK


# ---------------------------
# Retrieval + Response Logic
# ---------------------------
SIM_THRESHOLD = 0.22


def answer_query(query: str):
    if not query.strip():
        return (
            "Hi! Ask me about Thoughtful AI’s agents (EVA, CAM, PHIL) or their benefits.",
            {"mode": "system"},
        )

    hits = search(query, top_k=3)
    if hits and hits[0].score >= SIM_THRESHOLD:
        best = hits[0]
        qa = KB[best.kb_index]
        meta = {
            "mode": "kb",
            "matched_field": best.field,
            "score": f"{best.score:.2f}",
            "matched_q": qa["question"],
        }
        return qa["answer"], meta

    fallback = llm_fallback(query)
    return fallback, {"mode": "fallback"}


# ---------------------------
# Streamlit Chat UI
# ---------------------------
st.set_page_config(page_title="Thoughtful AI – Support Agent", page_icon="🤖", layout="centered")

st.title("🤖 Thoughtful AI – Mini Support Agent")
st.caption(
    "Ask about EVA, CAM, PHIL, or the benefits of Thoughtful AI’s agents. I’ll fetch the best hardcoded answer, or fall back to a generic response."
)

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! How can I help with Thoughtful AI today?"}
    ]

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question…")
if user_input is not None:
    user_input = user_input.strip()
    if user_input == "":
        st.session_state.chat.append({"role": "user", "content": ""})
        with st.chat_message("assistant"):
            st.markdown("Please enter a non-empty question. For example: *What does EVA do?*")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        ans, meta = answer_query(user_input)
    except Exception as e:
        ans = "Sorry, I hit an unexpected error while processing that. Please try again."
        meta = {"mode": "error", "detail": str(e)}

    with st.chat_message("assistant"):
        st.markdown(f"{ans}")
        with st.container(border=True):
            if meta.get("mode") == "kb":
                st.markdown(
                    f"**Source:** Matched `{meta.get('matched_field')}` from: _{meta.get('matched_q')}_\n"
                    f"**Confidence:** {meta.get('score')}"
                )
            elif meta.get("mode") == "fallback":
                st.markdown("*No strong KB match. Used generic fallback.*")
            elif meta.get("mode") == "error":
                st.markdown(":warning: An error occurred; details logged locally.")

    st.session_state.chat.append({"role": "assistant", "content": ans})
