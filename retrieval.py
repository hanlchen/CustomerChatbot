"""
Hybrid knowledge retrieval for CustomerChatbot.

Answers policy questions by searching *sections* of documents rather than whole
documents, so a question about the return window gets the eligibility
paragraph instead of the first 400 characters of the return policy.

Three signals are combined:

1. BM25 lexical scoring over stemmed tokens -- precise on exact terminology.
2. Concept expansion -- "send it back" and "return" are the same question to a
   customer, and lexical search alone never connects them.
3. Dense embeddings -- optional. Enabled automatically when `model2vec` or
   `sentence-transformers` is installed and a model is cached locally.

Rankings are merged with Reciprocal Rank Fusion, which combines ordered lists
without needing their scores to be on the same scale.

Everything degrades gracefully: with no optional dependency installed this
module still works, using signals 1 and 2 only.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "of", "for", "to",
    "in", "on", "at", "by", "with", "from", "and", "or", "but", "if", "so",
    "as", "than", "then", "there", "here", "what", "which", "who", "whom",
    "how", "when", "where", "why", "can", "could", "would", "should", "will",
    "shall", "may", "might", "must", "please", "thanks", "thank", "just",
    "get", "got", "want", "need", "know", "tell", "about", "any", "some",
}

# Suffix rules, longest first. A full stemmer is overkill here and mangles
# domain words; these rules cover the plurals and gerunds that actually appear.
_SUFFIX_RULES: Tuple[Tuple[str, str], ...] = (
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

_IRREGULAR = {
    "policies": "policy",
    "shipping": "ship",
    "shipped": "ship",
    "returning": "return",
    "returned": "return",
    "returns": "return",
    "cancelled": "cancel",
    "canceled": "cancel",
    "cancellation": "cancel",
    "delivered": "deliver",
    "delivery": "deliver",
    "deliveries": "deliver",
    "damaged": "damage",
    "warranties": "warranty",
    "purchased": "purchase",
    "purchases": "purchase",
    "billing": "bill",
    "charges": "charge",
    "charged": "charge",
    "refunded": "refund",
    "refunds": "refund",
    "exchanges": "exchange",
    "exchanged": "exchange",
}


def stem(word: str) -> str:
    """Reduce a word to a crude stem so 'returns'/'returning' match 'return'."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, stem."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [stem(w) for w in words]


# ---------------------------------------------------------------------------
# Concept expansion
# ---------------------------------------------------------------------------

# Each group is a set of surface forms customers use for one underlying idea.
# A query term matching any member also matches the rest. This is what lets
# "can I send this back" reach the return policy without an embedding model.
CONCEPT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("return", "send back", "give back", "take back", "bring back", "refund",
     "money back", "reimburse", "send it back"),
    ("exchange", "swap", "replace", "different size", "different colour",
     "different color"),
    ("ship", "deliver", "postage", "freight", "courier", "mail", "send",
     "dispatch", "arrive", "arrival"),
    ("cost", "price", "fee", "charge", "much", "expensive", "cheap", "rate"),
    ("track", "where", "status", "locate", "find", "eta", "late", "delayed",
     "missing", "lost"),
    ("cancel", "stop", "call off", "abort", "scrap"),
    ("warranty", "guarantee", "cover", "covered", "coverage", "protection",
     "broken", "faulty", "defect", "defective", "stopped working", "not working"),
    ("damage", "damaged", "broken", "cracked", "smashed", "dented", "crushed"),
    ("pay", "payment", "card", "credit", "debit", "billing", "invoice",
     "charge", "checkout", "amex", "visa", "mastercard", "paypal"),
    ("discount", "coupon", "promo", "voucher", "code", "sale", "offer", "deal"),
    ("account", "login", "sign in", "password", "profile", "email address"),
    ("privacy", "data", "personal information", "gdpr", "delete my data"),
    ("international", "overseas", "abroad", "customs", "duty", "import"),
    ("time", "long", "days", "week", "deadline", "window", "expire", "expiry"),
)


# Multi-word intents that single tokens can't capture. "send" alone is
# ambiguous (shipping or returning); "send it back" is not. Each pattern
# allows a couple of intervening words so "send this back" still matches.
PHRASE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bsend\s+(?:\w+\s+){0,2}back\b", "return"),
    (r"\bgive\s+(?:\w+\s+){0,2}back\b", "return"),
    (r"\btake\s+(?:\w+\s+){0,2}back\b", "return"),
    (r"\bship\s+(?:\w+\s+){0,2}back\b", "return"),
    (r"\bmoney\s+back\b", "refund"),
    (r"\bget\s+(?:my\s+)?(?:money|cash)\b", "refund"),
    (r"\bfull\s+refund\b", "refund"),
    (r"\b(?:stopped|not|isn'?t|doesn'?t)\s+work", "defective"),
    (r"\bdoesn'?t\s+fit\b", "exchange"),
    (r"\bwrong\s+(?:item|size|colou?r|product)\b", "exchange"),
    (r"\bnever\s+(?:arrived|came|showed)\b", "track"),
    (r"\bhasn'?t\s+(?:arrived|come|shown)\b", "track"),
)

# Deliberately NOT here: "how much", "how long", "where is". They are question
# forms, not topics -- the real subject is elsewhere in the sentence. Treating
# them as topic anchors injected the whole cost concept group (including
# "charge") at full weight, which pulled payment policies above shipping ones
# for "how much does express shipping cost?".


def phrase_concepts(query: str) -> List[str]:
    """Stemmed terms implied by multi-word phrases in the query."""
    lowered = (query or "").lower()
    found: List[str] = []
    for pattern, anchor in PHRASE_PATTERNS:
        if re.search(pattern, lowered):
            anchor_stem = stem(anchor)
            for term in CONCEPT_INDEX.get(anchor_stem, {anchor_stem}):
                if term not in found:
                    found.append(term)
    return found


def _build_concept_index() -> Dict[str, set]:
    """Map each stemmed term to the set of stemmed terms in its groups."""
    index: Dict[str, set] = {}
    for group in CONCEPT_GROUPS:
        stems = set()
        for phrase in group:
            stems.update(tokenize(phrase, drop_stopwords=False))
        for term in stems:
            index.setdefault(term, set()).update(stems)
    return index


CONCEPT_INDEX = _build_concept_index()


def expand(terms: Sequence[str]) -> List[str]:
    """Add related concept terms to a token list, preserving the originals."""
    expanded = list(terms)
    for term in terms:
        for related in CONCEPT_INDEX.get(term, ()):
            if related not in expanded:
                expanded.append(related)
    return expanded


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable passage."""
    chunk_id: str
    doc_id: str
    doc_type: str          # "policy" | "faq"
    title: str             # parent document title
    section: str           # section heading, or "" for the opening passage
    text: str
    category: str = ""
    tokens: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.title} — {self.section}" if self.section else self.title


def _split_sections(content: str) -> List[Tuple[str, str]]:
    """Split policy text into (heading, body) passages on blank lines.

    Policy bodies are written as an intro paragraph followed by labelled
    blocks ("Eligibility:", "Process:"), which makes them cleanly splittable.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content or "") if b.strip()]
    sections: List[Tuple[str, str]] = []
    for block in blocks:
        lines = block.split("\n")
        first = lines[0].strip()
        # A short line ending in ':' is a heading for the lines beneath it.
        if first.endswith(":") and len(first) < 60 and len(lines) > 1:
            sections.append((first.rstrip(":"), "\n".join(lines[1:]).strip()))
        else:
            sections.append(("", block))
    return sections


def build_corpus(policies: Iterable[Dict[str, Any]],
                 faqs: Iterable[Dict[str, Any]] = ()) -> List[Chunk]:
    """Turn policies and FAQs into retrievable chunks."""
    chunks: List[Chunk] = []

    for policy in policies:
        doc_id = str(policy.get("id", policy.get("title", "")))
        title = policy.get("title", "")
        category = policy.get("category", "")
        for index, (heading, body) in enumerate(_split_sections(policy.get("content", ""))):
            # Title and heading are part of what the chunk is "about".
            searchable = f"{title} {category} {heading} {body}"
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#s{index}",
                doc_id=doc_id,
                doc_type="policy",
                title=title,
                section=heading,
                text=body,
                category=category,
                tokens=tokenize(searchable),
            ))

    for faq in faqs:
        doc_id = str(faq.get("id", ""))
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        chunks.append(Chunk(
            chunk_id=f"faq:{doc_id}",
            doc_id=doc_id,
            doc_type="faq",
            title=question,
            section="",
            text=answer,
            category=faq.get("category", ""),
            tokens=tokenize(f"{question} {answer} {faq.get('category', '')}"),
        ))

    return chunks


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25:
    """Standard Okapi BM25 over pre-tokenized documents."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]],
                 k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.doc_lengths = [len(t) for t in corpus_tokens]
        self.avg_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        self.term_freqs: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}
        for tokens in corpus_tokens:
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Standard BM25 idf with the +1 that keeps common terms non-negative.
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        results = [0.0] * self.doc_count
        if not self.avg_length:
            return results
        for index, counts in enumerate(self.term_freqs):
            length = self.doc_lengths[index]
            total = 0.0
            for term in query_tokens:
                freq = counts.get(term)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += idf * (freq * (self.k1 + 1)) / denom
            results[index] = total
        return results


# ---------------------------------------------------------------------------
# Optional dense embeddings
# ---------------------------------------------------------------------------

class EmbeddingBackend:
    """Dense vectors, if a local model is available.

    Tries model2vec (static embeddings, numpy-only, ~30MB) then
    sentence-transformers. Absent either, the retriever runs lexical-only --
    this is an enhancement, never a requirement.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model = None
        self.kind = "none"
        self.error: Optional[str] = None
        if os.environ.get("CHATBOT_DISABLE_EMBEDDINGS", "").lower() in ("1", "true", "yes"):
            self.error = "disabled by CHATBOT_DISABLE_EMBEDDINGS"
            return
        self._load(model_name)

    def _load(self, model_name: Optional[str]) -> None:
        try:
            from model2vec import StaticModel  # type: ignore
            name = model_name or os.environ.get(
                "CHATBOT_EMBEDDING_MODEL", "minishlab/potion-base-8M"
            )
            self.model = StaticModel.from_pretrained(name)
            self.kind = f"model2vec:{name}"
            return
        except Exception as exc:  # not installed, or model not cached
            self.error = f"model2vec unavailable ({type(exc).__name__})"

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            name = model_name or os.environ.get(
                "CHATBOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )
            self.model = SentenceTransformer(name)
            self.kind = f"sentence-transformers:{name}"
            self.error = None
        except Exception as exc:
            self.error = f"{self.error}; sentence-transformers unavailable ({type(exc).__name__})"

    @property
    def available(self) -> bool:
        return self.model is not None

    def encode(self, texts: Sequence[str]):
        import numpy as np

        vectors = self.model.encode(list(texts))
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_type": self.chunk.doc_type,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "label": self.chunk.label,
            "category": self.chunk.category,
            "text": self.chunk.text,
            "score": round(self.score, 4),
        }


class HybridRetriever:
    """Combines BM25, concept expansion and optional embeddings via RRF."""

    RRF_K = 60  # standard constant; damps the influence of low ranks

    def __init__(self, chunks: Sequence[Chunk],
                 embedder: Optional[EmbeddingBackend] = None):
        self.chunks = list(chunks)
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.embedder = embedder if embedder is not None else EmbeddingBackend()
        self._matrix = None
        if self.embedder.available and self.chunks:
            try:
                self._matrix = self.embedder.encode(
                    [f"{c.title}. {c.section}. {c.text}" for c in self.chunks]
                )
            except Exception:
                self._matrix = None

    @property
    def backend(self) -> str:
        return self.embedder.kind if self._matrix is not None else "lexical-only"

    def _lexical_ranking(self, query: str) -> List[Tuple[int, float]]:
        base = tokenize(query)
        if not base:
            return []
        # A matched phrase states the intent outright, so its terms join the
        # primary pass rather than the discounted expansion pass.
        for term in phrase_concepts(query):
            if term not in base:
                base.append(term)

        # Original terms score at full weight; expansion terms are scored in a
        # second pass at a discount so a literal match still wins.
        primary = self.bm25.scores(base)
        expanded_terms = [t for t in expand(base) if t not in base]
        if expanded_terms:
            secondary = self.bm25.scores(expanded_terms)
            combined = [p + 0.45 * s for p, s in zip(primary, secondary)]
        else:
            combined = primary
        ranked = [(i, s) for i, s in enumerate(combined) if s > 0]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def _semantic_ranking(self, query: str) -> List[Tuple[int, float]]:
        if self._matrix is None:
            return []
        try:
            vector = self.embedder.encode([query])[0]
        except Exception:
            return []
        scores = self._matrix @ vector
        ranked = [(i, float(s)) for i, s in enumerate(scores)]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        # Cosine similarity is always positive-ish here; keep a sane cutoff.
        return [pair for pair in ranked if pair[1] > 0.15][:50]

    def search(self, query: str, top_k: int = 4,
               doc_type: Optional[str] = None) -> List[Hit]:
        """Return the most relevant passages for a natural-language query."""
        if not query or not query.strip():
            return []

        lexical = self._lexical_ranking(query)
        semantic = self._semantic_ranking(query)

        if not lexical and not semantic:
            return []

        lexical_rank = {idx: rank for rank, (idx, _) in enumerate(lexical, start=1)}
        semantic_rank = {idx: rank for rank, (idx, _) in enumerate(semantic, start=1)}

        fused: Dict[int, float] = {}
        for idx, rank in lexical_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
        for idx, rank in semantic_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)

        order = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

        hits: List[Hit] = []
        seen_sections = set()
        for idx, score in order:
            chunk = self.chunks[idx]
            if doc_type and chunk.doc_type != doc_type:
                continue
            # Avoid returning two near-identical passages from one document.
            key = (chunk.doc_id, chunk.section)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            hits.append(Hit(
                chunk=chunk,
                score=score,
                lexical_rank=lexical_rank.get(idx),
                semantic_rank=semantic_rank.get(idx),
            ))
            if len(hits) >= top_k:
                break

        # Drop weak tail results so a strong answer isn't padded with noise.
        if hits:
            cutoff = hits[0].score * 0.55
            hits = [h for h in hits if h.score >= cutoff]
        return hits


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Build (once) and return the shared retriever over the live database."""
    global _retriever
    if _retriever is None:
        from database import _db
        _retriever = HybridRetriever(build_corpus(_db.policies, _db.faqs))
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever (used by tests)."""
    global _retriever
    _retriever = None


def search_knowledge(query: str, top_k: int = 4,
                     doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience wrapper returning plain dictionaries."""
    return [hit.to_dict() for hit in get_retriever().search(query, top_k, doc_type)]
