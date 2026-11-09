# -*- coding: utf-8 -*-
"""
ai_examples.py -- example retrieval for the alert router (retrieval-augmented prompting).

HYBRID (v18, the industry default and the same design as AI BOaT's journal search):
  keyword side   = character-trigram TF-IDF cosine -> exact words: NYKAA, low, high, 0.2%
  meaning side   = gemini-embedding-001 (768 dims)  -> same meaning in other words: 'slips under' = 'breaks'
  merge          = Reciprocal Rank Fusion (RRF, k=60) -> best 5 examples in front of the sentence
Example embeddings are computed ONCE (cached in data/ai_examples_emb.json, refreshed when the case file
changes); each alert costs ONE embedding call. Any embedding failure -> keyword side alone.

Every alert sentence gets the K most similar SOLVED sentences (sentence -> correct rule) from
evals/alert_cases.jsonl placed in front of it, so Gemini copies a worked pattern instead of
guessing ("nykaa break todays low" sits next to "SBIN breaks yesterday low" and "any stock new
day low"). Adding a real sentence to the case file improves the bot with no code change.

Similarity = character-trigram TF-IDF cosine, in plain Python: no extra API call, no quota, no
new dependency, and it tolerates typos and romanised Gujarati ("todays" / "today's", "tute").
The code guards in ai_alerts still check the model's answer afterwards -- retrieval makes a
mistake rarer, the guards make a mistake harmless.
"""
import hashlib
import json
import math
import os
import re
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "evals", "alert_cases.jsonl")
K = 5
POOL = 10                       # each side proposes 10, RRF keeps 5
RRF_K = 60
EMBED = None                    # callable(list[str]) -> list[list[float]]; set by ai_chat (None = keyword only)
EMB_CACHE = os.path.join(HERE, "data", "ai_examples_emb.json")
_emb = {"key": None, "vecs": {}, "error": None}

_lock = threading.Lock()
_index = {"mtime": None, "docs": [], "idf": {}, "vecs": []}


def _grams(text):
    t = " " + re.sub(r"[^a-z0-9%.&]+", " ", (text or "").lower()).strip() + " "
    return [t[i:i + 3] for i in range(len(t) - 2)]


def _vec(text, idf):
    tf = {}
    for g in _grams(text):
        tf[g] = tf.get(g, 0) + 1
    v = {g: c * idf.get(g, 0.0) for g, c in tf.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {g: x / n for g, x in v.items()}


def _load(path):
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return []
    with _lock:
        if _index["mtime"] == mt:
            return _index
        docs = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("tool") in ("create_alert", "cannot_alert", "update_alert") and c.get("args") is not None:
                body = re.sub(r"^\s*alert\s*(:|me\b|-)\s*", "", c["q"], flags=re.I)
                docs.append((body, c["tool"], c["args"]))
        df = {}
        for body, _t, _a in docs:
            for g in set(_grams(body)):
                df[g] = df.get(g, 0) + 1
        n = len(docs) or 1
        idf = {g: math.log((1 + n) / (1 + d)) + 1 for g, d in df.items()}
        _index.update(mtime=mt, docs=docs, idf=idf, vecs=[_vec(b, idf) for b, _t, _a in docs])
        return _index


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def _example_vectors(docs, cache_path=None):
    """{sentence: vector} for every example: from the cache file, embedding only new sentences (one batch)."""
    if EMBED is None:
        return {}
    cache_path = cache_path or EMB_CACHE             # read at call time: evals point it elsewhere
    with _lock:
        if not _emb["vecs"] and os.path.exists(cache_path):
            try:
                _emb["vecs"] = json.load(open(cache_path, encoding="utf-8"))
            except Exception:
                _emb["vecs"] = {}
        missing = [b for b, _t, _a in docs if _key(b) not in _emb["vecs"]]
    if missing:
        try:
            vecs = EMBED(missing)
            with _lock:
                for b, v in zip(missing, vecs):
                    _emb["vecs"][_key(b)] = list(v)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                tmp = cache_path + ".tmp"
                json.dump(_emb["vecs"], open(tmp, "w", encoding="utf-8"))
                os.replace(tmp, cache_path)
            _emb["error"] = None
        except Exception as e:
            _emb["error"] = "%s: %s" % (type(e).__name__, str(e)[:80])
    with _lock:
        return {b: _emb["vecs"][_key(b)] for b, _t, _a in docs if _key(b) in _emb["vecs"]}


def similar(text, k=K, path=CASES, exclude=None):
    """[(sentence, tool, args)] most like `text` — keyword and meaning ranks merged with RRF.
    `exclude` = a sentence to leave out (evals: a case may not see its own answer)."""
    ix = _load(path)
    if not ix or not ix["docs"]:
        return []
    keep = [i for i, (body, _t, _a) in enumerate(ix["docs"])
            if not (exclude and body.strip().lower() == exclude.strip().lower())]
    q = _vec(text, ix["idf"])
    kw = sorted(((sum(x * ix["vecs"][i].get(g, 0.0) for g, x in q.items()), i) for i in keep), reverse=True)
    kw = [i for sc, i in kw[:POOL] if sc > 0.05]
    sem = []
    vecs = _example_vectors([ix["docs"][i] for i in keep])
    if vecs:
        try:
            qv = EMBED([text])[0]
            scored = sorted(((_cos(qv, vecs[ix["docs"][i][0]]), i) for i in keep if ix["docs"][i][0] in vecs),
                            reverse=True)
            sem = [i for _sc, i in scored[:POOL]]
            _emb["error"] = None
        except Exception as e:
            _emb["error"] = "%s: %s" % (type(e).__name__, str(e)[:80])
            sem = []
    score = {}
    for ranks in (kw, sem):
        for r, i in enumerate(ranks):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
    best = sorted(score, key=lambda i: -score[i])[:k]
    return [ix["docs"][i] for i in best]


def status():
    """For /check: is the meaning side working?"""
    if EMBED is None:
        return "keyword only (no embedder)"
    return ("hybrid (keyword + embedding)" if not _emb["error"] else "keyword only — embedding error: " + _emb["error"])


def gemini_embedder(client, model="gemini-embedding-001"):
    """EMBED for ai_chat: one call per batch, 768 dims, 8 s timeout via the client."""
    from google.genai import types

    def embed(texts):
        r = client.models.embed_content(model=model, contents=list(texts),
                                        config=types.EmbedContentConfig(output_dimensionality=768,
                                                                        task_type="SEMANTIC_SIMILARITY"))
        return [list(e.values) for e in r.embeddings]
    return embed


def prompt_with_examples(text, k=K, exclude=None):
    ex = similar(text, k, exclude=exclude)
    if not ex:
        return text
    lines = ["SOLVED EXAMPLES (copy the pattern, not the stock names):"]
    for b, t, a in ex:
        lines.append("%r -> %s %s" % (b, t, json.dumps(a, ensure_ascii=False)))
    return "\n".join(lines) + "\nNEW: " + text
