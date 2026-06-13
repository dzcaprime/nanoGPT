import json
import math
import os
import re
from collections import Counter


INDEX_FILE = "index.json"
SUPPORTED_EXTENSIONS = {".md", ".txt"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def discover_corpus_paths(paths):
    files = []
    for raw_path in paths:
        path = os.path.abspath(raw_path)
        if os.path.isfile(path):
            if _is_supported(path):
                files.append(path)
        elif os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in names:
                    file_path = os.path.join(root, name)
                    if _is_supported(file_path):
                        files.append(file_path)
        else:
            raise FileNotFoundError(raw_path)
    return sorted(files)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(path, text, chunk_chars, chunk_overlap):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_chars")

    chunks = []
    step = chunk_chars - chunk_overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end]
        if chunk:
            chunks.append({"source": path, "start": start, "end": end, "text": chunk})
        if end == len(text):
            break
    return chunks


def tokenize(text):
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def build_index(paths, chunk_chars=1200, chunk_overlap=150):
    chunks = []
    doc_freqs = Counter()
    for path in discover_corpus_paths(paths):
        source = _display_path(path)
        for chunk in chunk_text(source, read_text(path), chunk_chars, chunk_overlap):
            terms = tokenize(chunk["text"])
            chunk["id"] = len(chunks)
            chunk["terms"] = terms
            chunks.append(chunk)
            doc_freqs.update(set(terms))

    return {
        "version": 1,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "doc_freqs": dict(sorted(doc_freqs.items())),
        "chunks": chunks,
    }


def save_index(index, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, INDEX_FILE)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index_path


def load_index(index_dir):
    with open(os.path.join(index_dir, INDEX_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(index, query, top_k=3, max_chars=3000):
    query_terms = tokenize(query)
    if not query_terms or top_k <= 0 or max_chars <= 0:
        return []

    num_chunks = index["num_chunks"]
    doc_freqs = index["doc_freqs"]
    scored = []
    for chunk in index["chunks"]:
        counts = Counter(chunk["terms"])
        score = 0.0
        for term in query_terms:
            term_count = counts.get(term, 0)
            if term_count:
                score += term_count * _idf(num_chunks, doc_freqs.get(term, 0))
        if score > 0:
            result = {key: value for key, value in chunk.items() if key != "terms"}
            result["score"] = score
            scored.append(result)

    selected = []
    total_chars = 0
    for chunk in sorted(scored, key=lambda item: (-item["score"], item["id"])):
        remaining = max_chars - total_chars
        if remaining <= 0 or len(selected) >= top_k:
            break
        if len(chunk["text"]) > remaining:
            if selected:
                break
            chunk = dict(chunk)
            chunk["text"] = chunk["text"][:remaining]
        selected.append(chunk)
        total_chars += len(chunk["text"])
    return selected


def _is_supported(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def _display_path(path):
    rel_path = os.path.relpath(path)
    return rel_path if not rel_path.startswith("..") else path


def _idf(num_chunks, doc_freq):
    return math.log((num_chunks + 1) / (doc_freq + 1)) + 1
