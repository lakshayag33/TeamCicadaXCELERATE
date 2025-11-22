# rag_engine.py
# Robust RAG for textbook-style QA
# - Better chunking (250/120 default)
# - BGE embeddings + "query: " prefix
# - FAISS cosine index
# - Semantic re-ranking + definition keyword boost
# - Page/source metadata preserved

import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 120,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        debug: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.dimension: int | None = None

        if self.debug:
            print(f"[RAG] Loading embedding model: {embed_model_name} ...")
        self.embedding_model = SentenceTransformer(embed_model_name)
        if self.debug:
            print("[RAG] Embedding model loaded.")

        self._query_prefix = "query: "  # BGE convention

    # -----------------------------
    # PDF extraction
    # -----------------------------
    def _extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        if self.debug:
            print(f"[RAG] Extracting text from {pdf_path} ...")
        pages: List[Tuple[int, str]] = []
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc, start=1):
                pages.append((i, page.get_text()))
                if self.debug and (i % 10 == 0):
                    print(f"[RAG] Processed {i}/{len(doc)} pages")
            doc.close()
            if self.debug:
                total_chars = sum(len(t) for _, t in pages)
                print(f"[RAG] Done. {len(pages)} page(s), {total_chars} chars.")
        except Exception as e:
            print(f"[RAG ERROR] Failed to read {pdf_path}: {e}")
        return pages

    # -----------------------------
    # Chunking (word-based)
    # -----------------------------
    def _chunk_pages(self, pages: List[Tuple[int, str]], source: str) -> List[Dict[str, Any]]:
        words_with_page: List[Tuple[str, int]] = []
        for page_no, text in pages:
            for w in text.split():
                words_with_page.append((w, page_no))

        chunks: List[Dict[str, Any]] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        total_words = len(words_with_page)
        if self.debug:
            print(f"[RAG] Chunking words total={total_words}, size={self.chunk_size}, overlap={self.chunk_overlap}")

        i = 0
        while i < total_words:
            window = words_with_page[i : i + self.chunk_size]
            if not window:
                break

            text = " ".join(w for w, _ in window).strip()
            if text:
                page_start = window[0][1]
                page_end = window[-1][1]
                chunks.append({
                    "text": text,
                    "meta": {
                        "source": source,
                        "page_start": page_start,
                        "page_end": page_end,
                        "word_start": i,
                        "word_end": min(i + self.chunk_size, total_words),
                    }
                })
            i += step

        if self.debug:
            print(f"[RAG] Created {len(chunks)} chunk(s)")
        return chunks

    # -----------------------------
    # Public: process PDFs
    # -----------------------------
    def process_pdfs(self, pdf_paths: List[str]):
        self.chunks.clear()
        for pdf in pdf_paths:
            pages = self._extract_pages(pdf)
            self.chunks.extend(self._chunk_pages(pages, source=pdf))

        if not self.chunks:
            raise ValueError("[RAG] No text extracted from PDFs.")

        if self.debug:
            print(f"[RAG] Embedding {len(self.chunks)} chunk(s) ...")
        texts = [c["text"] for c in self.chunks]
        X = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        faiss.normalize_L2(X)
        self.embeddings = X
        self.dimension = X.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(X)

        if self.debug:
            print(f"[RAG] FAISS index built — vectors: {self.index.ntotal}, dim: {self.dimension}")

    # -----------------------------
    # Retrieval helpers
    # -----------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        q_text = self._query_prefix + query.strip()
        q_vec = self.embedding_model.encode([q_text], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        return q_vec

    def _keyword_boost(self, text: str, query: str) -> float:
        t = text.lower()
        q = query.lower()
        keywords = [
            "is the process",
            "is defined as",
            "refers to",
            "process of",
            "means",
            "called",
            "definition",
        ]
        domain = [
            "stigma", "anther", "pollen", "gamete", "zygote",
            "binary fission", "multiple fission", "cell divides"
        ]
        score = 0.0
        score += sum(1 for k in keywords if k in t) * 0.25
        score += sum(1 for k in domain if k in t) * 0.15
        for token in q.split():
            if token and token in t:
                score += 0.05
        return score

    def _rerank(self, query_vec: np.ndarray, results: List[Dict[str, Any]], alpha: float = 0.85) -> List[Dict[str, Any]]:
        if not results:
            return results
        docs = [r["text"] for r in results]
        D = self.embedding_model.encode(docs, convert_to_numpy=True)
        faiss.normalize_L2(D)
        sims = (D @ query_vec.T).squeeze(-1)

        reranked = []
        for sim, r in zip(sims, results):
            boost = self._keyword_boost(r["text"], r.get("query", ""))
            combined = float(alpha * float(sim) + (1 - alpha) * boost)
            rr = dict(r)
            rr["sim"] = float(sim)
            rr["boost"] = float(boost)
            rr["combined"] = combined
            reranked.append(rr)

        reranked.sort(key=lambda x: x["combined"], reverse=True)
        return reranked

    # -----------------------------
    # Public: retrieve
    # -----------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.embeddings is None or not self.chunks:
            print("[RAG ERROR] Index/chunks not available. Call process_pdfs() first.")
            return []

        if self.debug:
            print(f"\n[RAG] Retrieving for: {query}")

        q_vec = self._encode_query(query)
        sims, idxs = self.index.search(q_vec, top_k)
        sims = sims[0]
        idxs = idxs[0]

        initial = []
        for s, i in zip(sims, idxs):
            if 0 <= i < len(self.chunks):
                initial.append({
                    "text": self.chunks[i]["text"],
                    "score": float(s),
                    "meta": self.chunks[i]["meta"],
                    "query": query
                })

        ranked = self._rerank(q_vec, initial, alpha=0.85)

        if self.debug:
            for j, r in enumerate(ranked, start=1):
                m = r["meta"]
                print(f"[RAG] {j:02d}) ip={r['score']:.3f} sim={r['sim']:.3f} boost={r['boost']:.3f} "
                      f"combined={r['combined']:.3f} pages={m['page_start']}-{m['page_end']} src={m['source']}")
        return ranked

    # -----------------------------
    # Build final context
    # -----------------------------
    def build_context(self, query: str, top_k: int = 5, max_chars: int = 3000):
        ranked = self.retrieve(query, top_k=top_k)
        seen = set()
        parts: List[str] = []
        used = []
        total = 0

        for r in ranked:
            t = r["text"].strip()
            if t in seen:
                continue
            if total + len(t) + 2 > max_chars:
                break
            parts.append(t)
            used.append(r)
            seen.add(t)
            total += len(t) + 2

        context = "\n\n".join(parts)
        if self.debug:
            print(f"[RAG] Context length: {len(context)} from {len(parts)} chunk(s)")
        return context, used

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self):
        if self.debug:
            print("[RAG] Resetting engine ...")
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.dimension = None
        if self.debug:
            print("[RAG] Reset complete.")