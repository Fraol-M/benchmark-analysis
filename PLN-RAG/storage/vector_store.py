import uuid
import httpx
from collections import OrderedDict
from typing import Any, List, Tuple
from config import get_settings


class VectorStore:
    """
    Manages NL ↔ PLN atom mappings in Qdrant.

    Stores: { nl: sentence, pln: [atoms] } per ingested sentence.
    Retrieves: relevant PLN atoms to use as parser context.
    """

    def __init__(self):
        cfg = get_settings()
        self._qdrant = cfg.qdrant_url
        self._ollama = cfg.ollama_url
        self._ollama_model = cfg.ollama_model
        self._collection = cfg.qdrant_collection
        self._predicate_collection = cfg.predicate_mapping_collection
        self._client = httpx.Client(timeout=30)
        self._collection_sizes: dict[str, int] = {}
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embedding_cache_size = 256

    def embed(self, text: str) -> List[float]:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            self._embedding_cache.move_to_end(text)
            return cached
        resp = self._client.post(self._ollama, json={
            "model": self._ollama_model,
            "prompt": text
        })
        resp.raise_for_status()
        vector = resp.json()["embedding"]
        self._cache_embedding(text, vector)
        return vector

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed uncached texts in one Ollama request when possible."""
        if not texts:
            return []
        missing = list(dict.fromkeys(
            text for text in texts if text not in self._embedding_cache
        ))
        if len(missing) == 1:
            vector = self.embed(missing[0])
            if missing[0] not in self._embedding_cache:
                self._cache_embedding(missing[0], vector)
        elif missing:
            batch_url = self._ollama.replace("/api/embeddings", "/api/embed")
            try:
                resp = self._client.post(
                    batch_url,
                    json={"model": self._ollama_model, "input": missing},
                )
                resp.raise_for_status()
                vectors = resp.json().get("embeddings", [])
                if len(vectors) != len(missing):
                    raise ValueError("Ollama returned an incomplete embedding batch")
                for text, vector in zip(missing, vectors):
                    self._cache_embedding(text, vector)
            except Exception:
                # Older Ollama versions may not expose /api/embed.
                for text in missing:
                    vector = self.embed(text)
                    if text not in self._embedding_cache:
                        self._cache_embedding(text, vector)
        return [self._embedding_cache[text] for text in texts]

    def _cache_embedding(self, text: str, vector: List[float]) -> None:
        self._embedding_cache[text] = vector
        self._embedding_cache.move_to_end(text)
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)

    def _ensure_collection(self, collection: str, vector_size: int):
        if self._collection_sizes.get(collection) == vector_size:
            return
        try:
            self._client.get(
                f"{self._qdrant}/collections/{collection}"
            ).raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._client.put(
                    f"{self._qdrant}/collections/{collection}",
                    json={"vectors": {"size": vector_size, "distance": "Cosine"}}
                ).raise_for_status()
        self._collection_sizes[collection] = vector_size

    def store(
        self,
        sentence: str,
        atoms: List[str],
        vector: List[float],
        metadata: dict[str, Any] | None = None,
        query_targets: List[str] | None = None,
    ):
        self._ensure_collection(self._collection, len(vector))
        payload: dict[str, Any] = {
            "nl": sentence,
            "pln": atoms,
            "query_targets": query_targets or [],
        }
        if metadata:
            payload["metadata"] = metadata
        self._client.put(
            f"{self._qdrant}/collections/{self._collection}/points?wait=true",
            json={"points": [{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": payload
            }]}
        ).raise_for_status()

    def search(
        self,
        text: str,
        top_k: int,
        min_score: float | None = None,
    ) -> Tuple[List[dict[str, Any]], List[float]]:
        vector = self.embed(text)
        self._ensure_collection(self._collection, len(vector))

        resp = self._client.post(
            f"{self._qdrant}/collections/{self._collection}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True}
        )
        if resp.status_code != 200:
            return [], vector

        matches: List[dict[str, Any]] = []
        for item in resp.json().get("result", []):
            score = item.get("score", 0)
            if min_score is not None and score < min_score:
                continue
            payload = item.get("payload", {}) or {}
            matches.append(
                {
                    "score": score,
                    "nl": payload.get("nl", ""),
                    "pln": payload.get("pln", []),
                    "query_targets": payload.get("query_targets", []),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return matches, vector

    def retrieve_context(self, text: str, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Returns (context_atoms, embedding_vector).
        context_atoms: flat list of PLN atom strings from top-k similar sentences.
        """
        matches, vector = self.search(text, top_k=top_k)
        context: List[str] = []
        for item in matches:
            pln = item.get("pln", [])
            if isinstance(pln, list):
                context.extend(pln)

        return context, vector

    def upsert_predicate_cards(self, cards: List[dict[str, Any]]) -> None:
        """Embed and upsert predicate cards into a dedicated Qdrant collection."""
        prepared: List[tuple[dict[str, Any], str, str]] = []
        for card in cards:
            key = str(card.get("key", "")).strip()
            predicate = str(card.get("predicate", "")).strip()
            if not key or not predicate:
                continue
            search_text = str(card.get("search_text", "")).strip()
            if not search_text:
                search_text = self._predicate_card_search_text(card)
            prepared.append((card, key, search_text))

        vectors = self.embed_many([item[2] for item in prepared])
        points: List[dict[str, Any]] = []
        for (card, key, _search_text), vector in zip(prepared, vectors):
            self._ensure_collection(self._predicate_collection, len(vector))
            points.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._predicate_collection}:{key}",
                        )
                    ),
                    "vector": vector,
                    "payload": dict(card),
                }
            )
        if not points:
            return
        self._client.put(
            f"{self._qdrant}/collections/{self._predicate_collection}"
            "/points?wait=true",
            json={"points": points},
        ).raise_for_status()

    def search_predicate_cards(
        self,
        search_text: str,
        *,
        arity: int,
        top_k: int,
        min_score: float,
    ) -> List[dict[str, Any]]:
        vector = self.embed(search_text)
        self._ensure_collection(self._predicate_collection, len(vector))
        resp = self._client.post(
            f"{self._qdrant}/collections/{self._predicate_collection}"
            "/points/search",
            json={
                "vector": vector,
                "limit": top_k,
                "with_payload": True,
                "filter": {
                    "must": [
                        {"key": "kind", "match": {"value": "predicate_card"}},
                        {"key": "arity", "match": {"value": int(arity)}},
                    ]
                },
            },
        )
        if resp.status_code != 200:
            return []
        results: List[dict[str, Any]] = []
        for item in resp.json().get("result", []):
            score = float(item.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            results.append(
                {
                    "id": item.get("id"),
                    "score": score,
                    "payload": item.get("payload", {}) or {},
                }
            )
        return results

    def _predicate_card_search_text(self, card: dict[str, Any]) -> str:
        explicit = str(card.get("search_text", "")).strip()
        if explicit:
            return explicit
        parts = [
            f"predicate: {card.get('predicate', '')}",
            f"labels: {' '.join(card.get('labels', []) or [])}",
            f"argument types: {' '.join(card.get('argument_types', []) or [])}",
            f"definition: {card.get('definition', '')}",
            f"evidence: {' '.join(card.get('examples', [])[:1] or [])[:180]}",
            f"atom: {' '.join(card.get('source_atoms', [])[:1] or [])}",
        ]
        return "; ".join(part for part in parts if part.split(":", 1)[-1].strip())

    def list_points(self, limit: int = 50) -> List[dict[str, Any]]:
        """
        Return raw Qdrant payloads for debugging.
        Vectors are intentionally omitted because they are large and not useful
        for normal inspection.
        """
        try:
            resp = self._client.post(
                f"{self._qdrant}/collections/{self._collection}/points/scroll",
                json={
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if resp.status_code != 200:
                return []
            points = resp.json().get("result", {}).get("points", [])
            return [
                {
                    "id": point.get("id"),
                    "payload": point.get("payload", {}),
                }
                for point in points
            ]
        except Exception:
            return []

    def list_predicate_points(self, limit: int = 50) -> List[dict[str, Any]]:
        return self._list_collection_points(self._predicate_collection, limit)

    def _list_collection_points(
        self,
        collection: str,
        limit: int,
    ) -> List[dict[str, Any]]:
        try:
            resp = self._client.post(
                f"{self._qdrant}/collections/{collection}/points/scroll",
                json={
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if resp.status_code != 200:
                return []
            points = resp.json().get("result", {}).get("points", [])
            return [
                {
                    "id": point.get("id"),
                    "payload": point.get("payload", {}),
                }
                for point in points
            ]
        except Exception:
            return []

    def reset(self):
        for collection in (self._collection, self._predicate_collection):
            try:
                self._client.delete(f"{self._qdrant}/collections/{collection}")
            except Exception:
                pass
        self._collection_sizes.clear()

    @property
    def count(self) -> int:
        return self._collection_count(self._collection)

    @property
    def predicate_count(self) -> int:
        return self._collection_count(self._predicate_collection)

    def _collection_count(self, collection: str) -> int:
        try:
            resp = self._client.get(f"{self._qdrant}/collections/{collection}")
            return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            return 0
