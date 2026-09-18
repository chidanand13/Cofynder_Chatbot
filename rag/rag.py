import os
import math
from openai import OpenAI

class SimpleRAG:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.chunks = []
        self.chunk_embeddings = []
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        # 1. Load text
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "cofynder.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 2. Split into small chunks (by double newline for simplicity)
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        self.chunks = raw_chunks

        # 3. Create embeddings for chunks
        if self.chunks:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-004",
                    input=self.chunks
                )
                self.chunk_embeddings = [data.embedding for data in response.data]
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                self.chunk_embeddings = [[0]*768 for _ in self.chunks] # Fallback if API fails

    def _cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def retrieve(self, query, top_k=2):
        if not self.chunks or not self.chunk_embeddings:
            return ""

        # 1. Embed query
        try:
            response = self.client.embeddings.create(
                model="text-embedding-004",
                input=[query]
            )
            query_embedding = response.data[0].embedding
        except Exception:
            return ""

        # 2. Find most relevant chunks
        scored_chunks = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = self._cosine_similarity(query_embedding, chunk_emb)
            scored_chunks.append((score, self.chunks[i]))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 3. Return top_k chunks
        top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
        return "\n\n".join(top_chunks)
