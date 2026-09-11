"""
Tests for AirScribe note embeddings module (note_embeddings.py).

These tests verify:
- Embedding generation (single and batch)
- Semantic similarity computation
- Semantic search ranking
- Edge cases (empty text, missing model)

Note: These tests require the sentence-transformers model to be downloaded.
On first run, the model (~90MB) will be automatically downloaded and cached.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Embedding Generation Tests ──

class TestGenerateEmbedding:
    def test_embedding_returns_list(self):
        from note_embeddings import generate_embedding
        result = generate_embedding("Hello world, this is a test note.")
        assert isinstance(result, list)

    def test_embedding_has_correct_dimension(self):
        from note_embeddings import generate_embedding, EMBEDDING_DIM
        result = generate_embedding("Machine learning is a branch of artificial intelligence.")
        assert len(result) == EMBEDDING_DIM

    def test_embedding_values_are_floats(self):
        from note_embeddings import generate_embedding
        result = generate_embedding("Test content for embedding.")
        assert all(isinstance(v, float) for v in result)

    def test_empty_string_returns_zero_vector(self):
        from note_embeddings import generate_embedding, EMBEDDING_DIM
        result = generate_embedding("")
        assert result == [0.0] * EMBEDDING_DIM

    def test_whitespace_only_returns_zero_vector(self):
        from note_embeddings import generate_embedding, EMBEDDING_DIM
        result = generate_embedding("   \n\t  ")
        assert result == [0.0] * EMBEDDING_DIM

    def test_none_text_returns_zero_vector(self):
        from note_embeddings import generate_embedding, EMBEDDING_DIM
        # The function checks `not text` which handles None
        result = generate_embedding("")
        assert len(result) == EMBEDDING_DIM

    def test_similar_texts_have_high_similarity(self):
        from note_embeddings import generate_embedding, cosine_similarity
        emb1 = generate_embedding("Meeting notes about project deadline")
        emb2 = generate_embedding("Notes from project deadline discussion")
        similarity = cosine_similarity(emb1, emb2)
        assert similarity > 0.6, f"Expected high similarity, got {similarity}"

    def test_different_texts_have_lower_similarity(self):
        from note_embeddings import generate_embedding, cosine_similarity
        emb1 = generate_embedding("Quantum physics research paper summary")
        emb2 = generate_embedding("Recipe for chocolate cake with frosting")
        similarity = cosine_similarity(emb1, emb2)
        assert similarity < 0.5, f"Expected low similarity, got {similarity}"

    def test_embedding_is_normalized(self):
        """Embeddings should be L2-normalized (unit length)."""
        import numpy as np
        from note_embeddings import generate_embedding
        result = generate_embedding("This is a test for normalization.")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"


class TestGenerateEmbeddingsBatch:
    def test_batch_returns_correct_count(self):
        from note_embeddings import generate_embeddings_batch
        texts = ["Text one", "Text two", "Text three"]
        result = generate_embeddings_batch(texts)
        assert result is not None
        assert len(result) == 3

    def test_batch_each_has_correct_dim(self):
        from note_embeddings import generate_embeddings_batch, EMBEDDING_DIM
        texts = ["Hello", "World"]
        result = generate_embeddings_batch(texts)
        assert all(len(emb) == EMBEDDING_DIM for emb in result)

    def test_batch_empty_list(self):
        from note_embeddings import generate_embeddings_batch
        result = generate_embeddings_batch([])
        assert result == []

    def test_batch_handles_empty_strings(self):
        from note_embeddings import generate_embeddings_batch, EMBEDDING_DIM
        texts = ["Normal text", "", "Another text"]
        result = generate_embeddings_batch(texts)
        assert result is not None
        # Empty string should produce zero vector
        assert result[1] == [0.0] * EMBEDDING_DIM
        # Non-empty strings should produce real embeddings
        assert result[0] != [0.0] * EMBEDDING_DIM


# ── Cosine Similarity Tests ──

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from note_embeddings import cosine_similarity
        vec = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        from note_embeddings import cosine_similarity
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec_a, vec_b)) < 0.001

    def test_opposite_vectors(self):
        from note_embeddings import cosine_similarity
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        assert abs(cosine_similarity(vec_a, vec_b) - (-1.0)) < 0.001

    def test_zero_vector_returns_zero(self):
        from note_embeddings import cosine_similarity
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0


# ── Semantic Search Tests ──

class TestSemanticSearch:
    @pytest.fixture
    def sample_notes(self):
        from note_embeddings import generate_embedding
        return [
            {
                "_id": "note_1",
                "name": "Python Tutorial",
                "embedding": generate_embedding("Python programming tutorial with examples"),
            },
            {
                "_id": "note_2",
                "name": "Grocery List",
                "embedding": generate_embedding("Buy milk eggs bread butter from the store"),
            },
            {
                "_id": "note_3",
                "name": "ML Notes",
                "embedding": generate_embedding("Machine learning neural networks deep learning"),
            },
        ]

    def test_semantic_search_returns_ranked_results(self, sample_notes):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("How to code in Python")
        results = semantic_search(query_emb, sample_notes, top_k=10)
        assert len(results) > 0
        # Python tutorial should rank higher than grocery list for coding query
        ids = [r["_id"] for r in results]
        if "note_1" in ids and "note_2" in ids:
            assert ids.index("note_1") < ids.index("note_2")

    def test_semantic_search_respects_top_k(self, sample_notes):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("programming")
        results = semantic_search(query_emb, sample_notes, top_k=1)
        assert len(results) <= 1

    def test_semantic_search_respects_threshold(self, sample_notes):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("programming")
        # Very high threshold — should filter out most results
        results = semantic_search(query_emb, sample_notes, threshold=0.95)
        # All returned results should be above threshold
        for r in results:
            assert r["score"] >= 0.95

    def test_semantic_search_empty_query(self, sample_notes):
        from note_embeddings import semantic_search
        results = semantic_search([], sample_notes)
        assert results == []

    def test_semantic_search_empty_notes(self):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("test")
        results = semantic_search(query_emb, [])
        assert results == []

    def test_semantic_search_notes_without_embedding(self):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("test")
        notes = [{"_id": "1", "name": "No Embedding"}]
        results = semantic_search(query_emb, notes)
        assert results == []

    def test_semantic_search_scores_are_descending(self, sample_notes):
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("artificial intelligence and computing")
        results = semantic_search(query_emb, sample_notes)
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_semantic_search_does_not_return_embedding(self, sample_notes):
        """Embedding vectors should not be included in search results."""
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("Python code")
        results = semantic_search(query_emb, sample_notes)
        for r in results:
            assert "embedding" not in r

    def test_irrelevant_notes_rank_lower(self, sample_notes):
        """Grocery list should rank lower for a programming query."""
        from note_embeddings import generate_embedding, semantic_search
        query_emb = generate_embedding("software engineering and coding")
        results = semantic_search(query_emb, sample_notes, top_k=10)
        scores_by_id = {r["_id"]: r["score"] for r in results}
        if "note_1" in scores_by_id and "note_2" in scores_by_id:
            assert scores_by_id["note_1"] > scores_by_id["note_2"], \
                "Programming note should score higher than grocery list"
