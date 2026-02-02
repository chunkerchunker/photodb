"""Tests for prompt embedding cache."""

import pytest
from unittest.mock import MagicMock
import torch


class TestPromptCache:
    def test_load_category_embeddings(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="happy",
                prompt_text="happy scene",
                embedding=[0.1] * 512,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
            PromptEmbedding(
                id=2,
                category_id=1,
                label="sad",
                prompt_text="sad scene",
                embedding=[0.2] * 512,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)
        labels, embeddings, category = cache.get_category("test")

        assert labels == ["happy", "sad"]
        assert embeddings.shape == (2, 512)
        assert category.selection_mode == "single"

    def test_classify_returns_scores(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="a",
                prompt_text="a",
                embedding=[1.0] + [0.0] * 511,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
            PromptEmbedding(
                id=2,
                category_id=1,
                label="b",
                prompt_text="b",
                embedding=[0.0] + [1.0] + [0.0] * 510,  # Orthogonal to "a"
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        # Image embedding similar to "a"
        image_embedding = torch.tensor([[1.0] + [0.0] * 511])
        results = cache.classify(image_embedding, "test")

        assert "a" in results
        assert "b" in results
        assert results["a"] > results["b"]

    def test_classify_multi_returns_filtered_results(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="multi",
            min_confidence=0.5,
            max_results=3,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="high",
                prompt_text="high",
                embedding=[1.0] + [0.0] * 511,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
            PromptEmbedding(
                id=2,
                category_id=1,
                label="low",
                prompt_text="low",
                embedding=[-1.0] + [0.0] * 511,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        # Image embedding similar to "high"
        image_embedding = torch.tensor([[1.0] + [0.0] * 511])
        results = cache.classify_multi(image_embedding, "test")

        # Should return tuples of (label, confidence, prompt_id)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        # "high" should be above threshold, "low" should be filtered out
        labels = [r[0] for r in results]
        assert "high" in labels

    def test_get_prompt_ids(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=10,
                category_id=1,
                label="a",
                prompt_text="a",
                embedding=[0.1] * 512,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
            PromptEmbedding(
                id=20,
                category_id=1,
                label="b",
                prompt_text="b",
                embedding=[0.2] * 512,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)
        prompt_ids = cache.get_prompt_ids("test")

        assert prompt_ids == [10, 20]

    def test_invalidate_cache(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="a",
                prompt_text="a",
                embedding=[0.1] * 512,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        # Load category
        cache.get_category("test")
        assert mock_repo.get_prompt_category_by_name.call_count == 1

        # Should use cache
        cache.get_category("test")
        assert mock_repo.get_prompt_category_by_name.call_count == 1

        # Invalidate and reload
        cache.invalidate("test")
        cache.get_category("test")
        assert mock_repo.get_prompt_category_by_name.call_count == 2

    def test_unknown_category_raises_error(self):
        from photodb.utils.prompt_cache import PromptCache

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = None

        cache = PromptCache(mock_repo)

        with pytest.raises(ValueError, match="Unknown prompt category"):
            cache.get_category("nonexistent")

    def test_no_prompts_raises_error(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="empty",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = []

        cache = PromptCache(mock_repo)

        with pytest.raises(ValueError, match="No prompts found"):
            cache.get_category("empty")

    def test_prompts_without_embeddings_raises_error(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="a",
                prompt_text="a",
                embedding=None,  # No embedding
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        with pytest.raises(ValueError, match="No computed embeddings"):
            cache.get_category("test")

    def test_device_auto_detection(self):
        from photodb.utils.prompt_cache import PromptCache

        mock_repo = MagicMock()
        cache = PromptCache(mock_repo)

        # Should auto-detect device (cpu, mps, or cuda)
        assert cache._device in ["cpu", "mps", "cuda"]

    def test_explicit_device_override(self):
        from photodb.utils.prompt_cache import PromptCache

        mock_repo = MagicMock()
        cache = PromptCache(mock_repo, device="cpu")

        assert cache._device == "cpu"

    def test_classify_handles_1d_embedding(self):
        from photodb.utils.prompt_cache import PromptCache
        from photodb.database.models import PromptEmbedding, PromptCategory

        mock_repo = MagicMock()
        mock_repo.get_prompt_category_by_name.return_value = PromptCategory(
            id=1,
            name="test",
            target="scene",
            selection_mode="single",
            min_confidence=0.1,
            max_results=5,
            description=None,
            display_order=0,
            is_active=True,
            created_at=None,
            updated_at=None,
        )
        mock_repo.get_prompts_by_category.return_value = [
            PromptEmbedding(
                id=1,
                category_id=1,
                label="a",
                prompt_text="a",
                embedding=[1.0] + [0.0] * 511,
                model_name="test",
                model_version=None,
                display_name=None,
                parent_label=None,
                confidence_boost=0.0,
                metadata=None,
                is_active=True,
                embedding_computed_at=None,
                created_at=None,
                updated_at=None,
            ),
        ]

        cache = PromptCache(mock_repo)

        # Pass 1D tensor (should be handled)
        image_embedding = torch.tensor([1.0] + [0.0] * 511)
        results = cache.classify(image_embedding, "test")

        assert "a" in results
        assert isinstance(results["a"], float)
