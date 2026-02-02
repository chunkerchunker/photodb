"""
Prompt embedding cache for efficient zero-shot classification.

Caches prompt embeddings in GPU/CPU tensors for fast similarity computation.
Supports 1000+ prompts per category with single matrix multiplication.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..database.models import PromptCategory, PromptEmbedding

logger = logging.getLogger(__name__)


class PromptCache:
    """Cache prompt embeddings for fast classification.

    Usage:
        cache = PromptCache(repository)
        labels, embeddings, category = cache.get_category("scene_mood")
        scores = cache.classify(image_embedding, "scene_mood")
    """

    def __init__(self, repository: Any, device: Optional[str] = None):
        """
        Initialize prompt cache.

        Args:
            repository: Database repository for loading prompts.
            device: Device for tensors ('cpu', 'mps', 'cuda'). Auto-detects if None.
        """
        self._repository = repository
        self._cache: Dict[str, Tuple[List[str], List[int], torch.Tensor, PromptCategory]] = {}

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        logger.info(f"PromptCache initialized on {device}")

    def get_category(self, category_name: str) -> Tuple[List[str], torch.Tensor, PromptCategory]:
        """
        Get cached embeddings for a category.

        Args:
            category_name: Name of the prompt category.

        Returns:
            Tuple of (labels, embeddings_tensor, category).
        """
        if category_name not in self._cache:
            self._load_category(category_name)

        labels, prompt_ids, embeddings, category = self._cache[category_name]
        return labels, embeddings, category

    def get_prompt_ids(self, category_name: str) -> List[int]:
        """Get prompt IDs for a category (for creating tags)."""
        if category_name not in self._cache:
            self._load_category(category_name)
        return self._cache[category_name][1]

    def _load_category(self, category_name: str) -> None:
        """Load a category's prompts into cache."""
        category = self._repository.get_prompt_category_by_name(category_name)
        if not category:
            raise ValueError(f"Unknown prompt category: {category_name}")

        prompts: List[PromptEmbedding] = self._repository.get_prompts_by_category(
            category.id, active_only=True
        )
        if not prompts:
            raise ValueError(f"No prompts found for category: {category_name}")

        # Filter prompts with embeddings
        prompts_with_embeddings = [p for p in prompts if p.embedding is not None]
        if not prompts_with_embeddings:
            raise ValueError(f"No computed embeddings for category: {category_name}")

        labels = [p.label for p in prompts_with_embeddings]
        prompt_ids = [p.id for p in prompts_with_embeddings if p.id is not None]

        # Stack embeddings into tensor
        embeddings = torch.stack(
            [torch.tensor(p.embedding, dtype=torch.float32) for p in prompts_with_embeddings]
        ).to(self._device)

        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        self._cache[category_name] = (labels, prompt_ids, embeddings, category)
        logger.info(f"Cached {len(labels)} prompts for category '{category_name}'")

    def classify(
        self,
        image_embedding: torch.Tensor,
        category_name: str,
        temperature: float = 100.0,
    ) -> Dict[str, float]:
        """
        Classify image against all prompts in a category.

        Args:
            image_embedding: Image embedding tensor (1, 512) or (512,).
            category_name: Name of the prompt category.
            temperature: Softmax temperature (higher = sharper distribution).

        Returns:
            Dict mapping labels to confidence scores.
        """
        labels, text_embeddings, category = self.get_category(category_name)

        # Ensure image embedding is 2D and normalized
        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)
        image_embedding = image_embedding.to(self._device)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Compute cosine similarity via matrix multiplication
        similarities = image_embedding @ text_embeddings.T  # (1, num_prompts)

        # Apply softmax for probability distribution
        scores = torch.softmax(similarities * temperature, dim=-1).squeeze()

        # Handle single-label case (0-d tensor after squeeze)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        return {label: float(score) for label, score in zip(labels, scores)}

    def classify_multi(
        self,
        image_embedding: torch.Tensor,
        category_name: str,
    ) -> List[Tuple[str, float, int]]:
        """
        Classify for multi-select categories, returning results above threshold.

        Args:
            image_embedding: Image embedding tensor.
            category_name: Name of the prompt category.

        Returns:
            List of (label, confidence, prompt_id) tuples sorted by confidence.
        """
        labels, text_embeddings, category = self.get_category(category_name)
        prompt_ids = self.get_prompt_ids(category_name)

        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)
        image_embedding = image_embedding.to(self._device)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Raw cosine similarities (not softmaxed for multi-label)
        similarities = (image_embedding @ text_embeddings.T).squeeze()

        # Convert to 0-1 range: (similarity + 1) / 2
        confidences = (similarities + 1) / 2

        # Filter by threshold and max results
        results: List[Tuple[str, float, int]] = []
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            conf_val = float(conf)
            if conf_val >= category.min_confidence:
                results.append((label, conf_val, prompt_ids[i]))

        # Sort by confidence and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[: category.max_results]

    def invalidate(self, category_name: Optional[str] = None) -> None:
        """Invalidate cache for a category or all categories."""
        if category_name:
            self._cache.pop(category_name, None)
        else:
            self._cache.clear()
        logger.info(f"Cache invalidated: {category_name or 'all'}")
