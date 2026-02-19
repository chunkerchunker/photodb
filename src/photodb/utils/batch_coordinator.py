"""
Batch coordinator for ML inference requests.

Collects individual inference requests from multiple worker threads, batches them
together, runs a single batched inference call, and distributes results back to
callers via Futures. This amortizes per-call overhead (e.g., GPU kernel launch,
model dispatch) across many inputs.

Example:
    def my_inference(batch: list[torch.Tensor]) -> list[torch.Tensor]:
        stacked = torch.stack(batch)
        return list(stacked * 2)

    with BatchCoordinator(inference_fn=my_inference, max_batch_size=16) as coord:
        future = coord.submit(torch.tensor([1.0, 2.0]))
        result = future.result()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()


class BatchCoordinator:
    """
    Collect ML inference requests from worker threads, batch them, and run inference.

    Worker threads call ``submit(input_item)`` which returns a ``Future``. A background
    daemon thread collects submitted items, forms batches up to ``max_batch_size`` (or
    after ``max_wait_ms`` of idle time), invokes ``inference_fn`` on the batch, and
    resolves each caller's Future with the corresponding result slice.

    Supports both PyTorch tensor batching (via ``torch.cat``) and plain list
    concatenation. Torch is imported lazily to avoid overhead when only lists are used.
    """

    def __init__(
        self,
        inference_fn: Callable[[Any], Any],
        max_batch_size: int = 32,
        max_wait_ms: int = 50,
    ) -> None:
        """
        Initialize the BatchCoordinator.

        Args:
            inference_fn: Callable that accepts a batched input (tensor or list) and
                returns a batched output of the same length. The output is split and
                distributed to individual callers.
            max_batch_size: Maximum number of items to include in a single batch.
            max_wait_ms: Maximum time in milliseconds to wait for additional items
                after the first item arrives before dispatching a partial batch.
        """
        self._inference_fn = inference_fn
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

        self._queue: queue.Queue[tuple[Any, Future] | object] = queue.Queue()
        self._closed = False

        # Statistics
        self._batches_processed = 0
        self._items_processed = 0
        self._stats_lock = threading.Lock()

        # Start the background batch loop
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    @property
    def stats(self) -> dict[str, float]:
        """
        Return processing statistics.

        Returns:
            Dict with ``batches_processed``, ``items_processed``, and ``avg_batch_size``.
        """
        with self._stats_lock:
            batches = self._batches_processed
            items = self._items_processed
        return {
            "batches_processed": batches,
            "items_processed": items,
            "avg_batch_size": items / batches if batches > 0 else 0.0,
        }

    def submit(self, input_item: Any) -> Future:
        """
        Submit a single input for batched inference.

        Thread-safe. May be called concurrently from many worker threads.

        Args:
            input_item: A single inference input. Will be combined with other inputs
                into a batch before being passed to ``inference_fn``.

        Returns:
            A ``Future`` whose result will be set to the corresponding output slice
            once the batch containing this item has been processed.

        Raises:
            RuntimeError: If the coordinator has been closed.
        """
        if self._closed:
            raise RuntimeError("BatchCoordinator is closed")
        future: Future = Future()
        self._queue.put((input_item, future))
        return future

    def close(self) -> None:
        """
        Stop the background thread and release resources.

        Sends a sentinel value to the batch loop, causing it to exit after it
        finishes any in-progress batch. Any items still queued after the thread
        exits have their futures cancelled.
        """
        self._closed = True
        self._queue.put(_SENTINEL)
        self._thread.join()

        # Drain any items that were queued after _closed was set but before
        # the sentinel was consumed (TOCTOU race between submit and close).
        cancelled = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not _SENTINEL and isinstance(item, tuple):
                _, future = item
                future.cancel()
                cancelled += 1
        if cancelled:
            logger.debug("Cancelled %d queued items during close", cancelled)

    def __enter__(self) -> BatchCoordinator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _batch_loop(self) -> None:
        """Background loop: collect items, form batches, run inference."""
        while True:
            # Block until at least one item arrives
            first = self._queue.get()
            if first is _SENTINEL:
                return

            items: list[Any] = [first[0]]
            futures: list[Future] = [first[1]]

            # Collect more items up to max_batch_size or max_wait_ms
            deadline = time.monotonic() + self._max_wait_ms / 1000.0
            while len(items) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    # Process what we have, then exit
                    self._run_batch(items, futures)
                    return
                items.append(item[0])
                futures.append(item[1])

            self._run_batch(items, futures)

    def _run_batch(self, items: list[Any], futures: list[Future]) -> None:
        """
        Form a batch from collected items, run inference, and resolve futures.

        Supports PyTorch tensors (concatenated with ``torch.cat``) and plain lists
        (concatenated with ``+``). Torch is imported lazily.
        """
        try:
            batch, mode = self._form_batch(items)
            results = self._inference_fn(batch)
            split = self._split_results(results, len(items), mode)

            for future, result in zip(futures, split):
                future.set_result(result)

            with self._stats_lock:
                self._batches_processed += 1
                self._items_processed += len(items)

            logger.debug("Processed batch of %d items", len(items))

        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            for future in futures:
                future.set_exception(exc)

    def _form_batch(self, items: list[Any]) -> tuple[Any, str]:
        """
        Combine individual items into a single batch.

        If items are PyTorch tensors, uses ``torch.cat``. If items are lists,
        concatenates them. Otherwise wraps items in a plain list.

        Returns:
            A tuple of (batch, mode) where mode is one of "tensor", "list",
            or "scalar" to inform how results should be split.
        """
        if self._is_tensor(items[0]):
            import torch

            return torch.cat(items, dim=0), "tensor"

        # List batching: concatenate lists
        if isinstance(items[0], list):
            batch: list[Any] = []
            for item in items:
                batch.extend(item)
            return batch, "list"

        # Fallback: wrap in a list (each item is a single scalar/object)
        return items, "scalar"

    def _split_results(self, results: Any, count: int, mode: str) -> list[Any]:
        """
        Split batched results back into individual outputs.

        For tensors, splits along dim 0 using ``torch.split``. For lists formed by
        concatenation, splits evenly by length. For scalar-mode lists, returns each
        element directly.

        Args:
            results: The batched output from the inference function.
            count: Number of individual items that were batched.
            mode: One of "tensor", "list", or "scalar" (from ``_form_batch``).
        """
        if mode == "tensor" and self._is_tensor(results):
            import torch

            chunk_size = results.shape[0] // count
            return list(torch.split(results, chunk_size, dim=0))

        if mode == "list" and isinstance(results, list):
            if count == 1:
                return [results]
            chunk_size = len(results) // count
            return [results[i * chunk_size : (i + 1) * chunk_size] for i in range(count)]

        if mode == "scalar" and isinstance(results, (list, tuple)):
            return list(results)

        # Fallback: try indexing
        return [results[i] for i in range(count)]

    @staticmethod
    def _is_tensor(obj: Any) -> bool:
        """Check if an object is a PyTorch tensor without importing torch."""
        return type(obj).__module__.startswith("torch") and hasattr(obj, "shape")
