"""
Tests for BatchCoordinator utility.

Verifies thread-safe batching of ML inference requests from multiple worker threads.
"""

import threading
import time

import pytest
import torch
from concurrent.futures import Future

from photodb.utils.batch_coordinator import BatchCoordinator


class TestBatchCoordinator:
    """Tests for BatchCoordinator utility."""

    def test_single_item_batch(self):
        """Single submit should work even without batching."""

        def double_tensor(batch: torch.Tensor) -> torch.Tensor:
            return batch * 2

        with BatchCoordinator(inference_fn=double_tensor, max_wait_ms=10) as coord:
            future = coord.submit(torch.tensor([[3.0]]))
            result = future.result(timeout=2.0)

        assert torch.equal(result, torch.tensor([[6.0]]))

    def test_multiple_items_batched(self):
        """Multiple concurrent submits should be batched together."""
        call_counts: list[int] = []

        def tracking_double(batch: torch.Tensor) -> torch.Tensor:
            call_counts.append(batch.shape[0])
            return batch * 2

        with BatchCoordinator(
            inference_fn=tracking_double, max_batch_size=8, max_wait_ms=100
        ) as coord:
            barrier = threading.Barrier(4)
            # Use a dict to map value -> future for order-independent verification
            futures_map: dict[float, Future] = {}
            lock = threading.Lock()

            def submit_one(value: float) -> None:
                barrier.wait()
                f = coord.submit(torch.tensor([[value]]))
                with lock:
                    futures_map[value] = f

            threads = [
                threading.Thread(target=submit_one, args=(float(i),)) for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify each value was doubled correctly
            for value, future in futures_map.items():
                result = future.result(timeout=2.0)
                assert result.item() == value * 2

        # All 4 items should have been processed
        total_items = sum(call_counts)
        assert total_items == 4

    def test_max_batch_size_respected(self):
        """Batch size should not exceed max_batch_size."""
        batch_sizes: list[int] = []

        def tracking_fn(batch: torch.Tensor) -> torch.Tensor:
            batch_sizes.append(batch.shape[0])
            return batch * 2

        max_bs = 4
        with BatchCoordinator(
            inference_fn=tracking_fn, max_batch_size=max_bs, max_wait_ms=200
        ) as coord:
            # Submit more items than max_batch_size, synchronized to arrive together
            n_items = 10
            barrier = threading.Barrier(n_items)
            futures: list[Future] = []
            lock = threading.Lock()

            def submit_one(value: float) -> None:
                barrier.wait()
                f = coord.submit(torch.tensor([[value]]))
                with lock:
                    futures.append(f)

            threads = [
                threading.Thread(target=submit_one, args=(float(i),)) for i in range(n_items)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Wait for all futures
            for f in futures:
                f.result(timeout=2.0)

        # No batch should exceed max_batch_size
        for size in batch_sizes:
            assert size <= max_bs

    def test_timeout_triggers_batch(self):
        """Batch should be processed after max_wait_ms even if not full."""
        processed = threading.Event()

        def signal_fn(batch: torch.Tensor) -> torch.Tensor:
            processed.set()
            return batch * 2

        with BatchCoordinator(
            inference_fn=signal_fn, max_batch_size=100, max_wait_ms=20
        ) as coord:
            # Submit just one item (batch won't be full)
            future = coord.submit(torch.tensor([[1.0]]))

            # Should be processed within max_wait_ms + some slack
            assert processed.wait(timeout=2.0), "Batch was not processed within timeout"
            result = future.result(timeout=2.0)
            assert torch.equal(result, torch.tensor([[2.0]]))

    def test_exception_propagated_to_futures(self):
        """If inference_fn raises, all futures in batch should get the exception."""

        def failing_fn(batch: torch.Tensor) -> torch.Tensor:
            raise ValueError("Inference failed intentionally")

        with BatchCoordinator(inference_fn=failing_fn, max_wait_ms=10) as coord:
            barrier = threading.Barrier(3)
            futures: list[Future] = []
            lock = threading.Lock()

            def submit_one() -> None:
                barrier.wait()
                f = coord.submit(torch.tensor([[1.0]]))
                with lock:
                    futures.append(f)

            threads = [threading.Thread(target=submit_one) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All futures should raise the same exception
            for f in futures:
                with pytest.raises(ValueError, match="Inference failed intentionally"):
                    f.result(timeout=2.0)

    def test_close_stops_processing(self):
        """close() should stop the background thread."""

        def identity(batch: torch.Tensor) -> torch.Tensor:
            return batch

        coord = BatchCoordinator(inference_fn=identity, max_wait_ms=10)
        coord.close()

        assert not coord._thread.is_alive()

        # Submitting after close should raise
        with pytest.raises(RuntimeError, match="closed"):
            coord.submit(torch.tensor([[1.0]]))

    def test_context_manager(self):
        """Should work as context manager."""

        def identity(batch: torch.Tensor) -> torch.Tensor:
            return batch

        with BatchCoordinator(inference_fn=identity, max_wait_ms=10) as coord:
            future = coord.submit(torch.tensor([[42.0]]))
            result = future.result(timeout=2.0)
            assert result.item() == 42.0

        # After exiting context, thread should be stopped
        assert not coord._thread.is_alive()

    def test_stats_tracking(self):
        """Stats should track batches and items processed."""

        def identity(batch: torch.Tensor) -> torch.Tensor:
            return batch

        with BatchCoordinator(
            inference_fn=identity, max_batch_size=32, max_wait_ms=10
        ) as coord:
            # Submit several items sequentially (each will likely be its own batch)
            for i in range(3):
                future = coord.submit(torch.tensor([[float(i)]]))
                future.result(timeout=2.0)

            stats = coord.stats
            assert stats["items_processed"] == 3
            assert stats["batches_processed"] >= 1
            assert stats["avg_batch_size"] > 0

    def test_list_batching(self):
        """Should work with list inputs (not just tensors)."""

        def double_list(batch: list) -> list:
            return [x * 2 for x in batch]

        with BatchCoordinator(
            inference_fn=double_list, max_batch_size=32, max_wait_ms=10
        ) as coord:
            future = coord.submit([5])
            result = future.result(timeout=2.0)

        assert result == [10]

    def test_list_batching_multiple(self):
        """Multiple list submits should concatenate and split correctly."""

        def double_list(batch: list) -> list:
            return [x * 2 for x in batch]

        with BatchCoordinator(
            inference_fn=double_list, max_batch_size=32, max_wait_ms=100
        ) as coord:
            barrier = threading.Barrier(3)
            futures: list[Future] = []
            lock = threading.Lock()

            def submit_one(value: int) -> None:
                barrier.wait()
                f = coord.submit([value])
                with lock:
                    futures.append(f)

            threads = [
                threading.Thread(target=submit_one, args=(i,)) for i in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            results = [f.result(timeout=2.0) for f in futures]

        # Each result should be a list with one doubled item
        all_values = sorted([r[0] for r in results])
        assert all_values == [0, 2, 4]

    def test_concurrent_submitters(self):
        """Multiple threads submitting concurrently should all get correct results."""

        def double_tensor(batch: torch.Tensor) -> torch.Tensor:
            return batch * 2

        n_workers = 20
        results_map: dict[int, float] = {}
        results_lock = threading.Lock()

        with BatchCoordinator(
            inference_fn=double_tensor, max_batch_size=8, max_wait_ms=20
        ) as coord:
            barrier = threading.Barrier(n_workers)

            def worker(idx: int) -> None:
                barrier.wait()
                future = coord.submit(torch.tensor([[float(idx)]]))
                result = future.result(timeout=5.0)
                with results_lock:
                    results_map[idx] = result.item()

            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(n_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Every worker should have gotten the correct doubled value
        assert len(results_map) == n_workers
        for idx, value in results_map.items():
            assert value == float(idx) * 2, f"Worker {idx}: expected {idx * 2}, got {value}"

    def test_non_list_non_tensor_fallback(self):
        """Items that are neither tensors nor lists should be wrapped in a list."""

        def identity(batch: list) -> list:
            return batch

        with BatchCoordinator(inference_fn=identity, max_wait_ms=10) as coord:
            future = coord.submit(42)
            result = future.result(timeout=2.0)

        assert result == 42

    def test_close_drains_queued_items(self):
        """close() should cancel futures for items queued after _closed is set."""
        # Use a slow inference fn so items pile up
        slow_event = threading.Event()

        def slow_fn(batch: torch.Tensor) -> torch.Tensor:
            slow_event.wait(timeout=5.0)
            return batch

        coord = BatchCoordinator(inference_fn=slow_fn, max_batch_size=1, max_wait_ms=10)

        # Submit first item - will be picked up by batch loop and block on slow_event
        f1 = coord.submit(torch.tensor([[1.0]]))
        time.sleep(0.05)  # Let batch loop pick it up

        # Submit second item while first is blocking
        f2 = coord.submit(torch.tensor([[2.0]]))

        # Release the slow fn and close immediately
        slow_event.set()
        coord.close()

        # First future should have completed
        assert f1.result(timeout=2.0).item() == 1.0
        # Second future should either complete or be cancelled (not hang forever)
        assert f2.done()

    def test_preprocess_face_crops_batch(self):
        """preprocess_face_crops_batch should produce same tensors as individual calls."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer
        from PIL import Image
        import tempfile
        import os

        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (200, 200), color=(100, 150, 200))
            img.save(f, "JPEG")
            img.close()
            img_path = f.name

        try:
            analyzer = MobileCLIPAnalyzer()
            bboxes = [
                {"x1": 10, "y1": 10, "x2": 60, "y2": 60},
                {"x1": 80, "y1": 80, "x2": 150, "y2": 150},
            ]

            # Batch method
            batch_results = analyzer.preprocess_face_crops_batch(img_path, bboxes)

            # Individual method
            individual_results = [
                analyzer.preprocess_face_crop(img_path, bbox) for bbox in bboxes
            ]

            assert len(batch_results) == len(individual_results) == 2
            for batch_t, indiv_t in zip(batch_results, individual_results):
                assert torch.equal(batch_t, indiv_t)

            # Empty bboxes should return empty list
            assert analyzer.preprocess_face_crops_batch(img_path, []) == []
        finally:
            os.unlink(img_path)
