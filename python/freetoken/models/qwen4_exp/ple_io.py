"""Bounded byte readers and capability gates for Qwen4-Exp PLE streaming.

The production 256K profile deliberately requires a native io_uring +
``O_DIRECT`` implementation.  This source tree does not silently substitute a
thread pool for io_uring: :class:`DirectIOPageReader` is an explicit debug and
correctness backend, while :func:`probe_ple_streaming_capability` reports
whether the optional native extension is actually importable.

The debug reader is still useful.  It performs aligned ``preadv`` calls into
anonymous mmap buffers (which satisfy Linux ``O_DIRECT`` alignment), keeps a
strictly bounded LRU, exposes telemetry, and supports cancellable look-ahead
prefetch without ever materialising a complete PLE shard.
"""

from __future__ import annotations

import ctypes.util
import importlib.util
import mmap
import os
import platform
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ALIGNMENT = 4096
DEFAULT_CACHE_BYTES = 4 * 1024**3
DEFAULT_QUEUE_DEPTH = 512
DEFAULT_MAX_BATCH_PAGES = 4096
NATIVE_EXTENSION = "freetoken.models.qwen4_exp._ple_io_uring"


class PLEStreamingUnavailable(RuntimeError):
    """Raised when a profile requires native streaming that is unavailable."""


@dataclass(frozen=True)
class PLEStreamingCapability:
    linux: bool
    odirect_constant: bool
    preadv: bool
    odirect_read: bool
    kernel_io_uring: bool
    liburing: bool
    native_extension: bool
    filesystem_path: str
    detail: str

    @property
    def production_ready(self) -> bool:
        return bool(
            self.linux
            and self.odirect_constant
            and self.preadv
            and self.odirect_read
            and self.kernel_io_uring
            and self.native_extension
        )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["production_ready"] = self.production_ready
        return result


@dataclass(frozen=True)
class PLEIOTelemetry:
    read_calls: int
    requested_bytes: int
    storage_bytes: int
    cache_hit_pages: int
    cache_miss_pages: int
    cache_entries: int
    cache_bytes: int
    cache_capacity_bytes: int
    evicted_pages: int
    prefetch_submitted: int
    prefetch_completed: int
    prefetch_cancelled: int
    prefetch_errors: int
    wait_ns: int
    submission_batches: int
    submitted_sqes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _kernel_io_uring_enabled() -> bool:
    if sys.platform != "linux":
        return False
    try:
        # 0: enabled, 1: disabled for unprivileged callers, 2: disabled for all.
        disabled = int(
            Path("/proc/sys/kernel/io_uring_disabled")
            .read_text(encoding="ascii")
            .strip()
        )
        return disabled == 0 or (disabled == 1 and os.geteuid() == 0)
    except (OSError, ValueError):
        # Older kernels do not expose the sysctl.  A modern WSL2 kernel version
        # is necessary but not sufficient; the native extension remains the
        # authoritative runtime probe.
        release = platform.release().split("-")[0]
        try:
            major, minor, *_ = (int(value) for value in release.split("."))
        except ValueError:
            return False
        return (major, minor) >= (5, 10)


def _probe_file(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        try:
            return next(item for item in candidate.glob("*.safetensors") if item.is_file())
        except StopIteration:
            return None
    return None


def _probe_odirect_read(path: Path | None, alignment: int) -> tuple[bool, str]:
    if path is None:
        return False, "no existing safetensors file is available for a read-only probe"
    flag = getattr(os, "O_DIRECT", None)
    preadv = getattr(os, "preadv", None)
    if flag is None or preadv is None:
        return False, "Python does not expose O_DIRECT and preadv"
    descriptor: int | None = None
    buffer: mmap.mmap | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | flag)
        buffer = mmap.mmap(-1, alignment)
        count = preadv(descriptor, [buffer], 0)
        if count <= 0:
            return False, "O_DIRECT probe returned no bytes"
        return True, f"aligned O_DIRECT preadv read {count} bytes"
    except OSError as exc:
        return False, f"O_DIRECT probe failed: {exc}"
    finally:
        if buffer is not None:
            buffer.close()
        if descriptor is not None:
            os.close(descriptor)


def probe_ple_streaming_capability(
    path: str | os.PathLike[str],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> PLEStreamingCapability:
    """Probe the complete production contract without writing to ``path``."""

    linux = sys.platform == "linux"
    odirect_constant = getattr(os, "O_DIRECT", None) is not None
    has_preadv = getattr(os, "preadv", None) is not None
    probe_path = _probe_file(path)
    odirect_read, odirect_detail = (
        _probe_odirect_read(probe_path, alignment)
        if linux and odirect_constant and has_preadv
        else (False, "O_DIRECT probing requires Linux with os.O_DIRECT and os.preadv")
    )
    kernel_io_uring = _kernel_io_uring_enabled()
    liburing = bool(ctypes.util.find_library("uring")) if linux else False
    native_extension = importlib.util.find_spec(NATIVE_EXTENSION) is not None
    blockers = []
    if not odirect_read:
        blockers.append(odirect_detail)
    if not kernel_io_uring:
        blockers.append("io_uring is disabled or unavailable to this process")
    if not native_extension:
        blockers.append(
            f"optional native extension {NATIVE_EXTENSION!r} is not installed"
        )
    detail = "; ".join(blockers) if blockers else "native io_uring + O_DIRECT probe passed"
    return PLEStreamingCapability(
        linux=linux,
        odirect_constant=odirect_constant,
        preadv=has_preadv,
        odirect_read=odirect_read,
        kernel_io_uring=kernel_io_uring,
        liburing=liburing,
        native_extension=native_extension,
        filesystem_path=str(probe_path or Path(path).expanduser()),
        detail=detail,
    )


def require_production_ple_streaming(path: str | os.PathLike[str]) -> PLEStreamingCapability:
    capability = probe_ple_streaming_capability(path)
    if not capability.production_ready:
        raise PLEStreamingUnavailable(
            "the 256K profile requires native io_uring + O_DIRECT PLE streaming: "
            + capability.detail
        )
    return capability


class PrefetchHandle:
    """A small aggregate around independently cancellable read futures."""

    def __init__(self, futures: Sequence[Future[bytes | tuple[bytes, ...]]]) -> None:
        self._futures = tuple(futures)

    def cancel(self) -> int:
        return sum(1 for future in self._futures if future.cancel())

    def wait(self) -> tuple[bytes, ...]:
        payloads: list[bytes] = []
        for future in self._futures:
            value = future.result()
            if isinstance(value, tuple):
                payloads.extend(value)
            else:
                payloads.append(value)
        return tuple(payloads)

    @property
    def done(self) -> bool:
        return all(future.done() for future in self._futures)


class DirectIOPageReader:
    """Aligned synchronous O_DIRECT reader with bounded debug LRU/prefetch.

    This class is intentionally named ``DirectIO`` rather than ``IoUring``.  It
    is not accepted by the production capability gate and exists to validate
    offset, cache, cancellation and telemetry semantics on ordinary CI hosts.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        alignment: int = DEFAULT_ALIGNMENT,
        cache_capacity_bytes: int = DEFAULT_CACHE_BYTES,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
        max_batch_pages: int = DEFAULT_MAX_BATCH_PAGES,
    ) -> None:
        if sys.platform != "linux":
            raise PLEStreamingUnavailable("O_DIRECT PLE debug reader requires Linux")
        if alignment < 512 or alignment & (alignment - 1):
            raise ValueError("alignment must be a power of two and at least 512")
        if cache_capacity_bytes < 0:
            raise ValueError("cache_capacity_bytes must be non-negative")
        if queue_depth < 1:
            raise ValueError("queue_depth must be positive")
        if not 1 <= max_batch_pages <= DEFAULT_MAX_BATCH_PAGES:
            raise ValueError(
                f"max_batch_pages must be in [1, {DEFAULT_MAX_BATCH_PAGES}]"
            )
        flag = getattr(os, "O_DIRECT", None)
        if flag is None or getattr(os, "preadv", None) is None:
            raise PLEStreamingUnavailable("Python lacks O_DIRECT/preadv support")

        self.path = str(Path(path))
        self.alignment = alignment
        self.cache_capacity_bytes = cache_capacity_bytes
        self.queue_depth = queue_depth
        self.max_batch_pages = max_batch_pages
        self._fd = os.open(self.path, os.O_RDONLY | flag)
        self._size = os.fstat(self._fd).st_size
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_bytes = 0
        self._closed = False
        self._lock = threading.RLock()
        # A bounded worker count prevents debug prefetch from creating hundreds
        # of simultaneous large aligned mmaps.  queue_depth is still recorded
        # and enforced at submission as the production contract value.
        self._executor = ThreadPoolExecutor(
            max_workers=min(4, queue_depth), thread_name_prefix="ple-odirect"
        )
        self._outstanding: set[Future[bytes]] = set()
        self._read_calls = 0
        self._requested_bytes = 0
        self._storage_bytes = 0
        self._cache_hit_pages = 0
        self._cache_miss_pages = 0
        self._evicted_pages = 0
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._prefetch_cancelled = 0
        self._prefetch_errors = 0
        self._wait_ns = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> int:
        return self._size

    def _check_span(self, offset: int, length: int) -> None:
        if self._closed:
            raise RuntimeError("cannot read from a closed PLE reader")
        if offset < 0 or length < 0 or offset + length > self._size:
            raise ValueError(
                f"read span [{offset}, {offset + length}) is outside [0, {self._size})"
            )

    def _read_pages(self, first_page: int, page_count: int) -> dict[int, bytes]:
        result: dict[int, bytes] = {}
        remaining = page_count
        page = first_page
        while remaining:
            batch_pages = min(remaining, self.max_batch_pages)
            byte_count = batch_pages * self.alignment
            offset = page * self.alignment
            buffer = mmap.mmap(-1, byte_count)
            try:
                started = time.perf_counter_ns()
                read = os.preadv(self._fd, [buffer], offset)
                elapsed = time.perf_counter_ns() - started
                with self._lock:
                    self._wait_ns += elapsed
                    self._storage_bytes += read
                raw = buffer[:read]
            finally:
                buffer.close()
            for index in range(batch_pages):
                start = index * self.alignment
                if start >= len(raw):
                    break
                result[page + index] = raw[start : start + self.alignment]
            page += batch_pages
            remaining -= batch_pages
        return result

    def _insert_page(self, page: int, payload: bytes) -> None:
        if not self.cache_capacity_bytes or not payload:
            return
        previous = self._cache.pop(page, None)
        if previous is not None:
            self._cache_bytes -= len(previous)
        self._cache[page] = payload
        self._cache_bytes += len(payload)
        while self._cache_bytes > self.cache_capacity_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= len(evicted)
            self._evicted_pages += 1

    @staticmethod
    def _runs(pages: Sequence[int]) -> Iterable[tuple[int, int]]:
        if not pages:
            return
        first = previous = pages[0]
        for page in pages[1:]:
            if page != previous + 1:
                yield first, previous - first + 1
                first = page
            previous = page
        yield first, previous - first + 1

    def read(self, offset: int, length: int) -> bytes:
        self._check_span(offset, length)
        if length == 0:
            return b""
        first_page = offset // self.alignment
        last_page = (offset + length - 1) // self.alignment
        pages = list(range(first_page, last_page + 1))
        found: dict[int, bytes] = {}
        missing: list[int] = []
        with self._lock:
            self._read_calls += 1
            self._requested_bytes += length
            for page in pages:
                payload = self._cache.get(page)
                if payload is None:
                    missing.append(page)
                    self._cache_miss_pages += 1
                else:
                    self._cache.move_to_end(page)
                    found[page] = payload
                    self._cache_hit_pages += 1
        for run_start, run_count in self._runs(missing):
            loaded = self._read_pages(run_start, run_count)
            with self._lock:
                for page, payload in loaded.items():
                    found[page] = payload
                    self._insert_page(page, payload)
        aligned = b"".join(found[page] for page in pages)
        inner = offset - first_page * self.alignment
        return aligned[inner : inner + length]

    def _prefetch_done(self, future: Future[bytes]) -> None:
        with self._lock:
            self._outstanding.discard(future)
            if future.cancelled():
                self._prefetch_cancelled += 1
                return
            try:
                future.result()
            except BaseException:
                self._prefetch_errors += 1
            else:
                self._prefetch_completed += 1

    def prefetch(self, spans: Sequence[tuple[int, int]]) -> PrefetchHandle:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot prefetch from a closed PLE reader")
            available = self.queue_depth - len(self._outstanding)
            if len(spans) > available:
                raise RuntimeError(
                    f"PLE prefetch queue depth {self.queue_depth} exceeded: "
                    f"{len(spans)} new, {len(self._outstanding)} outstanding"
                )
            futures = []
            for offset, length in spans:
                self._check_span(offset, length)
                future = self._executor.submit(self.read, offset, length)
                self._outstanding.add(future)
                self._prefetch_submitted += 1
                future.add_done_callback(self._prefetch_done)
                futures.append(future)
        return PrefetchHandle(futures)

    def telemetry(self) -> PLEIOTelemetry:
        with self._lock:
            return PLEIOTelemetry(
                read_calls=self._read_calls,
                requested_bytes=self._requested_bytes,
                storage_bytes=self._storage_bytes,
                cache_hit_pages=self._cache_hit_pages,
                cache_miss_pages=self._cache_miss_pages,
                cache_entries=len(self._cache),
                cache_bytes=self._cache_bytes,
                cache_capacity_bytes=self.cache_capacity_bytes,
                evicted_pages=self._evicted_pages,
                prefetch_submitted=self._prefetch_submitted,
                prefetch_completed=self._prefetch_completed,
                prefetch_cancelled=self._prefetch_cancelled,
                prefetch_errors=self._prefetch_errors,
                wait_ns=self._wait_ns,
                submission_batches=0,
                submitted_sqes=0,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            outstanding = tuple(self._outstanding)
        for future in outstanding:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
        os.close(self._fd)

    def __enter__(self) -> "DirectIOPageReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class IoUringPageReader:
    """Production wrapper around the bundled native io_uring/O_DIRECT reader."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        alignment: int = DEFAULT_ALIGNMENT,
        cache_capacity_bytes: int = DEFAULT_CACHE_BYTES,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
        max_batch_pages: int = DEFAULT_MAX_BATCH_PAGES,
    ) -> None:
        if sys.platform != "linux":
            raise PLEStreamingUnavailable("native io_uring PLE reader requires Linux")
        try:
            from ._ple_io_uring import IoUringReader
        except ImportError as exc:
            raise PLEStreamingUnavailable(
                f"optional native extension {NATIVE_EXTENSION!r} is not installed"
            ) from exc
        self.path = str(Path(path))
        self.queue_depth = int(queue_depth)
        self._native = IoUringReader(
            self.path,
            alignment=alignment,
            cache_capacity_bytes=cache_capacity_bytes,
            queue_depth=queue_depth,
            max_batch_pages=max_batch_pages,
        )
        self._executor = ThreadPoolExecutor(
            # One native read_many call owns the whole ring batch. More Python
            # workers would only contend on the C++ reader mutex and turn a
            # queue-depth-512 submission back into serialized one-SQE reads.
            max_workers=1, thread_name_prefix="ple-io-uring"
        )
        self._lock = threading.RLock()
        self._outstanding: set[Future[bytes]] = set()
        self._outstanding_spans: dict[Future[bytes], int] = {}
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._prefetch_cancelled = 0
        self._prefetch_errors = 0

    @property
    def size(self) -> int:
        return int(self._native.size)

    @property
    def closed(self) -> bool:
        return bool(self._native.closed)

    def read(self, offset: int, length: int) -> bytes:
        return self._native.read(offset, length)

    def read_many(self, spans: Sequence[tuple[int, int]]) -> tuple[bytes, ...]:
        """Submit sparse page runs in queue-depth-sized native ring batches."""

        return tuple(self._native.read_many(tuple(spans)))

    def _prefetch_done(self, future: Future[bytes]) -> None:
        with self._lock:
            self._outstanding.discard(future)
            span_count = self._outstanding_spans.pop(future, 1)
            if future.cancelled():
                self._prefetch_cancelled += span_count
                return
            try:
                future.result()
            except BaseException:
                self._prefetch_errors += span_count
            else:
                self._prefetch_completed += span_count

    def prefetch(self, spans: Sequence[tuple[int, int]]) -> PrefetchHandle:
        spans = tuple(spans)
        if not spans:
            return PrefetchHandle(())
        with self._lock:
            if self.closed:
                raise RuntimeError("cannot prefetch from a closed PLE reader")
            # queue_depth limits SQEs in one native ring submission, not the
            # number of sparse rows a 512-token chunk may request.  A single
            # executor serializes these bounded batches and lets cancellation
            # discard batches that have not started yet.
            futures = []
            for start in range(0, len(spans), self.queue_depth):
                batch = spans[start : start + self.queue_depth]
                future = self._executor.submit(self.read_many, batch)
                self._outstanding.add(future)
                self._outstanding_spans[future] = len(batch)
                self._prefetch_submitted += len(batch)
                future.add_done_callback(self._prefetch_done)
                futures.append(future)
        return PrefetchHandle(futures)

    def telemetry(self) -> PLEIOTelemetry:
        native = self._native.telemetry()
        with self._lock:
            return PLEIOTelemetry(
                read_calls=int(native["read_calls"]),
                requested_bytes=int(native["requested_bytes"]),
                storage_bytes=int(native["storage_bytes"]),
                cache_hit_pages=int(native["cache_hit_pages"]),
                cache_miss_pages=int(native["cache_miss_pages"]),
                cache_entries=int(native["cache_entries"]),
                cache_bytes=int(native["cache_bytes"]),
                cache_capacity_bytes=int(native["cache_capacity_bytes"]),
                evicted_pages=int(native["evicted_pages"]),
                prefetch_submitted=self._prefetch_submitted,
                prefetch_completed=self._prefetch_completed,
                prefetch_cancelled=self._prefetch_cancelled,
                prefetch_errors=self._prefetch_errors,
                wait_ns=int(native["wait_ns"]),
                submission_batches=int(native["submission_batches"]),
                submitted_sqes=int(native["submitted_sqes"]),
            )

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            outstanding = tuple(self._outstanding)
        for future in outstanding:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._native.close()

    def __enter__(self) -> "IoUringPageReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_CACHE_BYTES",
    "DEFAULT_MAX_BATCH_PAGES",
    "DEFAULT_QUEUE_DEPTH",
    "DirectIOPageReader",
    "IoUringPageReader",
    "PLEIOTelemetry",
    "PLEStreamingCapability",
    "PLEStreamingUnavailable",
    "PrefetchHandle",
    "probe_ple_streaming_capability",
    "require_production_ple_streaming",
]
