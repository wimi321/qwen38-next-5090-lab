// Copyright 2026 Qwen3.8 Next 5090 Lab contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal Linux io_uring + O_DIRECT reader for the Qwen4-Exp PLE auxiliary
// bank. This file is an original downstream implementation built against the
// stable Linux UAPI and CPython ABI; liburing is not required. The SGLang PLE
// work that informed the design is acknowledged in THIRD_PARTY_NOTICES.md.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <list>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Ring {
    int fd = -1;
    void* sq_mapping = MAP_FAILED;
    void* cq_mapping = MAP_FAILED;
    size_t sq_mapping_size = 0;
    size_t cq_mapping_size = 0;
    io_uring_sqe* sqes = nullptr;
    size_t sqes_size = 0;

    unsigned* sq_head = nullptr;
    unsigned* sq_tail = nullptr;
    unsigned* sq_mask = nullptr;
    unsigned* sq_entries = nullptr;
    unsigned* sq_array = nullptr;
    unsigned* cq_head = nullptr;
    unsigned* cq_tail = nullptr;
    unsigned* cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;
};

void close_ring(Ring& ring) {
    if (ring.sqes != nullptr) {
        munmap(ring.sqes, ring.sqes_size);
        ring.sqes = nullptr;
    }
    if (ring.sq_mapping != MAP_FAILED) {
        munmap(ring.sq_mapping, ring.sq_mapping_size);
        if (ring.cq_mapping == ring.sq_mapping) {
            ring.cq_mapping = MAP_FAILED;
        }
        ring.sq_mapping = MAP_FAILED;
    }
    if (ring.cq_mapping != MAP_FAILED) {
        munmap(ring.cq_mapping, ring.cq_mapping_size);
        ring.cq_mapping = MAP_FAILED;
    }
    if (ring.fd >= 0) {
        close(ring.fd);
        ring.fd = -1;
    }
}

bool setup_ring(Ring& ring, unsigned entries, std::string& error) {
    io_uring_params params{};
    const int descriptor = static_cast<int>(
        syscall(__NR_io_uring_setup, entries, &params));
    if (descriptor < 0) {
        error = "io_uring_setup failed: " + std::string(strerror(errno));
        return false;
    }
    ring.fd = descriptor;
    ring.sq_mapping_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
    ring.cq_mapping_size = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
    if (params.features & IORING_FEAT_SINGLE_MMAP) {
        const size_t shared_size = std::max(ring.sq_mapping_size, ring.cq_mapping_size);
        void* mapping = mmap(nullptr, shared_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_POPULATE, descriptor,
                             IORING_OFF_SQ_RING);
        if (mapping == MAP_FAILED) {
            error = "mmap io_uring shared ring failed: " + std::string(strerror(errno));
            close_ring(ring);
            return false;
        }
        ring.sq_mapping = mapping;
        ring.cq_mapping = mapping;
        ring.sq_mapping_size = shared_size;
        ring.cq_mapping_size = shared_size;
    } else {
        ring.sq_mapping = mmap(nullptr, ring.sq_mapping_size,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, descriptor,
                               IORING_OFF_SQ_RING);
        if (ring.sq_mapping == MAP_FAILED) {
            error = "mmap io_uring SQ failed: " + std::string(strerror(errno));
            close_ring(ring);
            return false;
        }
        ring.cq_mapping = mmap(nullptr, ring.cq_mapping_size,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, descriptor,
                               IORING_OFF_CQ_RING);
        if (ring.cq_mapping == MAP_FAILED) {
            error = "mmap io_uring CQ failed: " + std::string(strerror(errno));
            close_ring(ring);
            return false;
        }
    }

    ring.sqes_size = params.sq_entries * sizeof(io_uring_sqe);
    ring.sqes = static_cast<io_uring_sqe*>(
        mmap(nullptr, ring.sqes_size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, descriptor, IORING_OFF_SQES));
    if (ring.sqes == MAP_FAILED) {
        ring.sqes = nullptr;
        error = "mmap io_uring SQEs failed: " + std::string(strerror(errno));
        close_ring(ring);
        return false;
    }

    auto* sq = static_cast<char*>(ring.sq_mapping);
    auto* cq = static_cast<char*>(ring.cq_mapping);
    ring.sq_head = reinterpret_cast<unsigned*>(sq + params.sq_off.head);
    ring.sq_tail = reinterpret_cast<unsigned*>(sq + params.sq_off.tail);
    ring.sq_mask = reinterpret_cast<unsigned*>(sq + params.sq_off.ring_mask);
    ring.sq_entries = reinterpret_cast<unsigned*>(sq + params.sq_off.ring_entries);
    ring.sq_array = reinterpret_cast<unsigned*>(sq + params.sq_off.array);
    ring.cq_head = reinterpret_cast<unsigned*>(cq + params.cq_off.head);
    ring.cq_tail = reinterpret_cast<unsigned*>(cq + params.cq_off.tail);
    ring.cq_mask = reinterpret_cast<unsigned*>(cq + params.cq_off.ring_mask);
    ring.cqes = reinterpret_cast<io_uring_cqe*>(cq + params.cq_off.cqes);
    return true;
}

int ring_read(Ring& ring, int file_fd, void* buffer, unsigned length,
              uint64_t offset, uint64_t user_data, std::string& error) {
    const unsigned head = __atomic_load_n(ring.sq_head, __ATOMIC_ACQUIRE);
    const unsigned tail = __atomic_load_n(ring.sq_tail, __ATOMIC_RELAXED);
    if (tail - head >= *ring.sq_entries) {
        error = "io_uring submission queue is full";
        return -1;
    }
    const unsigned array_index = tail & *ring.sq_mask;
    const unsigned sqe_index = array_index;
    io_uring_sqe* sqe = &ring.sqes[sqe_index];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = file_fd;
    sqe->off = offset;
    sqe->addr = reinterpret_cast<uint64_t>(buffer);
    sqe->len = length;
    sqe->user_data = user_data;
    ring.sq_array[array_index] = sqe_index;
    __atomic_store_n(ring.sq_tail, tail + 1, __ATOMIC_RELEASE);

    while (true) {
        const int entered = static_cast<int>(
            syscall(__NR_io_uring_enter, ring.fd, 1, 1,
                    IORING_ENTER_GETEVENTS, nullptr, 0));
        if (entered >= 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        error = "io_uring_enter failed: " + std::string(strerror(errno));
        return -1;
    }

    unsigned cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
    while (cq_head == __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE)) {
        const int waited = static_cast<int>(
            syscall(__NR_io_uring_enter, ring.fd, 0, 1,
                    IORING_ENTER_GETEVENTS, nullptr, 0));
        if (waited < 0 && errno != EINTR) {
            error = "waiting for io_uring completion failed: " +
                    std::string(strerror(errno));
            return -1;
        }
        cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
    }
    io_uring_cqe* cqe = &ring.cqes[cq_head & *ring.cq_mask];
    if (cqe->user_data != user_data) {
        error = "io_uring completion user_data mismatch";
        return -1;
    }
    const int result = cqe->res;
    __atomic_store_n(ring.cq_head, cq_head + 1, __ATOMIC_RELEASE);
    if (result < 0) {
        error = "io_uring read failed: " + std::string(strerror(-result));
        return -1;
    }
    return result;
}

struct PendingRead {
    void* buffer = nullptr;
    unsigned length = 0;
    uint64_t offset = 0;
    uint64_t user_data = 0;
};

bool ring_read_batch(Ring& ring, int file_fd,
                     const std::vector<PendingRead>& pending,
                     std::vector<int>& results, std::string& error) {
    if (pending.empty()) {
        results.clear();
        return true;
    }
    const unsigned head = __atomic_load_n(ring.sq_head, __ATOMIC_ACQUIRE);
    const unsigned tail = __atomic_load_n(ring.sq_tail, __ATOMIC_RELAXED);
    if (pending.size() > *ring.sq_entries ||
        tail - head + pending.size() > *ring.sq_entries) {
        error = "io_uring batch exceeds submission queue capacity";
        return false;
    }
    std::unordered_map<uint64_t, size_t> by_user_data;
    by_user_data.reserve(pending.size());
    for (size_t index = 0; index < pending.size(); ++index) {
        const unsigned array_index = (tail + static_cast<unsigned>(index)) & *ring.sq_mask;
        const unsigned sqe_index = array_index;
        io_uring_sqe* sqe = &ring.sqes[sqe_index];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = file_fd;
        sqe->off = pending[index].offset;
        sqe->addr = reinterpret_cast<uint64_t>(pending[index].buffer);
        sqe->len = pending[index].length;
        sqe->user_data = pending[index].user_data;
        ring.sq_array[array_index] = sqe_index;
        by_user_data.emplace(pending[index].user_data, index);
    }
    __atomic_store_n(ring.sq_tail,
                     tail + static_cast<unsigned>(pending.size()),
                     __ATOMIC_RELEASE);

    unsigned submitted = 0;
    while (submitted < pending.size()) {
        const unsigned remaining = static_cast<unsigned>(pending.size()) - submitted;
        const int entered = static_cast<int>(
            syscall(__NR_io_uring_enter, ring.fd, remaining, 0, 0, nullptr, 0));
        if (entered > 0) {
            submitted += static_cast<unsigned>(entered);
            continue;
        }
        if (entered < 0 && errno == EINTR) {
            continue;
        }
        error = "batched io_uring_enter submission failed: " +
                std::string(strerror(entered < 0 ? errno : EIO));
        return false;
    }

    while (true) {
        const unsigned cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
        const unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        const unsigned available = cq_tail - cq_head;
        if (available >= pending.size()) {
            break;
        }
        const unsigned wanted = static_cast<unsigned>(pending.size()) - available;
        const int waited = static_cast<int>(
            syscall(__NR_io_uring_enter, ring.fd, 0, wanted,
                    IORING_ENTER_GETEVENTS, nullptr, 0));
        if (waited < 0 && errno != EINTR) {
            error = "waiting for batched io_uring completions failed: " +
                    std::string(strerror(errno));
            return false;
        }
    }

    results.assign(pending.size(), -EIO);
    const unsigned cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
    for (size_t index = 0; index < pending.size(); ++index) {
        io_uring_cqe* cqe = &ring.cqes[
            (cq_head + static_cast<unsigned>(index)) & *ring.cq_mask];
        auto found = by_user_data.find(cqe->user_data);
        if (found == by_user_data.end()) {
            error = "batched io_uring completion user_data mismatch";
            __atomic_store_n(ring.cq_head,
                             cq_head + static_cast<unsigned>(pending.size()),
                             __ATOMIC_RELEASE);
            return false;
        }
        results[found->second] = cqe->res;
    }
    __atomic_store_n(ring.cq_head,
                     cq_head + static_cast<unsigned>(pending.size()),
                     __ATOMIC_RELEASE);
    for (int result : results) {
        if (result < 0) {
            error = "batched io_uring read failed: " +
                    std::string(strerror(-result));
            return false;
        }
    }
    return true;
}

struct CachedPage {
    uint64_t index;
    std::vector<unsigned char> bytes;
};

struct ReaderState {
    int file_fd = -1;
    uint64_t file_size = 0;
    size_t alignment = 4096;
    size_t cache_capacity = 4ULL * 1024 * 1024 * 1024;
    unsigned queue_depth = 512;
    unsigned max_batch_pages = 4096;
    Ring ring;
    bool closed = false;
    std::mutex mutex;
    std::list<CachedPage> lru;
    std::unordered_map<uint64_t, std::list<CachedPage>::iterator> cache;
    size_t cache_bytes = 0;
    uint64_t read_calls = 0;
    uint64_t requested_bytes = 0;
    uint64_t storage_bytes = 0;
    uint64_t cache_hit_pages = 0;
    uint64_t cache_miss_pages = 0;
    uint64_t evicted_pages = 0;
    uint64_t wait_ns = 0;
    uint64_t submission_batches = 0;
    uint64_t submitted_sqes = 0;
    uint64_t next_user_data = 1;
};

bool valid_power_of_two(size_t value) {
    return value >= 512 && (value & (value - 1)) == 0;
}

void close_state(ReaderState* state) {
    if (state == nullptr || state->closed) {
        return;
    }
    state->closed = true;
    close_ring(state->ring);
    if (state->file_fd >= 0) {
        close(state->file_fd);
        state->file_fd = -1;
    }
    state->cache.clear();
    state->lru.clear();
    state->cache_bytes = 0;
}

void insert_cache(ReaderState* state, uint64_t page,
                  const std::vector<unsigned char>& bytes) {
    if (state->cache_capacity == 0 || bytes.empty()) {
        return;
    }
    auto found = state->cache.find(page);
    if (found != state->cache.end()) {
        state->cache_bytes -= found->second->bytes.size();
        state->lru.erase(found->second);
        state->cache.erase(found);
    }
    state->lru.push_back(CachedPage{page, bytes});
    auto iterator = std::prev(state->lru.end());
    state->cache.emplace(page, iterator);
    state->cache_bytes += bytes.size();
    while (state->cache_bytes > state->cache_capacity && !state->lru.empty()) {
        auto oldest = state->lru.begin();
        state->cache_bytes -= oldest->bytes.size();
        state->cache.erase(oldest->index);
        state->lru.erase(oldest);
        state->evicted_pages += 1;
    }
}

bool load_run(ReaderState* state, uint64_t first_page, unsigned page_count,
              std::unordered_map<uint64_t, std::vector<unsigned char>>& pages,
              std::string& error) {
    unsigned remaining = page_count;
    uint64_t page = first_page;
    while (remaining > 0) {
        const unsigned batch_pages = std::min(remaining, state->max_batch_pages);
        const size_t requested = static_cast<size_t>(batch_pages) * state->alignment;
        if (requested > static_cast<size_t>(UINT32_MAX)) {
            error = "one io_uring read exceeds the UAPI 32-bit length";
            return false;
        }
        void* aligned = nullptr;
        const int allocation = posix_memalign(&aligned, state->alignment, requested);
        if (allocation != 0 || aligned == nullptr) {
            error = "posix_memalign failed: " + std::string(strerror(allocation));
            return false;
        }
        memset(aligned, 0, requested);
        const auto started = std::chrono::steady_clock::now();
        const int count = ring_read(
            state->ring, state->file_fd, aligned, static_cast<unsigned>(requested),
            page * state->alignment, state->next_user_data++, error);
        const auto ended = std::chrono::steady_clock::now();
        state->wait_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
        if (count < 0) {
            // A failed enter/wait can leave an SQE owned by the kernel. Destroy
            // the ring before releasing its userspace buffer so an exceptional
            // I/O path cannot become a use-after-free.
            close_state(state);
            free(aligned);
            return false;
        }
        state->storage_bytes += static_cast<uint64_t>(count);
        const auto* raw = static_cast<const unsigned char*>(aligned);
        for (unsigned index = 0; index < batch_pages; ++index) {
            const size_t start = static_cast<size_t>(index) * state->alignment;
            if (start >= static_cast<size_t>(count)) {
                break;
            }
            const size_t available = std::min(
                state->alignment, static_cast<size_t>(count) - start);
            std::vector<unsigned char> bytes(raw + start, raw + start + available);
            pages.emplace(page + index, bytes);
            insert_cache(state, page + index, bytes);
        }
        free(aligned);
        page += batch_pages;
        remaining -= batch_pages;
    }
    return true;
}

bool load_missing_pages_batched(
        ReaderState* state, const std::vector<uint64_t>& missing,
        std::unordered_map<uint64_t, std::vector<unsigned char>>& pages,
        std::string& error) {
    size_t cursor = 0;
    while (cursor < missing.size()) {
        std::vector<PendingRead> pending;
        std::vector<std::pair<uint64_t, unsigned>> layouts;
        pending.reserve(state->queue_depth);
        layouts.reserve(state->queue_depth);
        while (cursor < missing.size() && pending.size() < state->queue_depth) {
            const uint64_t first = missing[cursor];
            unsigned count = 1;
            while (
                cursor + count < missing.size()
                && missing[cursor + count] == missing[cursor + count - 1] + 1
                && count < state->max_batch_pages) {
                ++count;
            }
            const size_t requested = static_cast<size_t>(count) * state->alignment;
            if (requested > static_cast<size_t>(UINT32_MAX)) {
                error = "one io_uring read exceeds the UAPI 32-bit length";
                for (const auto& request : pending) {
                    free(request.buffer);
                }
                return false;
            }
            void* aligned = nullptr;
            const int allocation = posix_memalign(
                &aligned, state->alignment, requested);
            if (allocation != 0 || aligned == nullptr) {
                error = "posix_memalign failed: " +
                        std::string(strerror(allocation));
                for (const auto& request : pending) {
                    free(request.buffer);
                }
                return false;
            }
            memset(aligned, 0, requested);
            pending.push_back(PendingRead{
                aligned,
                static_cast<unsigned>(requested),
                first * state->alignment,
                state->next_user_data++,
            });
            layouts.emplace_back(first, count);
            cursor += count;
        }

        std::vector<int> results;
        const auto started = std::chrono::steady_clock::now();
        const bool success = ring_read_batch(
            state->ring, state->file_fd, pending, results, error);
        const auto ended = std::chrono::steady_clock::now();
        state->wait_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
        if (!success) {
            // ring_read_batch may fail after some SQEs were submitted. Closing
            // the reader first cancels/drains those requests before their
            // aligned buffers are released below.
            close_state(state);
            for (const auto& request : pending) {
                free(request.buffer);
            }
            return false;
        }
        state->submission_batches += 1;
        state->submitted_sqes += pending.size();
        for (size_t request_index = 0; request_index < pending.size(); ++request_index) {
            const int result = results[request_index];
            state->storage_bytes += static_cast<uint64_t>(result);
            const auto* raw = static_cast<const unsigned char*>(
                pending[request_index].buffer);
            const uint64_t first = layouts[request_index].first;
            const unsigned count = layouts[request_index].second;
            for (unsigned index = 0; index < count; ++index) {
                const size_t start = static_cast<size_t>(index) * state->alignment;
                if (start >= static_cast<size_t>(result)) {
                    break;
                }
                const size_t available = std::min(
                    state->alignment, static_cast<size_t>(result) - start);
                std::vector<unsigned char> bytes(
                    raw + start, raw + start + available);
                pages.emplace(first + index, bytes);
                insert_cache(state, first + index, bytes);
            }
            free(pending[request_index].buffer);
        }
    }
    return true;
}

bool assemble_span(
        const std::unordered_map<uint64_t, std::vector<unsigned char>>& pages,
        size_t alignment, uint64_t offset, uint64_t length,
        std::string& output, std::string& error) {
    if (length == 0) {
        output.clear();
        return true;
    }
    const uint64_t first_page = offset / alignment;
    const uint64_t last_page = (offset + length - 1) / alignment;
    output.resize(static_cast<size_t>(length));
    size_t written = 0;
    for (uint64_t page = first_page; page <= last_page; ++page) {
        auto found = pages.find(page);
        if (found == pages.end()) {
            error = "io_uring returned an incomplete page set";
            return false;
        }
        const uint64_t page_start = page * alignment;
        const size_t within = offset > page_start
            ? static_cast<size_t>(offset - page_start) : 0;
        const size_t available = found->second.size() > within
            ? found->second.size() - within : 0;
        const size_t wanted = std::min(
            available, static_cast<size_t>(length) - written);
        if (wanted == 0) {
            error = "io_uring page is shorter than the requested span";
            return false;
        }
        memcpy(output.data() + written, found->second.data() + within, wanted);
        written += wanted;
    }
    if (written != static_cast<size_t>(length)) {
        error = "io_uring read did not fill the requested span";
        return false;
    }
    return true;
}

bool read_many_bytes(
        ReaderState* state,
        const std::vector<std::pair<uint64_t, uint64_t>>& spans,
        std::vector<std::string>& outputs, std::string& error) {
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->closed) {
        error = "cannot read from a closed io_uring reader";
        return false;
    }
    std::vector<uint64_t> needed;
    uint64_t requested_total = 0;
    for (const auto& span : spans) {
        const uint64_t offset = span.first;
        const uint64_t length = span.second;
        if (offset > state->file_size || length > state->file_size - offset) {
            error = "read span is outside the file";
            return false;
        }
        requested_total += length;
        if (length == 0) {
            continue;
        }
        const uint64_t first = offset / state->alignment;
        const uint64_t last = (offset + length - 1) / state->alignment;
        for (uint64_t page = first; page <= last; ++page) {
            needed.push_back(page);
        }
    }
    state->read_calls += spans.size();
    state->requested_bytes += requested_total;
    std::sort(needed.begin(), needed.end());
    needed.erase(std::unique(needed.begin(), needed.end()), needed.end());

    std::unordered_map<uint64_t, std::vector<unsigned char>> pages;
    pages.reserve(needed.size());
    std::vector<uint64_t> missing;
    missing.reserve(needed.size());
    for (uint64_t page : needed) {
        auto found = state->cache.find(page);
        if (found == state->cache.end()) {
            state->cache_miss_pages += 1;
            missing.push_back(page);
        } else {
            state->cache_hit_pages += 1;
            pages.emplace(page, found->second->bytes);
            state->lru.splice(state->lru.end(), state->lru, found->second);
        }
    }
    if (!load_missing_pages_batched(state, missing, pages, error)) {
        return false;
    }
    outputs.resize(spans.size());
    for (size_t index = 0; index < spans.size(); ++index) {
        if (!assemble_span(
                pages, state->alignment, spans[index].first, spans[index].second,
                outputs[index], error)) {
            return false;
        }
    }
    return true;
}

bool read_bytes(ReaderState* state, uint64_t offset, uint64_t length,
                std::string& output, std::string& error) {
    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->closed) {
        error = "cannot read from a closed io_uring reader";
        return false;
    }
    if (offset > state->file_size || length > state->file_size - offset) {
        error = "read span is outside the file";
        return false;
    }
    state->read_calls += 1;
    state->requested_bytes += length;
    if (length == 0) {
        output.clear();
        return true;
    }
    const uint64_t first_page = offset / state->alignment;
    const uint64_t last_page = (offset + length - 1) / state->alignment;
    std::unordered_map<uint64_t, std::vector<unsigned char>> pages;
    std::vector<uint64_t> missing;
    pages.reserve(static_cast<size_t>(last_page - first_page + 1));
    for (uint64_t page = first_page; page <= last_page; ++page) {
        auto found = state->cache.find(page);
        if (found == state->cache.end()) {
            state->cache_miss_pages += 1;
            missing.push_back(page);
        } else {
            state->cache_hit_pages += 1;
            pages.emplace(page, found->second->bytes);
            state->lru.splice(state->lru.end(), state->lru, found->second);
        }
    }
    size_t cursor = 0;
    while (cursor < missing.size()) {
        const uint64_t first = missing[cursor];
        size_t end = cursor + 1;
        while (end < missing.size() && missing[end] == missing[end - 1] + 1) {
            ++end;
        }
        if (!load_run(state, first, static_cast<unsigned>(end - cursor), pages, error)) {
            return false;
        }
        cursor = end;
    }

    output.resize(static_cast<size_t>(length));
    size_t written = 0;
    for (uint64_t page = first_page; page <= last_page; ++page) {
        auto found = pages.find(page);
        if (found == pages.end()) {
            error = "io_uring returned an incomplete page set";
            return false;
        }
        const uint64_t page_start = page * state->alignment;
        const size_t within = offset > page_start
            ? static_cast<size_t>(offset - page_start) : 0;
        const size_t available = found->second.size() > within
            ? found->second.size() - within : 0;
        const size_t wanted = std::min(
            available, static_cast<size_t>(length) - written);
        if (wanted == 0) {
            error = "io_uring page is shorter than the requested span";
            return false;
        }
        memcpy(output.data() + written, found->second.data() + within, wanted);
        written += wanted;
    }
    if (written != static_cast<size_t>(length)) {
        error = "io_uring read did not fill the requested span";
        return false;
    }
    return true;
}

typedef struct {
    PyObject_HEAD
    ReaderState* state;
} ReaderObject;

int Reader_init(ReaderObject* self, PyObject* args, PyObject* kwargs) {
    const char* path = nullptr;
    unsigned long long alignment = 4096;
    unsigned long long cache_capacity = 4ULL * 1024 * 1024 * 1024;
    unsigned queue_depth = 512;
    unsigned max_batch_pages = 4096;
    static const char* names[] = {
        "path", "alignment", "cache_capacity_bytes", "queue_depth",
        "max_batch_pages", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "s|KKII", const_cast<char**>(names), &path,
            &alignment, &cache_capacity, &queue_depth, &max_batch_pages)) {
        return -1;
    }
    if (!valid_power_of_two(static_cast<size_t>(alignment))) {
        PyErr_SetString(PyExc_ValueError,
                        "alignment must be a power of two and at least 512");
        return -1;
    }
    if (queue_depth == 0 || queue_depth > 4096) {
        PyErr_SetString(PyExc_ValueError, "queue_depth must be in [1, 4096]");
        return -1;
    }
    if (max_batch_pages == 0 || max_batch_pages > 4096) {
        PyErr_SetString(PyExc_ValueError, "max_batch_pages must be in [1, 4096]");
        return -1;
    }
    auto* state = new (std::nothrow) ReaderState();
    if (state == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    state->alignment = static_cast<size_t>(alignment);
    state->cache_capacity = static_cast<size_t>(cache_capacity);
    state->queue_depth = queue_depth;
    state->max_batch_pages = max_batch_pages;
    state->file_fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (state->file_fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        delete state;
        return -1;
    }
    struct stat status{};
    if (fstat(state->file_fd, &status) != 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        close(state->file_fd);
        delete state;
        return -1;
    }
    state->file_size = static_cast<uint64_t>(status.st_size);
    std::string error;
    if (!setup_ring(state->ring, queue_depth, error)) {
        PyErr_SetString(PyExc_OSError, error.c_str());
        close(state->file_fd);
        delete state;
        return -1;
    }
    self->state = state;
    return 0;
}

void Reader_dealloc(ReaderObject* self) {
    if (self->state != nullptr) {
        {
            std::lock_guard<std::mutex> guard(self->state->mutex);
            close_state(self->state);
        }
        delete self->state;
        self->state = nullptr;
    }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Reader_read(ReaderObject* self, PyObject* args) {
    unsigned long long offset = 0;
    unsigned long long length = 0;
    if (!PyArg_ParseTuple(args, "KK", &offset, &length)) {
        return nullptr;
    }
    std::string output;
    std::string error;
    bool success = false;
    Py_BEGIN_ALLOW_THREADS
    success = read_bytes(self->state, offset, length, output, error);
    Py_END_ALLOW_THREADS
    if (!success) {
        PyErr_SetString(PyExc_OSError, error.c_str());
        return nullptr;
    }
    return PyBytes_FromStringAndSize(output.data(),
                                     static_cast<Py_ssize_t>(output.size()));
}

PyObject* Reader_read_many(ReaderObject* self, PyObject* argument) {
    PyObject* sequence = PySequence_Fast(
        argument, "read_many expects a sequence of (offset, length) pairs");
    if (sequence == nullptr) {
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    spans.reserve(static_cast<size_t>(count));
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence, index);
        PyObject* pair = PySequence_Fast(
            item, "each read_many span must be an (offset, length) pair");
        if (pair == nullptr) {
            Py_DECREF(sequence);
            return nullptr;
        }
        if (PySequence_Fast_GET_SIZE(pair) != 2) {
            Py_DECREF(pair);
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_ValueError,
                            "each read_many span must contain exactly two integers");
            return nullptr;
        }
        const unsigned long long offset = PyLong_AsUnsignedLongLong(
            PySequence_Fast_GET_ITEM(pair, 0));
        const unsigned long long length = PyLong_AsUnsignedLongLong(
            PySequence_Fast_GET_ITEM(pair, 1));
        Py_DECREF(pair);
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            return nullptr;
        }
        spans.emplace_back(offset, length);
    }
    Py_DECREF(sequence);

    std::vector<std::string> outputs;
    std::string error;
    bool success = false;
    Py_BEGIN_ALLOW_THREADS
    success = read_many_bytes(self->state, spans, outputs, error);
    Py_END_ALLOW_THREADS
    if (!success) {
        PyErr_SetString(PyExc_OSError, error.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(outputs.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
        PyObject* bytes = PyBytes_FromStringAndSize(
            outputs[index].data(), static_cast<Py_ssize_t>(outputs[index].size()));
        if (bytes == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(index), bytes);
    }
    return result;
}

PyObject* Reader_close(ReaderObject* self, PyObject*) {
    if (self->state != nullptr) {
        std::lock_guard<std::mutex> guard(self->state->mutex);
        close_state(self->state);
    }
    Py_RETURN_NONE;
}

PyObject* Reader_telemetry(ReaderObject* self, PyObject*) {
    if (self->state == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "reader state is unavailable");
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(self->state->mutex);
    PyObject* result = PyDict_New();
    if (result == nullptr) {
        return nullptr;
    }
    const std::pair<const char*, uint64_t> values[] = {
        {"read_calls", self->state->read_calls},
        {"requested_bytes", self->state->requested_bytes},
        {"storage_bytes", self->state->storage_bytes},
        {"cache_hit_pages", self->state->cache_hit_pages},
        {"cache_miss_pages", self->state->cache_miss_pages},
        {"cache_entries", self->state->cache.size()},
        {"cache_bytes", self->state->cache_bytes},
        {"cache_capacity_bytes", self->state->cache_capacity},
        {"evicted_pages", self->state->evicted_pages},
        {"wait_ns", self->state->wait_ns},
        {"submission_batches", self->state->submission_batches},
        {"submitted_sqes", self->state->submitted_sqes},
    };
    for (const auto& value : values) {
        PyObject* number = PyLong_FromUnsignedLongLong(value.second);
        if (number == nullptr || PyDict_SetItemString(result, value.first, number) != 0) {
            Py_XDECREF(number);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(number);
    }
    return result;
}

PyObject* Reader_get_closed(ReaderObject* self, void*) {
    if (self->state == nullptr || self->state->closed) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* Reader_get_size(ReaderObject* self, void*) {
    return PyLong_FromUnsignedLongLong(
        self->state == nullptr ? 0 : self->state->file_size);
}

PyMethodDef Reader_methods[] = {
    {"read", reinterpret_cast<PyCFunction>(Reader_read), METH_VARARGS,
     "Read one byte span through io_uring + O_DIRECT."},
    {"read_many", reinterpret_cast<PyCFunction>(Reader_read_many), METH_O,
     "Batch byte spans through one or more io_uring submissions."},
    {"telemetry", reinterpret_cast<PyCFunction>(Reader_telemetry), METH_NOARGS,
     "Return native reader/cache counters."},
    {"close", reinterpret_cast<PyCFunction>(Reader_close), METH_NOARGS,
     "Close the ring and O_DIRECT descriptor."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef Reader_getset[] = {
    {const_cast<char*>("closed"),
     reinterpret_cast<getter>(Reader_get_closed), nullptr,
     const_cast<char*>("whether the reader is closed"), nullptr},
    {const_cast<char*>("size"),
     reinterpret_cast<getter>(Reader_get_size), nullptr,
     const_cast<char*>("file size in bytes"), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyTypeObject ReaderType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_ple_io_uring",
    "Native io_uring + O_DIRECT PLE page reader.",
    -1,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__ple_io_uring() {
    ReaderType.tp_name = "freetoken.models.qwen4_exp._ple_io_uring.IoUringReader";
    ReaderType.tp_basicsize = sizeof(ReaderObject);
    ReaderType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    ReaderType.tp_doc = "Native io_uring + O_DIRECT reader with a bounded C++ LRU.";
    ReaderType.tp_new = PyType_GenericNew;
    ReaderType.tp_init = reinterpret_cast<initproc>(Reader_init);
    ReaderType.tp_dealloc = reinterpret_cast<destructor>(Reader_dealloc);
    ReaderType.tp_methods = Reader_methods;
    ReaderType.tp_getset = Reader_getset;
    if (PyType_Ready(&ReaderType) < 0) {
        return nullptr;
    }
    PyObject* result = PyModule_Create(&module);
    if (result == nullptr) {
        return nullptr;
    }
    Py_INCREF(&ReaderType);
    if (PyModule_AddObject(result, "IoUringReader",
                           reinterpret_cast<PyObject*>(&ReaderType)) < 0) {
        Py_DECREF(&ReaderType);
        Py_DECREF(result);
        return nullptr;
    }
    if (PyModule_AddIntConstant(result, "UAPI_VERSION", 1) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
