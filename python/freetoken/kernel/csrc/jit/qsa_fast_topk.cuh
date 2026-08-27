// Modified by Qwen3.8 Next 5090 Lab contributors in 2026.
//
// Apache-2.0 adaptation of SGLang's QSA radix-select kernel at
// https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120
// commit 30edf3503961a471b25150aa890f8166031b5738, file
// python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh.
// The host wrapper is converted to FreeToken's tvm-ffi kernel utilities and
// the specialization is intentionally limited to QSA block-topk 512.

#pragma once

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cuda_fp16.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace qsa_fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
constexpr int kRadix = 256;
constexpr int kHistogramSize = kRadix + 128;
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t); // 32 KiB
constexpr int kCandidateCapacity =
    kSmemBytes / (2 * static_cast<int>(sizeof(int)));

struct Params {
  const float *__restrict__ input;   // [rows, input_stride]
  int32_t *__restrict__ indices;     // [rows, kTopK]
  const int32_t *__restrict__ lengths;
  int64_t input_stride;
};

__device__ __forceinline__ uint8_t coarse_key(float x) {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t ordered =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                      : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(ordered >> 8);
}

__device__ __forceinline__ uint32_t exact_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int kTopK>
__device__ void fill_short_row(int32_t *__restrict__ output, int32_t length) {
  for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock)
    output[i] = i < length ? i : -1;
}

template <int kTopK>
__device__ void exact_radix_rescan(
    const float *__restrict__ input, int32_t *__restrict__ output,
    int32_t length, int (&histograms)[2][kHistogramSize], int &output_count,
    int &threshold_bin, int &tie_count) {
  const int tid = threadIdx.x;
  if (tid == 0)
    output_count = 0;
  __syncthreads();

  // The inherited kernel cached at most 4096 members of its threshold bin.
  // A 65,536-wide row can exceed that capacity, and a larger exact-fp32 value
  // appearing after the clipped prefix would then be silently omitted.  Four
  // exact radix passes rescan the row instead.  They use only a 2x256-bin
  // histogram, have no candidate-capacity failure mode, and preserve the
  // unspecified output ordering of the original atomic collector.
  int remaining = kTopK;
  uint32_t prefix_mask = 0;
  uint32_t prefix_value = 0;
#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    auto &histogram = histograms[0];
    if (tid < kRadix + 1)
      histogram[tid] = 0;
    __syncthreads();

    const int shift = 24 - pass * 8;
    for (int index = tid; index < length; index += kThreadsPerBlock) {
      const uint32_t key = exact_key(input[index]);
      if ((key & prefix_mask) == prefix_value)
        atomicAdd(&histogram[(key >> shift) & 0xff], 1);
    }
    __syncthreads();

#pragma unroll 8
    for (int scan = 0; scan < 8; ++scan) {
      if (tid < kRadix) {
        const int distance = 1 << scan;
        const int source = scan & 1;
        int value = histograms[source][tid];
        if (tid < kRadix - distance)
          value += histograms[source][tid + distance];
        histograms[source ^ 1][tid] = value;
      }
      __syncthreads();
    }

    if (tid < kRadix && histogram[tid] >= remaining &&
        histogram[tid + 1] < remaining)
      threshold_bin = tid;
    __syncthreads();

    const int threshold = threshold_bin;
    const int greater_count = histogram[threshold + 1];
    for (int index = tid; index < length; index += kThreadsPerBlock) {
      const uint32_t key = exact_key(input[index]);
      const int bin = static_cast<int>((key >> shift) & 0xff);
      if ((key & prefix_mask) == prefix_value && bin > threshold) {
        const int slot = atomicAdd(&output_count, 1);
        if (slot < kTopK)
          output[slot] = index;
      }
    }
    __syncthreads();
    remaining -= greater_count;

    if (pass == 3) {
      if (tid == 0)
        tie_count = 0;
      __syncthreads();
      for (int index = tid; index < length; index += kThreadsPerBlock) {
        const uint32_t key = exact_key(input[index]);
        const int bin = static_cast<int>((key >> shift) & 0xff);
        if ((key & prefix_mask) == prefix_value && bin == threshold) {
          const int tie = atomicAdd(&tie_count, 1);
          if (tie < remaining) {
            const int slot = atomicAdd(&output_count, 1);
            if (slot < kTopK)
              output[slot] = index;
          }
        }
      }
      __syncthreads();
    } else {
      prefix_mask |= 0xffu << shift;
      prefix_value |= static_cast<uint32_t>(threshold) << shift;
    }
  }

  for (int slot = output_count + tid; slot < kTopK;
       slot += kThreadsPerBlock)
    output[slot] = -1;
}

template <int kTopK>
__device__ void radix_select(const float *__restrict__ input,
                             int32_t *__restrict__ output, int32_t length) {
  int topk = kTopK;
  alignas(128) __shared__ int histograms[2][kHistogramSize];
  alignas(128) __shared__ int output_count;
  alignas(128) __shared__ int threshold_bin;
  alignas(128) __shared__ int candidate_count[2];
  extern __shared__ int candidate_indices[][kCandidateCapacity];

  const int tid = threadIdx.x;
  auto &histogram = histograms[0];
  if (tid < kRadix + 1)
    histogram[tid] = 0;
  __syncthreads();

  for (int index = tid; index < length; index += kThreadsPerBlock)
    atomicAdd(&histogram[coarse_key(input[index])], 1);
  __syncthreads();

  const auto reverse_cumsum = [&] {
#pragma unroll 8
    for (int pass = 0; pass < 8; ++pass) {
      if (tid < kRadix) {
        const int distance = 1 << pass;
        const int source = pass & 1;
        int value = histograms[source][tid];
        if (tid < kRadix - distance)
          value += histograms[source][tid + distance];
        histograms[source ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  reverse_cumsum();
  if (tid < kRadix && histogram[tid] > topk &&
      histogram[tid + 1] <= topk) {
    threshold_bin = tid;
    candidate_count[0] = 0;
    output_count = 0;
  }
  __syncthreads();

  int threshold = threshold_bin;
  topk -= histogram[threshold + 1];
  if (topk == 0) {
    for (int index = tid; index < length; index += kThreadsPerBlock) {
      if (static_cast<int>(coarse_key(input[index])) > threshold) {
        const int slot = atomicAdd(&output_count, 1);
        output[slot] = index;
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  if (tid < kRadix + 1)
    histogram[tid] = 0;
  __syncthreads();
  for (int index = tid; index < length; index += kThreadsPerBlock) {
    const float value = input[index];
    const int bin = static_cast<int>(coarse_key(value));
    if (bin > threshold) {
      const int slot = atomicAdd(&output_count, 1);
      output[slot] = index;
    } else if (bin == threshold) {
      const int slot = atomicAdd(&candidate_count[0], 1);
      if (slot < kCandidateCapacity) {
        candidate_indices[0][slot] = index;
        atomicAdd(&histogram[(exact_key(value) >> 24) & 0xff], 1);
      }
    }
  }
  __syncthreads();

  // The common path keeps the reference kernel's cheap coarse pass and small
  // candidate working set.  Overflowing rows are uncommon but correctness
  // critical: rescan their full fp32 keys on-device, with no host sync and no
  // silent 4096-entry truncation.
  if (candidate_count[0] > kCandidateCapacity) {
    exact_radix_rescan<kTopK>(input, output, length, histograms, output_count,
                              threshold_bin, candidate_count[1]);
    return;
  }

#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    __shared__ int final_ties;
    const int source = pass & 1;
    const int count = candidate_count[source];
    reverse_cumsum();
    if (tid < kRadix && histogram[tid] > topk &&
        histogram[tid + 1] <= topk) {
      threshold_bin = tid;
      candidate_count[source ^ 1] = 0;
      final_ties = topk - histogram[tid + 1];
    }
    __syncthreads();

    threshold = threshold_bin;
    topk -= histogram[threshold + 1];
    if (topk == 0) {
      for (int i = tid; i < count; i += kThreadsPerBlock) {
        const int index = candidate_indices[source][i];
        const int shift = 24 - pass * 8;
        if (static_cast<int>((exact_key(input[index]) >> shift) & 0xff) >
            threshold) {
          const int slot = atomicAdd(&output_count, 1);
          output[slot] = index;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tid < kRadix + 1)
      histogram[tid] = 0;
    __syncthreads();
    for (int i = tid; i < count; i += kThreadsPerBlock) {
      const int index = candidate_indices[source][i];
      const float value = input[index];
      const int shift = 24 - pass * 8;
      const int bin = static_cast<int>((exact_key(value) >> shift) & 0xff);
      if (bin > threshold) {
        const int slot = atomicAdd(&output_count, 1);
        output[slot] = index;
      } else if (bin == threshold) {
        if (pass == 3) {
          const int remaining = atomicAdd(&final_ties, -1);
          if (remaining > 0)
            output[kTopK - remaining] = index;
        } else {
          const int slot = atomicAdd(&candidate_count[source ^ 1], 1);
          if (slot < kCandidateCapacity) {
            candidate_indices[source ^ 1][slot] = index;
            atomicAdd(&histogram[(exact_key(value) >> (shift - 8)) & 0xff],
                      1);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(kThreadsPerBlock) void
kernel(const Params __grid_constant__ params) {
  device::PDL::wait<kUsePDL>();
  const uint64_t row = static_cast<uint64_t>(blockIdx.x);
  const int32_t raw_length = params.lengths[row];
  const int32_t length = raw_length < 0
                             ? 0
                             : (raw_length > params.input_stride
                                    ? static_cast<int32_t>(params.input_stride)
                                    : raw_length);
  int32_t *output = params.indices + row * kTopK;
  const float *input = params.input + row * params.input_stride;
  if (length <= kTopK)
    fill_short_row<kTopK>(output, length);
  else
    radix_select<kTopK>(input, output, length);
  device::PDL::launch<kUsePDL>();
}

} // namespace qsa_fast_topk_detail

template <int kTopK, bool kUsePDL> struct QSAFastTopKKernel {
  static_assert(kTopK == 512, "FreeToken QSA only validates block top-k 512");

  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView lengths,
                  const tvm::ffi::TensorView indices) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto stride = SymbolicSize{"stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, width})
        .with_strides({stride, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({rows})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);
    TensorMatcher({rows, kTopK})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);

    const auto params = qsa_fast_topk_detail::Params{
        .input = static_cast<const float *>(scores.data_ptr()),
        .indices = static_cast<int32_t *>(indices.data_ptr()),
        .lengths = static_cast<const int32_t *>(lengths.data_ptr()),
        .input_stride = stride.unwrap(),
    };
    const auto function = qsa_fast_topk_detail::kernel<kTopK, kUsePDL>;
    LaunchKernel(static_cast<uint32_t>(rows.unwrap()),
                 qsa_fast_topk_detail::kThreadsPerBlock, device.unwrap(),
                 qsa_fast_topk_detail::kSmemBytes)
        .with_attr(kUsePDL)(function, params);
  }
};
