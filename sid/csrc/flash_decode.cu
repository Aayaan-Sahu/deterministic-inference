// sid flash-decode kernels.
//
// Three kernels, all bf16 in / fp32 accumulate:
//   1. flash_decode_stage1: split-KV decode attention (flash-decoding style).
//      Grid (num_kv_heads, batch, max_splits). Each block computes the partial
//      attention output of one (request, kv-head, kv-split) for BOTH q heads of
//      the GQA group, using online softmax with fixed-order reductions.
//   2. flash_decode_stage2: merges the per-split partials with the standard
//      LSE rescale, looping over splits sequentially (fixed reduction order).
//   3. reshape_and_cache: scatters new K/V rows into the paged KV pool via a
//      slot mapping. Disjoint writes => trivially deterministic.
//
// Determinism note: for a FIXED (seq_len, num_splits) the reduction order in
// both stages is fully determined; nondeterminism across batch compositions
// enters only through the host-side choice of num_splits (see
// sid/kernels/decode_attention.py). Kernels never use atomics.
//
// Layout assumptions (checked on the host):
//   q:        [B, NUM_Q_HEADS, 128] bf16 contiguous
//   k_cache:  [num_slots, NUM_KV_HEADS, 128] bf16 contiguous
//   v_cache:  [num_slots, NUM_KV_HEADS, 128] bf16 contiguous
//   kv_indptr:  int32 [B+1]
//   kv_indices: int32 [total_kv]   (KV pool slot of each position, in order)
//   num_splits: int32 [B]
//   part_o:   fp32 [B, NUM_Q_HEADS, max_splits, 128]
//   part_lse: fp32 [B, NUM_Q_HEADS, max_splits]
//   o:        [B, NUM_Q_HEADS, 128] bf16
//
// GQA mapping: q head h reads kv head h / QPK (QPK = q heads per kv head = 2).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int QPK = 2;       // q heads per kv head (16 q / 8 kv)
constexpr int TILE_N = 32;   // kv tokens per inner tile
constexpr int NUM_THREADS = 128;
constexpr float NEG_INF = -1e30f;

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Split geometry. MUST be bit-identical between stage1 and stage2.
// fixed_split_size > 0  => batch-invariant mode: tiles of fixed_split_size
//                          tokens regardless of batch composition.
// fixed_split_size == 0 => heuristic mode: seq_len divided into num_splits
//                          chunks rounded up to a multiple of TILE_N.
__host__ __device__ __forceinline__ int kv_per_split(
    int seq_len, int n_splits, int fixed_split_size) {
  if (fixed_split_size > 0) return fixed_split_size;
  return ceil_div(ceil_div(seq_len, n_splits), TILE_N) * TILE_N;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  // Fixed butterfly order => deterministic.
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

__global__ void flash_decode_stage1_kernel(
    const __nv_bfloat16* __restrict__ q,         // [B, HQ, 128]
    const __nv_bfloat16* __restrict__ k_cache,   // [S, HKV, 128]
    const __nv_bfloat16* __restrict__ v_cache,   // [S, HKV, 128]
    const int32_t* __restrict__ kv_indptr,       // [B+1]
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ num_splits,      // [B]
    int fixed_split_size,
    float sm_scale,
    int num_kv_heads,
    int part_stride,                             // splits-dim stride of workspace
    float* __restrict__ part_o,                  // [B, HQ, part_stride, 128]
    float* __restrict__ part_lse) {              // [B, HQ, part_stride]
  const int h_kv = blockIdx.x;
  const int b = blockIdx.y;
  const int split_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid % 32;
  const int num_q_heads = num_kv_heads * QPK;

  const int kv_base = kv_indptr[b];
  const int seq_len = kv_indptr[b + 1] - kv_base;
  const int n_splits = num_splits[b];
  if (split_id >= n_splits) return;

  const int chunk = kv_per_split(seq_len, n_splits, fixed_split_size);
  const int split_start = split_id * chunk;
  const int split_end = min(split_start + chunk, seq_len);
  if (split_start >= seq_len) return;

  __shared__ __nv_bfloat16 sK[TILE_N][HEAD_DIM];
  __shared__ __nv_bfloat16 sV[TILE_N][HEAD_DIM];
  __shared__ float sQ[QPK][HEAD_DIM];
  __shared__ float sP[QPK][TILE_N];   // logits, then probabilities
  __shared__ float sM[QPK];           // running max
  __shared__ float sSum[QPK];         // running exp-sum
  __shared__ float sRescale[QPK];     // per-tile rescale factor

  // Load q for the QPK heads of this group into shared memory as fp32.
  // QPK * HEAD_DIM = 256 floats; NUM_THREADS = 128 => 2 elements per thread.
#pragma unroll
  for (int idx = tid; idx < QPK * HEAD_DIM; idx += NUM_THREADS) {
    const int h = idx / HEAD_DIM;
    const int d = idx % HEAD_DIM;
    const int hq = h_kv * QPK + h;
    sQ[h][d] = __bfloat162float(q[((int64_t)b * num_q_heads + hq) * HEAD_DIM + d]);
  }
  if (tid < QPK) {
    sM[tid] = NEG_INF;
    sSum[tid] = 0.0f;
  }
  __syncthreads();

  // Per-thread output accumulators: thread i owns output dim i for both heads.
  float acc[QPK];
#pragma unroll
  for (int h = 0; h < QPK; ++h) acc[h] = 0.0f;

  for (int tile_start = split_start; tile_start < split_end; tile_start += TILE_N) {
    const int valid = min(TILE_N, split_end - tile_start);

    // --- load K/V tile: 32 rows x 128 dims bf16 = 512 uint4 chunks, 4/thread.
#pragma unroll
    for (int c = 0; c < (TILE_N * HEAD_DIM * 2 / 16) / NUM_THREADS; ++c) {
      const int chunk_id = tid + c * NUM_THREADS;   // 0..511
      const int row = chunk_id / (HEAD_DIM * 2 / 16);      // /16 uint4 per row
      const int col = chunk_id % (HEAD_DIM * 2 / 16);      // uint4 index in row
      if (row < valid) {
        const int pos = tile_start + row;
        const int64_t slot = (int64_t)kv_indices[kv_base + pos];
        const int64_t base = (slot * num_kv_heads + h_kv) * HEAD_DIM;
        reinterpret_cast<uint4*>(&sK[row][0])[col] =
            reinterpret_cast<const uint4*>(&k_cache[base])[col];
        reinterpret_cast<uint4*>(&sV[row][0])[col] =
            reinterpret_cast<const uint4*>(&v_cache[base])[col];
      }
    }
    __syncthreads();

    // --- logits: warp w handles tokens 8w..8w+7; lane covers 4 dims.
    // 8 tokens/warp * 2 heads = 16 shuffle-reduced dot products per warp.
    {
      const int tokens_per_warp = TILE_N / 4;  // 8
#pragma unroll
      for (int h = 0; h < QPK; ++h) {
        for (int tt = 0; tt < tokens_per_warp; ++tt) {
          const int t = warp_id * tokens_per_warp + tt;
          float partial = 0.0f;
          if (t < valid) {
#pragma unroll
            for (int dd = 0; dd < HEAD_DIM / 32; ++dd) {  // 4 dims per lane
              const int d = lane * (HEAD_DIM / 32) + dd;
              partial += sQ[h][d] * __bfloat162float(sK[t][d]);
            }
          }
          const float dot = warp_reduce_sum(partial);
          if (lane == 0 && t < valid) {
            sP[h][t] = dot * sm_scale;
          }
        }
      }
    }
    __syncthreads();

    // --- online softmax stats: warp h (h < QPK) handles head h.
    if (warp_id < QPK) {
      const int h = warp_id;
      const float logit = (lane < valid) ? sP[h][lane] : NEG_INF;
      const float tile_max = warp_reduce_max(logit);
      const float m_old = sM[h];
      const float m_new = fmaxf(m_old, tile_max);
      const float p = (lane < valid) ? __expf(logit - m_new) : 0.0f;
      const float tile_sum = warp_reduce_sum(p);
      if (lane < valid) sP[h][lane] = p;
      if (lane == 0) {
        const float rescale = __expf(m_old - m_new);
        sRescale[h] = rescale;
        sM[h] = m_new;
        sSum[h] = sSum[h] * rescale + tile_sum;
      }
    }
    __syncthreads();

    // --- accumulate V: thread i owns dim i; sequential over tokens (fixed order).
#pragma unroll
    for (int h = 0; h < QPK; ++h) {
      float a = acc[h] * sRescale[h];
      for (int t = 0; t < valid; ++t) {
        a += sP[h][t] * __bfloat162float(sV[t][tid]);
      }
      acc[h] = a;
    }
    __syncthreads();  // before next tile overwrites sK/sV/sP
  }

  // --- epilogue: normalized partial output + logsumexp.
#pragma unroll
  for (int h = 0; h < QPK; ++h) {
    const int hq = h_kv * QPK + h;
    const float e_sum = sSum[h];
    const float inv = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;
    part_o[(((int64_t)b * num_q_heads + hq) * part_stride + split_id) * HEAD_DIM + tid] =
        acc[h] * inv;
    if (tid == 0) {
      part_lse[((int64_t)b * num_q_heads + hq) * part_stride + split_id] =
          sM[h] + logf(e_sum);
    }
  }
}

__global__ void flash_decode_stage2_kernel(
    const float* __restrict__ part_o,    // [B, HQ, part_stride, 128]
    const float* __restrict__ part_lse,  // [B, HQ, part_stride]
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ num_splits,
    int fixed_split_size,
    int num_q_heads,
    int part_stride,
    __nv_bfloat16* __restrict__ o) {     // [B, HQ, 128]
  const int hq = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;  // owns output dim tid

  const int seq_len = kv_indptr[b + 1] - kv_indptr[b];
  const int n_splits = num_splits[b];
  const int chunk = kv_per_split(seq_len, n_splits, fixed_split_size);

  const float* lse_row = &part_lse[((int64_t)b * num_q_heads + hq) * part_stride];
  const float* o_row = &part_o[(((int64_t)b * num_q_heads + hq) * part_stride) * HEAD_DIM];

  // Pass 1: global max lse over valid splits (sequential, fixed order).
  float m_g = NEG_INF;
  for (int s = 0; s < n_splits; ++s) {
    if (s * chunk >= seq_len) break;
    m_g = fmaxf(m_g, lse_row[s]);
  }

  // Pass 2: weighted merge (sequential, fixed order).
  float acc = 0.0f;
  float den = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    if (s * chunk >= seq_len) break;
    const float w = __expf(lse_row[s] - m_g);
    acc += w * o_row[s * HEAD_DIM + tid];
    den += w;
  }
  o[((int64_t)b * num_q_heads + hq) * HEAD_DIM + tid] =
      __float2bfloat16(acc / den);
}

__global__ void reshape_and_cache_kernel(
    const __nv_bfloat16* __restrict__ k,   // [T, HKV, 128]
    const __nv_bfloat16* __restrict__ v,   // [T, HKV, 128]
    __nv_bfloat16* __restrict__ k_cache,   // [S, HKV, 128]
    __nv_bfloat16* __restrict__ v_cache,   // [S, HKV, 128]
    const int64_t* __restrict__ slot_mapping,  // [T]
    int num_kv_heads) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;  // padding safety

  const int row_bytes_u4 = num_kv_heads * HEAD_DIM * 2 / 16;  // uint4 per token
  const int64_t src_base = (int64_t)t * num_kv_heads * HEAD_DIM;
  const int64_t dst_base = slot * num_kv_heads * HEAD_DIM;
  for (int c = tid; c < row_bytes_u4; c += blockDim.x) {
    reinterpret_cast<uint4*>(&k_cache[dst_base])[c] =
        reinterpret_cast<const uint4*>(&k[src_base])[c];
    reinterpret_cast<uint4*>(&v_cache[dst_base])[c] =
        reinterpret_cast<const uint4*>(&v[src_base])[c];
  }
}

void check_kv_layout(const torch::Tensor& t, int num_kv_heads, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(t.dim() == 3 && t.size(1) == num_kv_heads && t.size(2) == HEAD_DIM,
              name, " must be [slots, ", num_kv_heads, ", ", HEAD_DIM, "]");
}

}  // namespace

void flash_decode_fwd(
    torch::Tensor q,           // [B, HQ, 128] bf16
    torch::Tensor k_cache,     // [S, HKV, 128] bf16
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,   // int32 [B+1]
    torch::Tensor kv_indices,  // int32
    torch::Tensor num_splits,  // int32 [B]
    int64_t max_splits,        // grid bound: >= max(num_splits)
    int64_t fixed_split_size,
    double sm_scale,
    torch::Tensor part_o,      // fp32 [>=B, HQ, part_stride, 128] workspace
    torch::Tensor part_lse,    // fp32 [>=B, HQ, part_stride]
    torch::Tensor o) {         // [B, HQ, 128] bf16 out
  const at::cuda::CUDAGuard guard(q.device());
  const int B = q.size(0);
  const int num_q_heads = q.size(1);
  const int num_kv_heads = k_cache.size(1);

  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(q.size(2) == HEAD_DIM, "head_dim must be ", HEAD_DIM);
  TORCH_CHECK(num_q_heads == QPK * num_kv_heads, "expected ", QPK, " q heads per kv head");
  check_kv_layout(k_cache, num_kv_heads, "k_cache");
  check_kv_layout(v_cache, num_kv_heads, "v_cache");
  TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32 && kv_indptr.numel() == B + 1);
  TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32);
  TORCH_CHECK(num_splits.scalar_type() == torch::kInt32 && num_splits.numel() == B);
  TORCH_CHECK(part_o.scalar_type() == torch::kFloat32 && part_o.is_contiguous());
  TORCH_CHECK(part_lse.scalar_type() == torch::kFloat32 && part_lse.is_contiguous());
  const int part_stride = part_o.size(2);
  TORCH_CHECK(part_o.size(0) >= B && part_o.size(1) == num_q_heads &&
              part_stride >= max_splits && part_o.size(3) == HEAD_DIM,
              "part_o workspace too small / wrong shape");
  TORCH_CHECK(part_lse.size(0) >= B && part_lse.size(1) == num_q_heads &&
              part_lse.size(2) == part_stride);
  TORCH_CHECK(o.is_contiguous() && o.sizes() == q.sizes());

  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid1(num_kv_heads, B, (unsigned)max_splits);
  flash_decode_stage1_kernel<<<grid1, NUM_THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
      kv_indptr.data_ptr<int32_t>(),
      kv_indices.data_ptr<int32_t>(),
      num_splits.data_ptr<int32_t>(),
      (int)fixed_split_size,
      (float)sm_scale,
      num_kv_heads,
      part_stride,
      part_o.data_ptr<float>(),
      part_lse.data_ptr<float>());

  dim3 grid2(num_q_heads, B);
  flash_decode_stage2_kernel<<<grid2, NUM_THREADS, 0, stream>>>(
      part_o.data_ptr<float>(),
      part_lse.data_ptr<float>(),
      kv_indptr.data_ptr<int32_t>(),
      num_splits.data_ptr<int32_t>(),
      (int)fixed_split_size,
      num_q_heads,
      part_stride,
      reinterpret_cast<__nv_bfloat16*>(o.data_ptr()));
}

void reshape_and_cache(
    torch::Tensor k,             // [T, HKV, 128] bf16
    torch::Tensor v,
    torch::Tensor k_cache,       // [S, HKV, 128] bf16
    torch::Tensor v_cache,
    torch::Tensor slot_mapping)  // int64 [T]
{
  const at::cuda::CUDAGuard guard(k.device());
  const int T = k.size(0);
  const int num_kv_heads = k_cache.size(1);
  TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(v.is_contiguous() && v.sizes() == k.sizes());
  TORCH_CHECK(k.dim() == 3 && k.size(1) == num_kv_heads && k.size(2) == HEAD_DIM);
  check_kv_layout(k_cache, num_kv_heads, "k_cache");
  check_kv_layout(v_cache, num_kv_heads, "v_cache");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64 && slot_mapping.numel() == T);
  if (T == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  reshape_and_cache_kernel<<<T, NUM_THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
      slot_mapping.data_ptr<int64_t>(),
      num_kv_heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_decode_fwd", &flash_decode_fwd,
        "split-KV flash decode attention (stage1 + stage2)");
  m.def("reshape_and_cache", &reshape_and_cache,
        "scatter new K/V into the paged KV pool");
}
