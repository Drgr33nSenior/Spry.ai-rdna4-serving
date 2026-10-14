#pragma once
#include <cstdint>

// Experimental C ABI, version 1. All pointers except stream are device pointers.
// The caller owns allocations and the HIP stream. Calls synchronize work they
// enqueue before returning; callers must retain buffers until return. A host
// argument refusal enqueues no work and does not synchronize earlier caller work.
// Return 0 on success, 1 for a refused shape/value, 2 for HIP failure.
extern "C" {
int uq_device(); // Refuses devices other than gfx1201; does not launch a kernel.
int uq_initialize(uint8_t* k, uint8_t* v, int slots, int kv_heads, int dim, void* stream);
int uq_append(uint8_t* k, uint8_t* v, int slots, int kv_heads, int dim,
              const uint16_t* new_k, const uint16_t* new_v, const int64_t* locations,
              int tokens, int* status, void* stream);
int uq_attention(const uint16_t* q, const uint8_t* k, const uint8_t* v,
                 int slots, int kv_heads, int q_heads, int dim,
                 const int64_t* indices, int index_count, const int32_t* starts,
                 const int32_t* lengths, int queries, float scale,
                 uint16_t* output, int* status, void* stream);
}

// V2 is a separate, uncompiled source ABI. Descriptor/identity are host objects;
// all buffer pointers are device addresses. BF16 buffers are uint16 bits aligned
// to 2, locations to 8, spans/status to 4. K/V are disjoint, caller-owned, exact
// byte sizes below. No caller may mutate/reuse buffers until synchronized return.
// Same 0=success, 1=validation refusal, 2=HIP error convention. A refusal changes
// only device status; a HIP error is not transactional and poisons the owner.
struct uq_bf16_cache_v2 {
  void* k; void* v; uint64_t bytes_per_buffer;
  uint64_t identity[4]; uint32_t slots; uint32_t protected_layer;
};
extern "C" {
int uq_bf16_append_v2(const uq_bf16_cache_v2* cache, const uint64_t* identity,
  const uint16_t* k, const uint16_t* v, uint64_t input_bytes,
  const int64_t* locations, int tokens, int* status, void* stream);
int uq_bf16_attention_v2(const uq_bf16_cache_v2* cache, const uint64_t* identity,
  const uint16_t* q, uint64_t q_bytes, const int64_t* indices, int index_count,
  const int32_t* starts, const int32_t* lengths, int queries, float scale,
  uint16_t* output, uint64_t output_bytes, int* status, void* stream);
}
