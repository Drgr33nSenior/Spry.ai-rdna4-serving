// Experimental UltraQuant-derived storage, with software decode and gfx12 FP16
// WMMA QK. This is not the paper's CDNA4 scaled-MFMA implementation.
#include "uq.h"
#include "bf16_math.h"
#include "../codec/codec.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstring>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "This experimental kernel supports only gfx1201"
#endif

namespace {
constexpr int kMaxQueries = 4096;
constexpr int kMaxSlots = 1048576;
constexpr int kMaxContext = 131072;
constexpr float kInputLimit = 1024.0f;
bool shape(int slots, int heads, int dim) {
    return slots > 1 && slots <= kMaxSlots && heads > 0 && heads <= 64 &&
           uq::supported_head_dim(dim);
}
__device__ int record_bytes(int dim) { return dim / 32 * 17; }
__device__ bool input_ok(half x) {
    const float v = __half2float(x);
    return isfinite(v) && fabsf(v) <= kInputLimit;
}
__device__ float unpack(const uint8_t* record, int channel) {
    return __half2float(__float2half_rn(static_cast<float>(
        uq::decode_scalar_unchecked(record + channel / 32 * 17, channel % 32))));
}
__global__ void validate_append(const half* k, const half* v, const int64_t* loc,
                               int heads, int dim, int slots, int* status) {
    const int t = blockIdx.x;
    if (loc[t] <= 0 || loc[t] >= slots) atomicExch(status, 1);
    for (int other = threadIdx.x; other < t; other += blockDim.x)
        if (loc[other] == loc[t]) atomicExch(status, 1);
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x)
        if (!input_ok(k[t * heads * dim + i]) || !input_ok(v[t * heads * dim + i]))
            atomicExch(status, 1);
}
__global__ void append(const half* k, const half* v, const int64_t* loc,
                       uint8_t* packed_k, uint8_t* packed_v, int heads, int dim) {
    const int t = blockIdx.x, h = blockIdx.y;
    // One bounded head. The codec uses double butterflies and returns FP32.
    __shared__ float rotated[256];
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        rotated[i] = __half2float(k[(t * heads + h) * dim + i]);
    __syncthreads();
    if (threadIdx.x == 0) uq::rotate_head_in_place(rotated, dim);
    __syncthreads();
    const int group = threadIdx.x;
    if (group < dim / 32) {
        float values[32];
        for (int i = 0; i < 32; ++i)
            values[i] = __half2float(v[(t * heads + h) * dim + group * 32 + i]);
        const size_t base = (static_cast<size_t>(loc[t]) * heads + h) * record_bytes(dim) + group * 17;
        uq::encode_group(rotated + group * 32, packed_k + base);
        uq::encode_group(values, packed_v + base);
    }
}
__global__ void validate_attention(const half* q, const uint8_t* k, const uint8_t* v,
                                   int slots, int kv_heads, int q_heads, int dim,
                                   const int64_t* indices, int index_count,
                                   const int32_t* starts, const int32_t* lengths,
                                   int* status) {
    const int row = blockIdx.x;
    const int start = starts[row], len = lengths[row];
    if (start < 0 || len <= 0 || len > kMaxContext || start > index_count - len) {
        atomicExch(status, 1); return;
    }
    for (int i = threadIdx.x; i < q_heads * dim; i += blockDim.x)
        if (!input_ok(q[row * q_heads * dim + i])) atomicExch(status, 1);
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const int64_t slot = indices[start + t];
        if (slot <= 0 || slot >= slots) { atomicExch(status, 1); continue; }
        for (int h = 0; h < kv_heads; ++h) {
            const size_t base = (static_cast<size_t>(slot) * kv_heads + h) * record_bytes(dim);
            for (int g = 0; g < dim / 32; ++g) {
                // Appends in this runtime cannot produce a larger scale. Reject
                // uninitialised/corrupt records before reading any attention tile.
                if (uq::validate_block(k + base + g * 17) != uq::Status::kOk ||
                    uq::validate_block(v + base + g * 17) != uq::Status::kOk ||
                    k[base + g * 17 + 16] > 139 || v[base + g * 17 + 16] > 135)
                    atomicExch(status, 1);
            }
        }
    }
}

using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Float8 = float __attribute__((ext_vector_type(8)));
__global__ void attention(const half* q, const uint8_t* k, const uint8_t* v,
                          int kv_heads, int q_heads, int dim, const int64_t* indices,
                          const int32_t* starts, const int32_t* lengths, float scale,
                          half* output) {
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const int kvhead = head / (q_heads / kv_heads);
    const int len = lengths[row], start = starts[row], bytes = record_bytes(dim);
    __shared__ float rotated[256], scores[16], weights[16], alpha, denominator;
    __shared__ float running_max;
    for (int d = lane; d < dim; d += 32)
        rotated[d] = __half2float(q[(row * q_heads + head) * dim + d]);
    __syncthreads();
    if (lane == 0) {
        uq::rotate_head_in_place(rotated, dim);
        running_max = -INFINITY;
        denominator = 0;
    }
    __syncthreads();
    float accum[8] = {};
    for (int tile = 0; tile < len; tile += 16) {
        Float8 product = {};
        for (int dc = 0; dc < dim; dc += 16) {
            Half8 a, b;
            for (int e = 0; e < 8; ++e) {
                const int d = dc + e + (lane / 16) * 8;
                a[e] = static_cast<_Float16>(rotated[d]);
                const int pos = tile + lane % 16;
                b[e] = pos < len ? static_cast<_Float16>(unpack(
                    k + (static_cast<size_t>(indices[start + pos]) * kv_heads + kvhead) * bytes, d)) : 0;
            }
#if defined(__HIP_DEVICE_COMPILE__)
            // gfx12 mapping: A[row=lane%16,k=e+8*(lane/16)],
            // B[k=e+8*(lane/16),col=lane%16]. All A rows contain Q.
            product = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, product);
#else
            // HIP's host pass parses the body but never executes this kernel.
            (void)a;
            (void)b;
#endif
        }
        if (lane < 16) scores[lane] = tile + lane < len ? product[0] * scale : -INFINITY;
        __syncthreads();
        if (lane == 0) {
            float next_max = running_max;
            for (int j = 0; j < 16; ++j) next_max = fmaxf(next_max, scores[j]);
            alpha = expf(running_max - next_max);
            denominator *= alpha;
            for (int j = 0; j < 16; ++j) {
                weights[j] = expf(scores[j] - next_max);
                denominator += weights[j];
            }
            running_max = next_max;
        }
        __syncthreads();
        for (int e = 0, d = lane; d < dim; ++e, d += 32) {
            accum[e] *= alpha;
            for (int j = 0; j < 16 && tile + j < len; ++j) {
                const uint8_t* value = v + (static_cast<size_t>(indices[start + tile + j]) * kv_heads + kvhead) * bytes;
                accum[e] = fmaf(weights[j], unpack(value, d), accum[e]);
            }
        }
        __syncthreads();
    }
    for (int e = 0, d = lane; d < dim; ++e, d += 32)
        output[(row * q_heads + head) * dim + d] = __float2half_rn(accum[e] / denominator);
}

int finish(hipStream_t stream) {
    const hipError_t launch = hipGetLastError();
    const hipError_t completion = hipStreamSynchronize(stream);
    return launch == hipSuccess && completion == hipSuccess ? 0 : 2;
}
int checked_status(int* status, hipStream_t stream) {
    if (finish(stream)) return 2;
    int value = 0;
    if (hipMemcpy(&value, status, sizeof(value), hipMemcpyDeviceToHost) != hipSuccess) return 2;
    return value ? 1 : 0;
}
}
extern "C" int uq_device() {
    int device;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) return 2;
    return std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0 &&
           (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':') ? 0 : 1;
}
extern "C" int uq_initialize(uint8_t* k, uint8_t* v, int slots, int heads, int dim, void* opaque) {
    if (!k || !v || k == v || !shape(slots, heads, dim) || uq_device()) return 1;
    auto stream = static_cast<hipStream_t>(opaque);
    const size_t bytes = static_cast<size_t>(slots) * heads * (dim / 32 * 17);
    if (hipMemsetAsync(k, 255, bytes, stream) != hipSuccess || hipMemsetAsync(v, 255, bytes, stream) != hipSuccess) {
        (void)hipStreamSynchronize(stream); return 2;
    }
    return finish(stream);
}
extern "C" int uq_append(uint8_t* k, uint8_t* v, int slots, int heads, int dim,
                          const uint16_t* new_k, const uint16_t* new_v, const int64_t* loc,
                          int tokens, int* status, void* opaque) {
    if (!k || !v || !new_k || !new_v || !loc || !status || !shape(slots, heads, dim) ||
        tokens <= 0 || tokens > kMaxQueries || uq_device()) return 1;
    auto stream = static_cast<hipStream_t>(opaque);
    if (hipMemsetAsync(status, 0, sizeof(int), stream) != hipSuccess) {
        (void)hipStreamSynchronize(stream); return 2;
    }
    hipLaunchKernelGGL(validate_append, dim3(tokens), dim3(32), 0, stream,
                       reinterpret_cast<const half*>(new_k), reinterpret_cast<const half*>(new_v),
                       loc, heads, dim, slots, status);
    const int check = checked_status(status, stream);
    if (check) return check; // No cache mutation on a validation refusal.
    hipLaunchKernelGGL(append, dim3(tokens, heads), dim3(32), 0, stream,
                       reinterpret_cast<const half*>(new_k), reinterpret_cast<const half*>(new_v),
                       loc, k, v, heads, dim);
    return finish(stream);
}
extern "C" int uq_attention(const uint16_t* q, const uint8_t* k, const uint8_t* v,
                             int slots, int kv_heads, int q_heads, int dim,
                             const int64_t* indices, int count, const int32_t* starts,
                             const int32_t* lengths, int queries, float scale,
                             uint16_t* output, int* status, void* opaque) {
    if (!q || !k || !v || !indices || !starts || !lengths || !output || !status ||
        !shape(slots, kv_heads, dim) || q_heads < kv_heads || q_heads > 128 ||
        q_heads % kv_heads || queries <= 0 || queries > kMaxQueries || count <= 0 ||
        count > kMaxSlots || !std::isfinite(scale) || scale <= 0 || scale > 1 || uq_device()) return 1;
    auto stream = static_cast<hipStream_t>(opaque);
    if (hipMemsetAsync(status, 0, sizeof(int), stream) != hipSuccess) {
        (void)hipStreamSynchronize(stream); return 2;
    }
    hipLaunchKernelGGL(validate_attention, dim3(queries), dim3(32), 0, stream,
                       reinterpret_cast<const half*>(q), k, v, slots, kv_heads, q_heads, dim,
                       indices, count, starts, lengths, status);
    const int check = checked_status(status, stream);
    if (check) return check;
    hipLaunchKernelGGL(attention, dim3(queries, q_heads), dim3(32), 0, stream,
                       reinterpret_cast<const half*>(q), k, v, kv_heads, q_heads, dim,
                       indices, starts, lengths, scale, reinterpret_cast<half*>(output));
    return finish(stream);
}
#include "uq.h"
#include "bf16_math.h"
#include <hip/hip_runtime.h>
#include <cmath>
namespace { int finish(hipStream_t); int checked_status(int*, hipStream_t); }

namespace bf16_v2 {
bool aligned(const void* p, uintptr_t n) { return p && uintptr_t(p)%n==0; }
bool valid(const uq_bf16_cache_v2* c, const uint64_t* identity) {
    if (!c || !identity || !aligned(c->k,2) || !aligned(c->v,2) ||
        c->slots<2 || c->slots>1048576 || c->protected_layer>1) return false;
    const uint64_t bytes=uint64_t(c->slots)*2*(c->protected_layer ? 512 : 136);
    if(c->bytes_per_buffer!=bytes) return false;
    const uintptr_t k=uintptr_t(c->k),v=uintptr_t(c->v);
    if(k>UINTPTR_MAX-bytes || v>UINTPTR_MAX-bytes || (k<v+bytes && v<k+bytes)) return false;
    uint64_t nonzero=0;
    for(int i=0;i<4;++i) { if(c->identity[i]!=identity[i]) return false; nonzero|=identity[i]; }
    return nonzero!=0;
}
bool disjoint(const void* a, uint64_t an, const void* b, uint64_t bn) {
    uintptr_t x=uintptr_t(a),y=uintptr_t(b);
    return x<=UINTPTR_MAX-an && y<=UINTPTR_MAX-bn && !(x<y+bn && y<x+an);
}
bool outside(const uq_bf16_cache_v2* c,const void* p,uint64_t n) {
    return disjoint(c->k,c->bytes_per_buffer,p,n)&&disjoint(c->v,c->bytes_per_buffer,p,n);
}
__global__ void validate_write(const uint16_t* k,const uint16_t* v,const int64_t* loc,
                              [[maybe_unused]] int tokens,int slots,int* status) {
    int t=blockIdx.x;
    if(loc[t]<=0 || loc[t]>=slots) atomicExch(status,1);
    for(int j=threadIdx.x;j<t;j+=blockDim.x) if(loc[t]==loc[j]) atomicExch(status,1);
    for(int j=threadIdx.x;j<512;j+=blockDim.x)
        if(!uq_bf16::input_ok(k[t*512+j]) || !uq_bf16::input_ok(v[t*512+j])) atomicExch(status,1);
}
__global__ void write(uq_bf16_cache_v2 c,const uint16_t* k,const uint16_t* v,const int64_t* loc) {
    int token=blockIdx.x,head=blockIdx.y;
    const uint16_t* kin=k+(token*2+head)*256; const uint16_t* vin=v+(token*2+head)*256;
    size_t record=size_t(loc[token])*2+head;
    if(c.protected_layer) {
        for(int j=0;j<256;++j) { static_cast<uint16_t*>(c.k)[record*256+j]=kin[j];
                              static_cast<uint16_t*>(c.v)[record*256+j]=vin[j]; }
    } else {
        uq_bf16::encode_head(kin,true,static_cast<uint8_t*>(c.k)+record*136);
        uq_bf16::encode_head(vin,false,static_cast<uint8_t*>(c.v)+record*136);
    }
}
__device__ float load(const void* buffer,size_t record,int channel,bool protect) {
    return protect ? uq_bf16::from_bits(static_cast<const uint16_t*>(buffer)[record*256+channel])
                   : uq_bf16::unpack(static_cast<const uint8_t*>(buffer)+record*136,channel);
}
__global__ void validate_read(uq_bf16_cache_v2 c,const uint16_t* q,const int64_t* indices,
                             int count,const int32_t* starts,const int32_t* lengths,int* status) {
    int row=blockIdx.x, start=starts[row],len=lengths[row];
    if(start<0 || len<=0 || len>131072 || start>count || len>count-start) {atomicExch(status,1);return;}
    for(int j=threadIdx.x;j<3072;j+=blockDim.x) if(!uq_bf16::input_ok(q[row*3072+j])) atomicExch(status,1);
    // Flatten each token's two heads so adjacent lanes check adjacent values.
    // At the maximum context, both task counts fit in int (at most 2^26).
    if(c.protected_layer) {
        for(int i=threadIdx.x;i<len*512;i+=blockDim.x) {
            int64_t slot=indices[start+i/512];
            if(slot<=0 || slot>=c.slots) {atomicExch(status,1);continue;}
            size_t offset=size_t(slot)*512+i%512;
            if(!uq_bf16::input_ok(static_cast<const uint16_t*>(c.k)[offset]) ||
               !uq_bf16::input_ok(static_cast<const uint16_t*>(c.v)[offset])) atomicExch(status,1);
        }
    } else {
        for(int i=threadIdx.x;i<len*256;i+=blockDim.x) {
            int64_t slot=indices[start+i/256];
            if(slot<=0 || slot>=c.slots) {atomicExch(status,1);continue;}
            int within=i%256,byte=within%16;
            size_t base=(size_t(slot)*2+within/128)*136+(within%128)/16*17;
            const uint8_t* k=static_cast<const uint8_t*>(c.k)+base;
            const uint8_t* v=static_cast<const uint8_t*>(c.v)+base;
            uint8_t kb=k[byte],vb=v[byte];
            if(!uq::is_canonical_code(kb&15) || !uq::is_canonical_code(kb>>4) ||
               !uq::is_canonical_code(vb&15) || !uq::is_canonical_code(vb>>4)) atomicExch(status,1);
            // One lane per group checks both scales; these bounds reject 255.
            if(byte==0 && (k[16]>139 || v[16]>135)) atomicExch(status,1);
        }
    }
}
// Cooperate over channels; lane 0 preserves ordered FP32 dot/online softmax.
// 2060 shared bytes per head, eight accumulators per lane; no matrix operations.
__global__ void read(uq_bf16_cache_v2 c,const uint16_t* q,const int64_t* indices,
                    const int32_t* starts,const int32_t* lengths,float scale,uint16_t* output) {
    int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;
    __shared__ float rotated[256],products[256],alpha,weight,denominator;
    float acc[8]={},maximum=-INFINITY;
    for(int i=0;i<8;++i) {
        int j=lane+32*i;
        rotated[j]=uq_bf16::from_bits(q[(row*12+head)*256+j]);
    }
    __syncthreads();
    if(lane==0) {
        if(!c.protected_layer) uq_bf16::wht256(rotated);
        denominator=0;
    }
    __syncthreads();
    for(int t=0;t<lengths[row];++t) {
        size_t record=size_t(indices[starts[row]+t])*2+head/6;
        for(int i=0;i<8;++i) {
            int j=lane+32*i;
            products[j]=uq_bf16::mul(rotated[j],load(c.k,record,j,c.protected_layer));
        }
        __syncthreads();
        if(lane==0) {
            float score=0;
            for(int j=0;j<256;++j) score=uq_bf16::add(score,products[j]);
            score=uq_bf16::mul(score,scale);
            float next=fmaxf(maximum,score);
            alpha=expf(maximum-next);weight=expf(score-next);
            denominator=uq_bf16::add(uq_bf16::mul(denominator,alpha),weight);
            maximum=next;
        }
        __syncthreads();
        for(int i=0;i<8;++i) {
            int j=lane+32*i;
            acc[i]=uq_bf16::add(uq_bf16::mul(acc[i],alpha),
                uq_bf16::mul(weight,load(c.v,record,j,c.protected_layer)));
        }
    }
    for(int i=0;i<8;++i) {
        int j=lane+32*i;
        output[(row*12+head)*256+j]=uq_bf16::to_bits(acc[i]/denominator);
    }
}

// Compact cache only: six waves share one decoded KV head. The per-head
// dot, softmax and accumulation order is unchanged. 14,408 shared bytes;
// eight accumulators per lane. All 192 lanes reach every block barrier.
__global__ void read_grouped(uq_bf16_cache_v2 c,const uint16_t* q,const int64_t* indices,
                            const int32_t* starts,const int32_t* lengths,float scale,uint16_t* output) {
    int row=blockIdx.x,kvhead=blockIdx.y,group=threadIdx.x/32,lane=threadIdx.x%32;
    int head=kvhead*6+group;
    __shared__ float rotated[6][256],products[6][256],decoded_k[256],decoded_v[256];
    __shared__ float alpha[6],weight[6],denominator[6];
    float acc[8]={},maximum=-INFINITY;
    // phase: query
    for(int i=0;i<8;++i) {
        int j=lane+32*i;
        rotated[group][j]=uq_bf16::from_bits(q[(row*12+head)*256+j]);
    }
    __syncthreads();
    // phase: rotate
    if(lane==0) {
        uq_bf16::wht256(rotated[group]);
        denominator[group]=0;
    }
    __syncthreads();
    for(int t=0;t<lengths[row];++t) {
        // phase: decode
        if(group==0) {
            size_t record=size_t(indices[starts[row]+t])*2+kvhead;
            for(int i=0;i<8;++i) {
                int j=lane+32*i;
                decoded_k[j]=load(c.k,record,j,false);
                decoded_v[j]=load(c.v,record,j,false);
            }
        }
        __syncthreads();
        // phase: product
        for(int i=0;i<8;++i) {
            int j=lane+32*i;
            products[group][j]=uq_bf16::mul(rotated[group][j],decoded_k[j]);
        }
        __syncthreads();
        // phase: score
        if(lane==0) {
            float score=0;
            for(int j=0;j<256;++j) score=uq_bf16::add(score,products[group][j]);
            score=uq_bf16::mul(score,scale);
            float next=fmaxf(maximum,score);
            alpha[group]=expf(maximum-next);weight[group]=expf(score-next);
            denominator[group]=uq_bf16::add(uq_bf16::mul(denominator[group],alpha[group]),weight[group]);
            maximum=next;
        }
        __syncthreads();
        // phase: accumulate
        for(int i=0;i<8;++i) {
            int j=lane+32*i;
            acc[i]=uq_bf16::add(uq_bf16::mul(acc[i],alpha[group]),
                uq_bf16::mul(weight[group],decoded_v[j]));
        }
        // Do not overwrite decoded_v until all six waves finish consuming it.
        __syncthreads();
    }
    // phase: output
    for(int i=0;i<8;++i) {
        int j=lane+32*i;
        output[(row*12+head)*256+j]=uq_bf16::to_bits(acc[i]/denominator[group]);
    }
}
}
extern "C" int uq_bf16_append_v2(const uq_bf16_cache_v2* c,const uint64_t* identity,
    const uint16_t* k,const uint16_t* v,uint64_t bytes,const int64_t* loc,int tokens,int* status,void* opaque) {
    using namespace bf16_v2;
    if(!valid(c,identity)||tokens<1||tokens>4096||bytes!=uint64_t(tokens)*1024 ||
       !aligned(k,2)||!aligned(v,2)||!aligned(loc,8)||!aligned(status,4) ||
       !outside(c,k,bytes)||!outside(c,v,bytes)||!outside(c,loc,tokens*8)||!outside(c,status,4)||
       !disjoint(status,4,k,bytes)||!disjoint(status,4,v,bytes)||!disjoint(status,4,loc,tokens*8)) return 1;
    int device=uq_device();if(device) return device;
    auto stream=static_cast<hipStream_t>(opaque);
    if(hipMemsetAsync(status,0,4,stream)!=hipSuccess) {(void)hipStreamSynchronize(stream);return 2;}
    hipLaunchKernelGGL(validate_write,dim3(tokens),dim3(32),0,stream,k,v,loc,tokens,c->slots,status);
    int checked=checked_status(status,stream);if(checked) return checked;
    hipLaunchKernelGGL(write,dim3(tokens,2),dim3(1),0,stream,*c,k,v,loc);
    return finish(stream);
}
extern "C" int uq_bf16_attention_v2(const uq_bf16_cache_v2* c,const uint64_t* identity,
    const uint16_t* q,uint64_t qbytes,const int64_t* indices,int count,const int32_t* starts,
    const int32_t* lengths,int queries,float scale,uint16_t* out,uint64_t outbytes,int* status,void* opaque) {
    using namespace bf16_v2;
    if(!valid(c,identity)||queries<1||queries>4096||count<1||count>1048576||
       qbytes!=uint64_t(queries)*6144||outbytes!=qbytes||!std::isfinite(scale)||scale<=0||scale>1||
       !aligned(q,2)||!aligned(out,2)||!aligned(indices,8)||!aligned(starts,4)||!aligned(lengths,4)||!aligned(status,4)) return 1;
    const void* inputs[]={q,indices,starts,lengths};uint64_t sizes[]={qbytes,uint64_t(count)*8,uint64_t(queries)*4,uint64_t(queries)*4};
    if(!outside(c,out,outbytes)||!outside(c,status,4)||!disjoint(out,outbytes,status,4)) return 1;
    for(int i=0;i<4;++i) if(!outside(c,inputs[i],sizes[i])||!disjoint(out,outbytes,inputs[i],sizes[i])||
        !disjoint(status,4,inputs[i],sizes[i])) return 1;
    int device=uq_device();if(device) return device;
    auto stream=static_cast<hipStream_t>(opaque);
    if(hipMemsetAsync(status,0,4,stream)!=hipSuccess) {(void)hipStreamSynchronize(stream);return 2;}
    hipLaunchKernelGGL(validate_read,dim3(queries),dim3(256),0,stream,*c,q,indices,count,starts,lengths,status);
    int checked=checked_status(status,stream);if(checked) return checked;
    if(c->protected_layer) {
        hipLaunchKernelGGL(read,dim3(queries,12),dim3(32),0,stream,*c,q,indices,starts,lengths,scale,out);
    } else {
        hipLaunchKernelGGL(read_grouped,dim3(queries,2),dim3(192),0,stream,*c,q,indices,starts,lengths,scale,out);
    }
    return finish(stream);
}
