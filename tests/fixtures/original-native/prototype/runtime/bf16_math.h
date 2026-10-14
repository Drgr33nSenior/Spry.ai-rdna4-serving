#pragma once
#include "../codec/codec.h"
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define UQ_HOST_DEVICE __host__ __device__
#else
#define UQ_HOST_DEVICE
#endif

// Source candidate v2. No native FP4 or BF16 matrix instruction is implied.
namespace uq_bf16 {
UQ_HOST_DEVICE inline float from_bits(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16; float out;
  __builtin_memcpy(&out, &bits, sizeof(out)); return out;
}
UQ_HOST_DEVICE inline uint16_t to_bits(float value) {
  uint32_t bits; __builtin_memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
UQ_HOST_DEVICE inline float add(float a, float b) { volatile float v=a+b; return v; }
UQ_HOST_DEVICE inline float sub(float a, float b) { volatile float v=a-b; return v; }
UQ_HOST_DEVICE inline float mul(float a, float b) { volatile float v=a*b; return v; }
UQ_HOST_DEVICE inline bool input_ok(uint16_t bits) {
  return (bits & 0x7fffu) <= 0x4480u; // finite, abs <= 1024; signed zero accepted
}
// Eight ascending radix-two stages, each butterfly rounded to FP32. Normalize
// once by exact 1/16. No BF16 rounding between butterflies; no double scratch.
UQ_HOST_DEVICE inline void wht256(float* values) {
  for (int stride=1; stride<256; stride*=2)
    for (int base=0; base<256; base+=2*stride)
      for (int j=0; j<stride; ++j) {
        float a=values[base+j], b=values[base+j+stride];
        values[base+j]=add(a,b); values[base+j+stride]=sub(a,b);
      }
  for (int j=0; j<256; ++j) values[j]=mul(values[j],0.0625f);
}
UQ_HOST_DEVICE inline float unpack(const uint8_t* record, int channel) {
  const uint8_t* group=record+(channel/32)*17;
  const uint8_t code=(group[(channel%32)/2] >> (4*(channel%2))) & 15;
  // Explicit tile-local BF16 rounding, not FP16 reinterpretation.
  return from_bits(to_bits(uq::decode_fp4_scaled(code,group[16])));
}
UQ_HOST_DEVICE inline bool record_ok(const uint8_t* record, int bound) {
  for(int g=0;g<8;++g)
    if(uq::validate_block(record+17*g)!=uq::Status::kOk || record[17*g+16]>bound) return false;
  return true;
}
UQ_HOST_DEVICE inline bool encode_head(const uint16_t* input, bool rotate, uint8_t* out) {
  float values[256]; uint8_t packed[136];
  for(int i=0;i<256;++i) { if(!input_ok(input[i])) return false; values[i]=from_bits(input[i]); }
  if(rotate) wht256(values);
  for(int g=0;g<8;++g) {
    if(uq::encode_group(values+32*g,packed+17*g)!=uq::Status::kOk) return false;
  }
  for(int i=0;i<136;++i) out[i]=packed[i];
  return true;
}
}
#undef UQ_HOST_DEVICE
