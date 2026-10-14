#include "../runtime/bf16_math.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>

static float fp32(uint32_t bits) { float x; std::memcpy(&x,&bits,4); return x; }
int main() {
  // Independent exact bit oracle: BF16 maps sign/exponent/fraction into FP32.
  for(uint32_t b=0;b<65536;++b) {
    float x=uq_bf16::from_bits(uint16_t(b)); uint32_t observed;
    std::memcpy(&observed,&x,4); assert(observed==(b<<16));
    if((b&0x7f80)!=0x7f80) assert(uq_bf16::to_bits(x)==b);
  }
  // RNE midpoint fixtures computed from representable neighbours, not the
  // production bias trick. Cover both signs and normal/subnormal boundaries.
  for(uint32_t high: {0u,1u,0x7fu,0x80u,0x3f80u,0x3f81u,0x7f7eu,0x8000u,0xbf80u}) {
    for(uint32_t low: {0x7fffu,0x8000u,0x8001u}) {
      uint16_t want=uint16_t(high+((low>0x8000u)||(low==0x8000u&&(high&1))));
      assert(uq_bf16::to_bits(fp32((high<<16)|low))==want);
    }
  }
  assert(uq_bf16::to_bits(fp32(0x7f800000))==0x7f80);
  assert(uq_bf16::to_bits(fp32(0xff800000))==0xff80);
  assert((uq_bf16::to_bits(fp32(0x7f800001))&0x7fc0)==0x7fc0);
  assert(uq_bf16::input_ok(0x8000)&&uq_bf16::input_ok(1)&&uq_bf16::input_ok(0x4480));
  assert(!uq_bf16::input_ok(0x4481)&&!uq_bf16::input_ok(0x7f7f)&&!uq_bf16::input_ok(0x7f80));
  float values[256],original[256];
  for(int i=0;i<256;++i) values[i]=original[i]=float((i*17)%31-15)/8;
  uq_bf16::wht256(values);
  for(unsigned i=0;i<256;++i) {
    double sum=0;
    for(unsigned j=0;j<256;++j) sum+=(__builtin_parity(i&j) ? -1:1)*double(original[j]);
    assert(values[i]==float(sum/16)); // dyadic exact reference, no tolerance
  }
  uint16_t ones[256];for(auto& x:ones)x=0x3f80;
  uint8_t guarded[138];std::memset(guarded,0xa5,sizeof guarded);
  assert(uq_bf16::encode_head(ones,true,guarded+1));
  assert(guarded[0]==0xa5&&guarded[137]==0xa5);
  for(int g=0;g<8;++g) for(int j=0;j<17;++j) {
    int want=j==16 ? (g==0 ? 128:127) : (g==0&&j==0 ? 7:0);
    assert(guarded[1+17*g+j]==want);
  }
  for(uint16_t bad: {uint16_t(0x7fc1),uint16_t(0x7f80),uint16_t(0xff80),uint16_t(0x4481)}) {
    ones[255]=bad;std::memset(guarded,0xa5,sizeof guarded);
    assert(!uq_bf16::encode_head(ones,true,guarded+1));
    for(auto b:guarded) assert(b==0xa5);
  }
  std::cout<<"BF16 shared production math: exhaustive bits, RNE, parity WHT, packed bytes and canaries PASS\n";
}
