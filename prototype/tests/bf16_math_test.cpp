#include "../runtime/bf16_math.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>

static float fp32(uint32_t bits) { float x; std::memcpy(&x,&bits,4); return x; }
static uint32_t fp32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Preserve the previous production unpack body as the compatibility oracle.
// Expected values still use the generic codec and explicit BF16 roundtrip.
static float generic_unpack(const uint8_t* record, int channel) {
  const uint8_t* group=record+(channel/32)*17;
  const uint8_t code=(group[(channel%32)/2] >> (4*(channel%2))) & 15;
  return uq_bf16::from_bits(uq_bf16::to_bits(uq::decode_fp4_scaled(code,group[16])));
}

struct DecodeCounts {
  std::size_t comparisons = 0;
  std::size_t positive_zero = 0;
  std::size_t negative_zero = 0;
  std::size_t positive_subnormal = 0;
  std::size_t negative_subnormal = 0;
  std::size_t positive_overflow = 0;
  std::size_t negative_overflow = 0;
};

static bool compare_unpack(const uint8_t* record, DecodeCounts& counts) {
  for (int channel = 0; channel < 256; ++channel) {
    const uint32_t expected = fp32_bits(generic_unpack(record, channel));
    const uint32_t actual = fp32_bits(uq_bf16::unpack(record, channel));
    if (actual != expected) {
      std::cerr << "unpack mismatch channel=" << channel
                << " scale=" << unsigned(record[(channel / 32) * 17 + 16])
                << " expected=" << std::hex << expected << " actual=" << actual
                << std::dec << '\n';
      return false;
    }
    ++counts.comparisons;
    const uint32_t magnitude = expected & 0x7fffffffu;
    const bool negative = (expected & 0x80000000u) != 0;
    if (magnitude == 0) {
      ++(negative ? counts.negative_zero : counts.positive_zero);
    } else if ((magnitude & 0x7f800000u) == 0) {
      ++(negative ? counts.negative_subnormal : counts.positive_subnormal);
    } else if (magnitude == 0x7f800000u) {
      ++(negative ? counts.negative_overflow : counts.positive_overflow);
    }
  }
  return true;
}

static bool test_exact_unpack() {
  constexpr std::size_t kCanary = 16;
  constexpr std::size_t kRecordBytes = 136;
  std::array<uint8_t, kCanary + kRecordBytes + kCanary> guarded;
  uint8_t* const record = guarded.data() + kCanary;
  DecodeCounts counts;
  std::size_t canonical_bounded_comparisons = 0;
  std::size_t rejection_checks = 0;
  // Every nibble and scale reaches every channel in both nibble positions.
  // Different adjacent nibbles expose low/high extraction mistakes.
  for (unsigned scale = 0; scale < 256; ++scale) {
    for (unsigned code = 0; code < 16; ++code) {
      for (unsigned parity = 0; parity < 2; ++parity) {
        guarded.fill(0xa5);
        const unsigned neighbor = (code + 7u) & 15u;
        const uint8_t packed = parity == 0
            ? uint8_t(code | (neighbor << 4)) : uint8_t(neighbor | (code << 4));
        for (unsigned group = 0; group < 8; ++group) {
          std::memset(record + group * 17, packed, 16);
          record[group * 17 + 16] = uint8_t(scale);
        }
        const auto before = guarded;
        if (!compare_unpack(record, counts)) return false;
        assert(guarded == before);
        if (code != 8 && neighbor != 8 && scale <= 139)
          canonical_bounded_comparisons += 256;
        for (int bound : {135, 139}) {
          const bool expected = scale <= unsigned(bound) && code != 8 && neighbor != 8;
          assert(uq_bf16::record_ok(record, bound) == expected);
          ++rejection_checks;
        }
      }
    }
  }
  // Adjacent records have different scales and nibbles in every group. This
  // checks group/record offsets, signs, subnormals and overflow together.
  constexpr unsigned kMixedRecords = 3;
  std::array<uint8_t, kCanary + kMixedRecords * kRecordBytes + kCanary> mixed;
  for (unsigned first = 0; first < 256; ++first) {
    mixed.fill(0xa5);
    for (unsigned index = 0; index < kMixedRecords; ++index) {
      uint8_t* const current = mixed.data() + kCanary + index * kRecordBytes;
      for (unsigned group = 0; group < 8; ++group) {
        current[group * 17 + 16] = uint8_t((first + 37u * group + 83u * index) & 255u);
        for (unsigned pair = 0; pair < 16; ++pair)
          current[group * 17 + pair] = uint8_t((pair + 3u * group + 5u * index) & 15u)
              | uint8_t(((15u - pair + group + 7u * index) & 15u) << 4);
      }
    }
    const auto before = mixed;
    for (unsigned index = 0; index < kMixedRecords; ++index)
      if (!compare_unpack(mixed.data() + kCanary + index * kRecordBytes, counts)) return false;
    assert(mixed == before);
  }
  assert(counts.positive_zero && counts.negative_zero);
  assert(counts.positive_subnormal && counts.negative_subnormal);
  assert(counts.positive_overflow && counts.negative_overflow);
  std::cout << "EXACT_DECODER_CPU_PASS comparisons=" << counts.comparisons
            << " canonical_bounded_comparisons=" << canonical_bounded_comparisons
            << " unchanged_rejection_checks=" << rejection_checks
            << " positive_zero=" << counts.positive_zero
            << " negative_zero=" << counts.negative_zero
            << " positive_subnormal=" << counts.positive_subnormal
            << " negative_subnormal=" << counts.negative_subnormal
            << " positive_overflow=" << counts.positive_overflow
            << " negative_overflow=" << counts.negative_overflow << '\n';
  return true;
}

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
  if (!test_exact_unpack()) return 1;
  std::cout<<"BF16 shared production math: exhaustive bits, RNE, parity WHT, packed bytes and canaries PASS\n";
}
