#include "../codec/codec.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed\\n", __FILE__, __LINE__, \
                   #condition);                                                \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

constexpr std::array<double, 8> kOracleMagnitudes = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

int oracle_round_even(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) return static_cast<int>(lower);
  if (fraction > 0.5) return static_cast<int>(lower + 1.0);
  const int integer = static_cast<int>(lower);
  return (integer % 2 == 0) ? integer : integer + 1;
}

std::uint8_t oracle_nearest_code(double value) {
  std::uint8_t selected = 0;
  double distance = std::fabs(value - kOracleMagnitudes[0]);
  for (std::uint8_t candidate = 1; candidate != 8; ++candidate) {
    const double candidate_distance =
        std::fabs(value - kOracleMagnitudes[candidate]);
    if (candidate_distance < distance ||
        (candidate_distance == distance && (candidate % 2) == 0)) {
      selected = candidate;
      distance = candidate_distance;
    }
  }
  return selected;
}

// Deliberately independent implementation of the documented representation.
// It must not call a uq encoder or decoder.
bool oracle_encode(const std::array<float, 32>& input,
                   std::array<std::uint8_t, 17>* output) {
  double maximum = 0.0;
  for (float value : input) {
    if (!std::isfinite(value)) return false;
    maximum = std::max(maximum, std::fabs(static_cast<double>(value)));
  }
  if (maximum == 0.0) {
    output->fill(0);
    (*output)[16] = 127;
    return true;
  }
  int exponent = oracle_round_even(std::log2(0.156 * maximum));
  exponent = std::max(-127, std::min(127, exponent));
  const double scale = std::ldexp(1.0, exponent);
  for (std::size_t pair = 0; pair != 16; ++pair) {
    std::uint8_t byte = 0;
    for (std::size_t lane = 0; lane != 2; ++lane) {
      const float value = input[pair * 2 + lane];
      const std::uint8_t magnitude =
          oracle_nearest_code(std::fabs(static_cast<double>(value)) / scale);
      const std::uint8_t code = magnitude == 0
                                    ? 0
                                    : static_cast<std::uint8_t>(
                                          magnitude | (std::signbit(value) ? 8 : 0));
      byte |= static_cast<std::uint8_t>(code << (4 * lane));
    }
    (*output)[pair] = byte;
  }
  (*output)[16] = static_cast<std::uint8_t>(exponent + 127);
  return true;
}

double oracle_decode(const std::array<std::uint8_t, 17>& encoded,
                     std::size_t lane) {
  const std::uint8_t byte = encoded[lane / 2];
  const std::uint8_t code = (lane % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
  const double signed_value = (code & 8) ? -kOracleMagnitudes[code & 7]
                                          : kOracleMagnitudes[code & 7];
  return std::ldexp(signed_value, static_cast<int>(encoded[16]) - 127);
}

void test_golden_layout_and_oracle() {
  std::array<float, 32> input{};
  input[0] = 0.0F;
  input[1] = 0.03125F;
  input[2] = 0.09375F;
  input[3] = 0.125F;
  input[4] = 0.1875F;
  input[5] = 0.25F;
  input[6] = 0.375F;
  input[7] = 0.5F;
  input[8] = 0.75F;
  input[9] = 1.0F;
  input[10] = -0.125F;
  input[11] = -0.1875F;
  std::array<std::uint8_t, 17> expected{};
  CHECK(oracle_encode(input, &expected));
  const std::array<std::uint8_t, 17> golden = {
      0x00, 0x22, 0x43, 0x65, 0x77, 0xba, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 124};
  CHECK(expected == golden);
  std::array<std::uint8_t, 17> actual{};
  CHECK(uq::encode_group(input.data(), actual.data()) == uq::Status::kOk);
  CHECK(actual == expected);
  for (std::size_t i = 0; i != 32; ++i) {
    uq::Status status = uq::Status::kOk;
    const double decoded = uq::decode_scalar_reference(actual.data(), i, &status);
    CHECK(status == uq::Status::kOk);
    CHECK(decoded == oracle_decode(expected, i));
  }
}

void test_ties_and_limits() {
  CHECK(uq::round_nearest_even(-2.5) == -2);
  CHECK(uq::round_nearest_even(-1.5) == -2);
  CHECK(uq::round_nearest_even(0.5) == 0);
  CHECK(uq::round_nearest_even(1.5) == 2);
  CHECK(uq::round_nearest_even(2.5) == 2);
  CHECK(uq::nearest_magnitude_code(0.25) == 0);
  CHECK(uq::nearest_magnitude_code(0.75) == 2);
  CHECK(uq::nearest_magnitude_code(1.25) == 2);
  CHECK(uq::nearest_magnitude_code(1.75) == 4);
  CHECK(uq::nearest_magnitude_code(5.0) == 6);

  std::array<float, 32> zeros{};
  zeros[0] = -0.0F;
  std::array<std::uint8_t, 17> encoded{};
  CHECK(uq::encode_group(zeros.data(), encoded.data()) == uq::Status::kOk);
  for (std::size_t i = 0; i != 16; ++i) CHECK(encoded[i] == 0);
  CHECK(encoded[16] == 127);

  std::array<float, 32> tiny{};
  tiny.fill(std::numeric_limits<float>::denorm_min());
  CHECK(uq::encode_group(tiny.data(), encoded.data()) == uq::Status::kOk);
  CHECK(encoded[16] == 0);
  for (std::size_t i = 0; i != 16; ++i) CHECK(encoded[i] == 0);

  std::array<float, 32> huge{};
  huge.fill(std::numeric_limits<float>::max());
  std::array<std::uint8_t, 17> oracle{};
  CHECK(oracle_encode(huge, &oracle));
  CHECK(uq::encode_group(huge.data(), encoded.data()) == uq::Status::kOk);
  CHECK(encoded == oracle);

  std::array<std::uint8_t, 17> extreme{};
  extreme[0] = 0x07;
  extreme[16] = 254;
  uq::Status status = uq::Status::kOk;
  const double high = uq::decode_scalar_reference(extreme.data(), 0, &status);
  CHECK(status == uq::Status::kOk);
  CHECK(high == std::ldexp(6.0, 127));
  float narrow = 0.0F;
  CHECK(uq::decode_scalar(extreme.data(), 0, &narrow) == uq::Status::kOk);
  CHECK(std::isinf(narrow));
}

void test_rejections() {
  std::array<float, 32> input{};
  std::array<std::uint8_t, 17> output{};
  output.fill(0xa5);
  input[7] = std::numeric_limits<float>::infinity();
  CHECK(uq::encode_group(input.data(), output.data()) == uq::Status::kNonFiniteInput);
  for (std::uint8_t byte : output) CHECK(byte == 0xa5);
  input[7] = std::numeric_limits<float>::quiet_NaN();
  CHECK(uq::encode_group(input.data(), output.data()) == uq::Status::kNonFiniteInput);
  CHECK(uq::encode_group(nullptr, output.data()) == uq::Status::kInvalidArgument);

  std::array<std::uint8_t, 17> malformed{};
  malformed[16] = 255;
  CHECK(uq::validate_block(malformed.data()) == uq::Status::kMalformedEncoding);
  malformed[16] = 127;
  malformed[0] = 0x08;
  CHECK(uq::validate_block(malformed.data()) == uq::Status::kMalformedEncoding);
  float ignored = 0.0F;
  CHECK(uq::decode_scalar(malformed.data(), 32, &ignored) ==
        uq::Status::kMalformedEncoding);
  malformed[0] = 0;
  CHECK(uq::decode_scalar(malformed.data(), 32, &ignored) ==
        uq::Status::kInvalidArgument);

  std::size_t bytes = 0;
  CHECK(!uq::encoded_head_bytes(33, &bytes));
  CHECK(uq::encode_head(input.data(), 33, output.data()) ==
        uq::Status::kUnsupportedDimension);
  CHECK(uq::rotate_head_in_place(input.data(), 33) ==
        uq::Status::kUnsupportedDimension);

  std::array<float, 32> overflowing{};
  overflowing.fill(std::numeric_limits<float>::max());
  const auto original = overflowing;
  CHECK(uq::rotate_head_in_place(overflowing.data(), overflowing.size()) ==
        uq::Status::kOutOfRange);
  CHECK(overflowing == original);
}

std::array<double, 32> parity_hadamard(const std::array<double, 32>& input) {
  std::array<double, 32> output{};
  for (std::size_t row = 0; row != 32; ++row) {
    for (std::size_t column = 0; column != 32; ++column) {
      const unsigned parity = static_cast<unsigned>(row & column);
      const bool negative = (__builtin_popcount(parity) & 1) != 0;
      output[row] += negative ? -input[column] : input[column];
    }
    output[row] /= std::sqrt(32.0);
  }
  return output;
}

void apply_rope(std::array<float, 32>* values) {
  for (std::size_t pair = 0; pair != 16; ++pair) {
    const double angle = 0.03125 * static_cast<double>(pair + 1);
    const float left = (*values)[2 * pair];
    const float right = (*values)[2 * pair + 1];
    (*values)[2 * pair] = static_cast<float>(left * std::cos(angle) - right * std::sin(angle));
    (*values)[2 * pair + 1] = static_cast<float>(left * std::sin(angle) + right * std::cos(angle));
  }
}

double dot(const std::array<float, 32>& left, const std::array<float, 32>& right) {
  double result = 0.0;
  for (std::size_t i = 0; i != 32; ++i) result += static_cast<double>(left[i]) * right[i];
  return result;
}

void test_rotation_after_rope() {
  std::array<float, 32> query{};
  std::array<float, 32> key{};
  for (std::size_t i = 0; i != 32; ++i) {
    query[i] = static_cast<float>(static_cast<int>(i) - 11) / 19.0F;
    key[i] = static_cast<float>(7 - static_cast<int>(i)) / 23.0F;
  }
  apply_rope(&query);
  apply_rope(&key);
  const double before = dot(query, key);
  std::array<double, 32> query_double{};
  for (std::size_t i = 0; i != 32; ++i) query_double[i] = query[i];
  const auto expected = parity_hadamard(query_double);
  CHECK(uq::rotate_head_in_place(query.data(), query.size()) == uq::Status::kOk);
  CHECK(uq::rotate_head_in_place(key.data(), key.size()) == uq::Status::kOk);
  for (std::size_t i = 0; i != 32; ++i) CHECK(std::fabs(query[i] - expected[i]) < 2e-7);
  CHECK(std::fabs(dot(query, key) - before) < 4e-7);
}

void test_head_size_and_paged_reuse() {
  std::size_t bytes = 0;
  CHECK(uq::encoded_head_bytes(32, &bytes) && bytes == 17);
  CHECK(uq::encoded_head_bytes(64, &bytes) && bytes == 34);
  CHECK(uq::encoded_head_bytes(128, &bytes) && bytes == 68);
  CHECK(uq::encoded_head_bytes(256, &bytes) && bytes == 136);

  std::array<std::array<float, 32>, 4> keys{};
  std::array<std::array<float, 32>, 4> values{};
  std::array<std::array<std::uint8_t, 17>, 4> compressed_keys{};
  std::array<std::array<std::uint8_t, 17>, 4> compressed_values{};
  for (std::size_t token = 0; token != 4; ++token) {
    for (std::size_t lane = 0; lane != 32; ++lane) {
      keys[token][lane] = static_cast<float>(
          static_cast<int>((token + 1) * (lane % 5)) - 6) / 17.0F;
      values[token][lane] = static_cast<float>(
          static_cast<int>((token + 2) * (lane % 7)) - 8) / 21.0F;
    }
    CHECK(uq::rotate_head_in_place(keys[token].data(), 32) == uq::Status::kOk);
    CHECK(uq::encode_group(keys[token].data(), compressed_keys[token].data()) == uq::Status::kOk);
    CHECK(uq::encode_group(values[token].data(), compressed_values[token].data()) == uq::Status::kOk);
  }
  // Page table reuses token 0 after an append on the second page.
  const std::array<std::size_t, 4> page_table = {2, 0, 3, 0};
  std::array<float, 32> query{};
  for (std::size_t lane = 0; lane != 32; ++lane) {
    query[lane] = static_cast<float>(static_cast<int>(lane) - 13) / 29.0F;
  }
  CHECK(uq::rotate_head_in_place(query.data(), 32) == uq::Status::kOk);
  std::array<double, 4> scores{};
  for (std::size_t row = 0; row != page_table.size(); ++row) {
    for (std::size_t lane = 0; lane != 32; ++lane) {
      scores[row] += static_cast<double>(query[lane]) *
                     oracle_decode(compressed_keys[page_table[row]], lane);
    }
    scores[row] /= std::sqrt(32.0);
  }
  const double maximum = std::max(std::max(scores[0], scores[1]), std::max(scores[2], scores[3]));
  double total = 0.0;
  for (double& score : scores) { score = std::exp(score - maximum); total += score; }
  std::array<double, 32> attention{};
  for (std::size_t row = 0; row != page_table.size(); ++row) {
    for (std::size_t lane = 0; lane != 32; ++lane) {
      attention[lane] += (scores[row] / total) *
                         oracle_decode(compressed_values[page_table[row]], lane);
    }
  }
  CHECK(std::isfinite(attention[0]));
  CHECK(std::isfinite(attention[31]));
  CHECK(page_table[1] == page_table[3]);
}

// Promoted from sealed kv4-parallel-20261012.01iXhb/tests/focused.cpp.
// Research manifest: 45c89669030b6a58c3e9a6f8c68c5d6ed183837d78f3fa91b71598046b4c461e.
// Fixed vLLM vectors derive from Apache-2.0 sources at
// ee8f7ea4493bf95e7c04951aec701af4000abdb7; their notices/licence are
// retained in the sealed research and CPU-contract evidence roots.
// WHT and softmax tolerances are unchanged from that predeclared CPU plan.
using Head = std::array<float, 256>;
using Packed = std::array<std::uint8_t, 136>;
using Wide = std::array<double, 256>;
double rotation_error = 0, dot_error = 0, attention_error = 0;
int incompatible_tie_bytes = 0;

Packed independent_encode_head(const Head& input) {
  Packed packed{};
  for (std::size_t g = 0; g < 8; ++g) {
    std::array<float, 32> block{};
    std::copy_n(input.begin() + g * 32, 32, block.begin());
    std::array<std::uint8_t, 17> encoded{};
    CHECK(oracle_encode(block, &encoded));
    std::copy(encoded.begin(), encoded.end(), packed.begin() + g * 17);
  }
  return packed;
}

double independent_decode(const Packed& p, std::size_t d) {
  std::array<std::uint8_t, 17> block{};
  std::copy_n(p.begin() + (d / 32) * 17, 17, block.begin());
  return oracle_decode(block, d % 32);
}

Wide matrix_hadamard(const Head& input) {
  Wide output{};
  for (unsigned row = 0; row < 256; ++row)
    for (unsigned col = 0; col < 256; ++col)
      output[row] += ((__builtin_popcount(row & col) & 1) ? -1.0 : 1.0)
                     * static_cast<double>(input[col]) / 16.0;
  return output;
}

void partial_rope(Head& input, int position) {
  // Synthetic NeoX pairing of 64/256 channels; not an MRoPE implementation.
  for (int i = 0; i < 32; ++i) {
    const double angle = static_cast<double>((i + 1) * position) / 101.0;
    const double a = input[i], b = input[i + 32];
    input[i] = static_cast<float>(a * std::cos(angle) - b * std::sin(angle));
    input[i + 32] = static_cast<float>(a * std::sin(angle) + b * std::cos(angle));
  }
}

double dot256(const Head& a, const Head& b) {
  double result = 0;
  for (int d = 0; d < 256; ++d) result += static_cast<double>(a[d]) * b[d];
  return result;
}

void head256_bytes_and_rejections() {
  Head h{};
  for (int d = 0; d < 256; ++d)
    h[d] = std::ldexp(static_cast<float>((d * 13) % 31 - 15) / 8.0F, d / 32 - 4);
  std::array<std::uint8_t, 138> guarded{};
  guarded.fill(0xa5);
  const Packed expected = independent_encode_head(h);
  CHECK(uq::encode_head(h.data(), h.size(), guarded.data() + 1) == uq::Status::kOk);
  CHECK(guarded.front() == 0xa5 && guarded.back() == 0xa5);
  CHECK(std::equal(expected.begin(), expected.end(), guarded.begin() + 1));
  Head decoded{};
  CHECK(uq::decode_head(expected.data(), 256, decoded.data()) == uq::Status::kOk);
  for (int d = 0; d < 256; ++d) CHECK(decoded[d] == independent_decode(expected, d));

  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    Head bad = h;
    bad[255] = invalid;
    guarded.fill(0xa5);
    CHECK(uq::encode_head(bad.data(), 256, guarded.data() + 1) == uq::Status::kNonFiniteInput);
    for (auto byte : guarded) CHECK(byte == 0xa5);
  }
  Head bad = h;
  bad[255] = std::numeric_limits<float>::infinity();
  const Head before = bad;
  CHECK(uq::rotate_head_in_place(bad.data(), 256) == uq::Status::kNonFiniteInput);
  CHECK(bad == before);
  bad.fill(std::numeric_limits<float>::max());
  const Head overflow_before = bad;
  CHECK(uq::rotate_head_in_place(bad.data(), 256) == uq::Status::kOutOfRange);
  CHECK(bad == overflow_before);
  CHECK(uq::encode_head(h.data(), 255, guarded.data() + 1) == uq::Status::kUnsupportedDimension);
  std::puts("head256 exact packing, independent oracle, guards, late-invalid and rotation range: PASS");
}

void fixed_vllm_differences() {
  std::array<float, 32> input = {
      .25F, .75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.F, 6.F,
      -.25F, -.75F, -1.25F, -1.75F, -2.5F, -3.5F, -5.F, -6.F};
  const std::array<std::uint8_t, 17> local = {
      0x20, 0x42, 0x64, 0x76, 0xa0, 0xca, 0xec, 0xfe,
      0, 0, 0, 0, 0, 0, 0, 0, 127};
  // reference.py bucketize(right=False), triton_store.py strict '>' midpoints.
  const std::array<std::uint8_t, 17> vllm = {
      0x10, 0x32, 0x54, 0x76, 0xa9, 0xcb, 0xed, 0xff,
      0, 0, 0, 0, 0, 0, 0, 0, 127};
  std::array<std::uint8_t, 17> actual{}, independent{};
  CHECK(uq::encode_group(input.data(), actual.data()) == uq::Status::kOk);
  CHECK(oracle_encode(input, &independent));
  CHECK(actual == local && independent == local);
  for (int i = 0; i < 17; ++i) incompatible_tie_bytes += actual[i] != vllm[i];
  CHECK(incompatible_tie_bytes == 7);
  const std::array<float, 8> magnitudes = {0, .5F, 1, 1.5F, 2, 3, 4, 6};
  const std::array<float, 16> lower_values = {
      0, .5F, 1, 1.5F, 2, 3, 4, 6, -.5F, -1, -1.5F, -2, -3, -4, -6, -6};
  for (int d = 0; d < 16; ++d) {
    const auto code = (vllm[d / 2] >> ((d % 2) * 4)) & 15;
    CHECK(((code & 8) ? -magnitudes[code & 7] : magnitudes[code & 7]) == lower_values[d]);
  }
  input.fill(0);
  input[0] = -0.F;
  CHECK(uq::encode_group(input.data(), actual.data()) == uq::Status::kOk);
  CHECK(actual[16] == 127); // vLLM zero group: zero codes and scale byte 0.
  for (int i = 0; i < 16; ++i) CHECK(actual[i] == 0);
  for (double tie : {-3.5, -1.5, .5, 2.5}) {
    CHECK(uq::round_nearest_even(tie) == oracle_round_even(tie));
    CHECK(uq::round_nearest_even(tie) != std::floor(tie + .5));
  }
  input.fill(std::ldexp(1.0F, -124));
  CHECK(uq::encode_group(input.data(), actual.data()) == uq::Status::kOk);
  for (int i = 0; i < 16; ++i) CHECK(actual[i] == 0x77);
  CHECK(actual[16] == 0);
  uq::Status status{};
  CHECK(uq::decode_scalar_reference(actual.data(), 0, &status) == std::ldexp(6.0, -127));
  CHECK(status == uq::Status::kOk);
  // vLLM tensor reference clamps exponent to -126: codes 0x66, byte 1.
  // Triton store uses unclamped -127: codes 0x77, byte 0.
  // Its V reader treats byte 0 as zero. Those sources disagree at this boundary.
  CHECK(std::ldexp(4.0, -126) != std::ldexp(6.0, -127));
  std::puts("vLLM fixed source vectors: expected tie/zero/exponent/range incompatibilities CONFIRMED");
}

void head256_layout_and_rotation() {
  Head q{}, k{}, v{};
  for (int d = 0; d < 256; ++d) {
    q[d] = static_cast<float>((d * 7) % 23 - 11) / 13.0F;
    k[d] = static_cast<float>((d * 11) % 29 - 14) / 17.0F;
    v[d] = std::ldexp(static_cast<float>(d % 13 - 6) / 8.0F, d / 32 - 3);
  }
  partial_rope(q, 4);
  partial_rope(k, 2);
  const auto expected_q = matrix_hadamard(q);
  const auto expected_k = matrix_hadamard(k);
  const auto before_dot = dot256(q, k);
  const Head original_v = v;
  CHECK(uq::rotate_head_in_place(q.data(), 256) == uq::Status::kOk);
  CHECK(uq::rotate_head_in_place(k.data(), 256) == uq::Status::kOk);
  for (int d = 0; d < 256; ++d) {
    rotation_error = std::max(rotation_error, std::fabs(q[d] - expected_q[d]));
    rotation_error = std::max(rotation_error, std::fabs(k[d] - expected_k[d]));
  }
  CHECK(rotation_error <= 2e-6);
  dot_error = std::fabs(dot256(q, k) - before_dot);
  CHECK(dot_error <= 1e-5 + 2e-7 * std::fabs(before_dot));
  CHECK(v == original_v);
  Packed pk{}, pv{};
  CHECK(uq::encode_head(k.data(), 256, pk.data()) == uq::Status::kOk);
  CHECK(uq::encode_head(v.data(), 256, pv.data()) == uq::Status::kOk);
  CHECK(pk == independent_encode_head(k));
  CHECK(pv == independent_encode_head(original_v));
  std::array<std::uint8_t, 272> interleaved{}, planar{};
  std::copy(pk.begin(), pk.end(), interleaved.begin());
  std::copy(pv.begin(), pv.end(), interleaved.begin() + 136);
  // Explicit fixture-only layout permutation; it does not convert semantics.
  for (int g = 0; g < 8; ++g) {
    std::copy_n(pk.begin() + g * 17, 16, planar.begin() + g * 16);
    planar[128 + g] = pk[g * 17 + 16];
    std::copy_n(pv.begin() + g * 17, 16, planar.begin() + 136 + g * 16);
    planar[264 + g] = pv[g * 17 + 16];
  }
  CHECK(interleaved != planar);
  for (int d = 0; d < 256; ++d) {
    for (int side = 0; side < 2; ++side) {
      const int codes_base = side ? 136 : 0, scales_base = side ? 264 : 128;
      const auto code = (planar[codes_base + d / 2] >> ((d % 2) * 4)) & 15;
      const double value = std::ldexp((code & 8 ? -1 : 1) * kOracleMagnitudes[code & 7],
                                     static_cast<int>(planar[scales_base + d / 32]) - 127);
      CHECK(value == independent_decode(side ? pv : pk, d));
    }
  }
  std::puts("head256 post-partial-RoPE parity rotation, unrotated V and byte-layout permutation: PASS");
}

using Pool = std::array<Packed, 44>;
Wide attention(const Head& q, const Pool& keys, const Pool& values,
               const std::vector<int>& slots, int visible, bool reference) {
  Wide out{};
  std::vector<double> scores(visible);
  auto decode = [reference](const Packed& p, int d) {
    if (reference) return independent_decode(p, d);
    float value = 0;
    CHECK(uq::decode_scalar(p.data() + (d / 32) * 17, d % 32, &value) == uq::Status::kOk);
    return static_cast<double>(value);
  };
  for (int t = 0; t < visible; ++t)
    for (int d = 0; d < 256; ++d)
      scores[t] += static_cast<double>(q[d]) * decode(keys[slots[t]], d) / 16.0;
  if (reference) {
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double sum = 0;
    for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
    for (int t = 0; t < visible; ++t)
      for (int d = 0; d < 256; ++d)
        out[d] += scores[t] / sum * decode(values[slots[t]], d);
  } else {
    double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
    for (int start = 0; start < visible; start += 16) {
      const int stop = std::min(visible, start + 16);
      const double next = std::max(maximum, *std::max_element(scores.begin() + start, scores.begin() + stop));
      const double alpha = std::exp(maximum - next);
      sum *= alpha;
      for (auto& value : out) value *= alpha;
      for (int t = start; t < stop; ++t) {
        const double weight = std::exp(scores[t] - next);
        sum += weight;
        for (int d = 0; d < 256; ++d) out[d] += weight * decode(values[slots[t]], d);
      }
      maximum = next;
    }
    for (auto& value : out) value /= sum;
  }
  return out;
}

void causal_slot_model() {
  Pool keys{}, values{}, oracle_keys{}, oracle_values{};
  std::vector<int> slots;
  Head q{};
  for (int d = 0; d < 256; ++d) q[d] = static_cast<float>(d % 19 - 9) / 16.F;
  partial_rope(q, 34);
  CHECK(uq::rotate_head_in_place(q.data(), 256) == uq::Status::kOk);
  for (int t = 0; t < 33; ++t) {
    const int slot = (t * 7 + 3) % 43 + 1;
    slots.push_back(slot);
    Head k{}, v{};
    for (int d = 0; d < 256; ++d) {
      k[d] = static_cast<float>((t * 3 + d * 5) % 31 - 15) / 16.F;
      v[d] = static_cast<float>((t * 11 + d * 3) % 29 - 14) / 8.F;
    }
    partial_rope(k, t + 1);
    const auto rotated = matrix_hadamard(k);
    CHECK(uq::rotate_head_in_place(k.data(), 256) == uq::Status::kOk);
    Head independent_k{};
    for (int d = 0; d < 256; ++d) independent_k[d] = static_cast<float>(rotated[d]);
    oracle_keys[slot] = independent_encode_head(independent_k);
    oracle_values[slot] = independent_encode_head(v);
    CHECK(uq::encode_head(k.data(), 256, keys[slot].data()) == uq::Status::kOk);
    CHECK(uq::encode_head(v.data(), 256, values[slot].data()) == uq::Status::kOk);
    CHECK(keys[slot] == oracle_keys[slot] && values[slot] == oracle_values[slot]);
  }
  for (int visible : {1, 2, 15, 16, 17, 31, 32, 33}) {
    const Wide expected = attention(q, oracle_keys, oracle_values, slots, visible, true);
    const Wide actual = attention(q, keys, values, slots, visible, false);
    for (int d = 0; d < 256; ++d)
      attention_error = std::max(attention_error, std::fabs(actual[d] - expected[d]));
  }
  CHECK(attention_error <= 1e-11);
  const auto prefix_before = values[slots[0]];
  const auto causal_before = attention(q, keys, values, slots, 17, false);
  Head replacement{};
  replacement.fill(6.F);
  // Slot 32 is outside this request's visible prefix. Model cancellation and
  // subsequent slot reuse; there is no claim that a scheduler was executed.
  CHECK(uq::encode_head(replacement.data(), 256, values[slots[32]].data()) == uq::Status::kOk);
  oracle_values[slots[32]] = independent_encode_head(replacement);
  CHECK(values[slots[0]] == prefix_before);
  CHECK(attention(q, keys, values, slots, 17, false) == causal_before);
  const Wide reused = attention(q, keys, values, slots, 33, false);
  const Wide expected = attention(q, oracle_keys, oracle_values, slots, 33, true);
  for (int d = 0; d < 256; ++d) CHECK(std::fabs(reused[d] - expected[d]) <= 1e-11);
  std::puts("causal physical-slot CPU model, tile tails, continuation and cancelled-slot reuse: PASS");
}

}  // namespace

int main() {
  test_golden_layout_and_oracle();
  test_ties_and_limits();
  test_rejections();
  test_rotation_after_rope();
  test_head_size_and_paged_reuse();
  head256_bytes_and_rejections();
  fixed_vllm_differences();
  head256_layout_and_rotation();
  causal_slot_model();
  std::printf("head256 maxima: rotation=%.17g dot=%.17g softmax=%.17g\n",
              rotation_error, dot_error, attention_error);
  if (failures != 0) {
    std::fprintf(stderr, "%d codec test(s) failed\\n", failures);
    return 1;
  }
  std::puts("codec tests passed");
  return 0;
}
