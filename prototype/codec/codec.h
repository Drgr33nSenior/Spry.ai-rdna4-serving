// Experimental UltraQuant-derived MXFP4 codec. This header defines only the
// portable representation and CPU reference operations; it does not dispatch
// a GPU kernel or establish runtime support for any accelerator.
#ifndef UQ_CODEC_CODEC_H_
#define UQ_CODEC_CODEC_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define UQ_HOST_DEVICE __host__ __device__
#else
#define UQ_HOST_DEVICE
#endif

namespace uq {

constexpr std::size_t kGroupSize = 32;
constexpr std::size_t kPackedCodeBytes = 16;
constexpr std::size_t kEncodedGroupBytes = 17;
constexpr std::uint8_t kInvalidScale = 255;
constexpr double kScaleConstant = 0.156;

enum class Status : std::uint8_t {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedDimension,
  kNonFiniteInput,
  kOutOfRange,
  kMalformedEncoding,
};

UQ_HOST_DEVICE inline bool supported_head_dim(std::size_t dimension) {
  return dimension == 32 || dimension == 64 || dimension == 128 ||
         dimension == 256;
}

inline bool encoded_head_bytes(std::size_t dimension, std::size_t* bytes) {
  if (bytes == nullptr || !supported_head_dim(dimension)) {
    return false;
  }
  *bytes = (dimension / kGroupSize) * kEncodedGroupBytes;
  return true;
}

// E2M1 finite values indexed by the three low bits of an FP4 nibble. Bit 3 is
// the sign. Nibble 8 is negative zero in the hardware format and is rejected
// by this canonical software format.
UQ_HOST_DEVICE inline float fp4_magnitude(std::uint8_t magnitude_code) {
  constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                    2.0F, 3.0F, 4.0F, 6.0F};
  return kMagnitudes[magnitude_code & 0x7U];
}

UQ_HOST_DEVICE inline float decode_fp4(std::uint8_t code) {
  const float magnitude = fp4_magnitude(code);
  return (code & 0x8U) == 0 ? magnitude : -magnitude;
}

UQ_HOST_DEVICE inline std::int16_t scale_exponent(std::uint8_t scale_byte) {
  return static_cast<std::int16_t>(scale_byte) - 127;
}

// This is usable in a HIP device kernel. It deliberately has float range;
// decode_scalar_reference uses double for numerical reference tests.
UQ_HOST_DEVICE inline float decode_fp4_scaled(std::uint8_t code,
                                               std::uint8_t scale_byte) {
  // ldexp is supported by C++ and HIP device compilation. Callers validate
  // scale_byte and canonical codes when they consume untrusted data.
  return ::ldexpf(decode_fp4(code), scale_exponent(scale_byte));
}

// Unchecked, GPU-shareable scalar dequantization. It accepts the hardware
// negative-zero code and byte 255 because a kernel consumes trusted cache
// memory. CPU-facing decode functions below reject those non-canonical forms.
// Double preserves every finite FP4/UE8M0 reconstruction through 6 * 2^127.
UQ_HOST_DEVICE inline double decode_scalar_unchecked(
    const std::uint8_t* encoded, std::size_t channel) {
  const std::uint8_t packed = encoded[channel / 2U];
  const std::uint8_t code = (channel & 1U) == 0
                                ? static_cast<std::uint8_t>(packed & 0x0FU)
                                : static_cast<std::uint8_t>((packed >> 4U) &
                                                            0x0FU);
  return ::ldexp(static_cast<double>(decode_fp4(code)),
                 scale_exponent(encoded[16]));
}

UQ_HOST_DEVICE inline void decode_group_unchecked(const std::uint8_t* encoded,
                                                  float* output) {
  for (std::size_t i = 0; i < kGroupSize; ++i) {
    output[i] = static_cast<float>(decode_scalar_unchecked(encoded, i));
  }
}

UQ_HOST_DEVICE inline bool is_canonical_code(std::uint8_t code) {
  return code != 0x8U;
}

UQ_HOST_DEVICE inline Status validate_block(const std::uint8_t* encoded) {
  if (encoded == nullptr) {
    return Status::kInvalidArgument;
  }
  if (encoded[16] == kInvalidScale) {
    return Status::kMalformedEncoding;
  }
  for (std::size_t byte_index = 0; byte_index < kPackedCodeBytes;
       ++byte_index) {
    const std::uint8_t packed = encoded[byte_index];
    if (!is_canonical_code(packed & 0x0FU) ||
        !is_canonical_code((packed >> 4U) & 0x0FU)) {
      return Status::kMalformedEncoding;
    }
  }
  return Status::kOk;
}

inline double decode_scalar_reference(const std::uint8_t* encoded,
                                      std::size_t channel, Status* status) {
  if (status == nullptr) {
    return 0.0;
  }
  *status = validate_block(encoded);
  if (*status != Status::kOk || channel >= kGroupSize) {
    if (*status == Status::kOk) {
      *status = Status::kInvalidArgument;
    }
    return 0.0;
  }
  return decode_scalar_unchecked(encoded, channel);
}

inline Status decode_scalar(const std::uint8_t* encoded, std::size_t channel,
                            float* value) {
  if (value == nullptr) {
    return Status::kInvalidArgument;
  }
  Status status = validate_block(encoded);
  if (status != Status::kOk) {
    return status;
  }
  if (channel >= kGroupSize) {
    return Status::kInvalidArgument;
  }
  const std::uint8_t packed = encoded[channel / 2U];
  const std::uint8_t code = (channel & 1U) == 0
                                ? static_cast<std::uint8_t>(packed & 0x0FU)
                                : static_cast<std::uint8_t>((packed >> 4U) &
                                                            0x0FU);
  *value = decode_fp4_scaled(code, encoded[16]);
  return Status::kOk;
}

inline Status decode_block(const std::uint8_t* encoded, float* output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  const Status status = validate_block(encoded);
  if (status != Status::kOk) {
    return status;
  }
  for (std::size_t i = 0; i < kGroupSize; ++i) {
    const std::uint8_t packed = encoded[i / 2U];
    const std::uint8_t code = (i & 1U) == 0
                                  ? static_cast<std::uint8_t>(packed & 0x0FU)
                                  : static_cast<std::uint8_t>((packed >> 4U) &
                                                              0x0FU);
    output[i] = decode_fp4_scaled(code, encoded[16]);
  }
  return Status::kOk;
}

inline Status decode_block_reference(const std::uint8_t* encoded,
                                     double* output) {
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  const Status status = validate_block(encoded);
  if (status != Status::kOk) {
    return status;
  }
  for (std::size_t i = 0; i < kGroupSize; ++i) {
    Status scalar_status = Status::kOk;
    output[i] = decode_scalar_reference(encoded, i, &scalar_status);
    if (scalar_status != Status::kOk) {
      return scalar_status;
    }
  }
  return Status::kOk;
}

UQ_HOST_DEVICE inline std::int32_t round_nearest_even(double value) {
  const double lower = ::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) {
    return static_cast<std::int32_t>(lower);
  }
  if (fraction > 0.5) {
    return static_cast<std::int32_t>(lower + 1.0);
  }
  const std::int32_t lower_integer = static_cast<std::int32_t>(lower);
  return (lower_integer & 1) == 0 ? lower_integer : lower_integer + 1;
}

UQ_HOST_DEVICE inline std::uint8_t nearest_magnitude_code(double magnitude) {
  constexpr double kMagnitudes[8] = {0.0, 0.5, 1.0, 1.5,
                                     2.0, 3.0, 4.0, 6.0};
  std::uint8_t best = 0;
  double best_distance = ::fabs(magnitude - kMagnitudes[0]);
  for (std::uint8_t code = 1; code < 8; ++code) {
    const double distance = ::fabs(magnitude - kMagnitudes[code]);
    if (distance < best_distance ||
        (distance == best_distance && (code & 1U) == 0U)) {
      best = code;
      best_distance = distance;
    }
  }
  return best;
}

// Encodes exactly one contiguous 32-channel group. On a non-finite input it
// returns kNonFiniteInput before modifying output. Zero is encoded as sixteen
// zero bytes and UE8M0 byte 127 (2^0), the sole canonical zero encoding.
UQ_HOST_DEVICE inline Status encode_group(const float* input,
                                          std::uint8_t* output) {
  if (input == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  double absmax = 0.0;
  for (std::size_t i = 0; i < kGroupSize; ++i) {
    if (!::isfinite(input[i])) {
      return Status::kNonFiniteInput;
    }
    const double magnitude = ::fabs(static_cast<double>(input[i]));
    if (magnitude > absmax) {
      absmax = magnitude;
    }
  }

  if (absmax == 0.0) {
    for (std::size_t i = 0; i < kPackedCodeBytes; ++i) {
      output[i] = 0;
    }
    output[16] = 127;
    return Status::kOk;
  }

  std::int32_t exponent = round_nearest_even(::log2(kScaleConstant * absmax));
  if (exponent < -127) {
    exponent = -127;
  } else if (exponent > 127) {
    exponent = 127;
  }
  const double scale = ::ldexp(1.0, exponent);
  for (std::size_t pair = 0; pair < kPackedCodeBytes; ++pair) {
    std::uint8_t packed = 0;
    for (std::size_t lane = 0; lane < 2; ++lane) {
      const float value = input[pair * 2U + lane];
      const std::uint8_t magnitude_code =
          nearest_magnitude_code(::fabs(static_cast<double>(value)) / scale);
      const std::uint8_t code = magnitude_code == 0
                                    ? 0
                                    : static_cast<std::uint8_t>(
                                          magnitude_code | (::signbit(value)
                                                                ? 0x8U
                                                                : 0U));
      packed |= static_cast<std::uint8_t>(code << (lane * 4U));
    }
    output[pair] = packed;
  }
  output[16] = static_cast<std::uint8_t>(exponent + 127);
  return Status::kOk;
}

inline Status encode_head(const float* input, std::size_t dimension,
                          std::uint8_t* output) {
  if (input == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!supported_head_dim(dimension)) {
    return Status::kUnsupportedDimension;
  }
  // Validate first so rejected heads never leave a mixed representation.
  for (std::size_t i = 0; i < dimension; ++i) {
    if (!std::isfinite(input[i])) {
      return Status::kNonFiniteInput;
    }
  }
  for (std::size_t group = 0; group < dimension / kGroupSize; ++group) {
    const Status status = encode_group(input + group * kGroupSize,
                                       output + group * kEncodedGroupBytes);
    if (status != Status::kOk) {
      return status;
    }
  }
  return Status::kOk;
}

inline Status decode_head(const std::uint8_t* encoded, std::size_t dimension,
                          float* output) {
  if (encoded == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!supported_head_dim(dimension)) {
    return Status::kUnsupportedDimension;
  }
  for (std::size_t group = 0; group < dimension / kGroupSize; ++group) {
    const Status status = decode_block(encoded + group * kEncodedGroupBytes,
                                       output + group * kGroupSize);
    if (status != Status::kOk) {
      return status;
    }
  }
  return Status::kOk;
}

// Apply the normalized Walsh-Hadamard transform to one post-RoPE full head.
// The caller applies it to both Q and K; V remains unrotated. The operation is
// its own inverse to normal floating-point rounding. This CPU reference only
// accepts the dimensions selected for the experiment.
UQ_HOST_DEVICE inline Status rotate_head_in_place(float* values,
                                                  std::size_t dimension) {
  if (values == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!supported_head_dim(dimension)) {
    return Status::kUnsupportedDimension;
  }
  for (std::size_t i = 0; i < dimension; ++i) {
    if (!::isfinite(values[i])) {
      return Status::kNonFiniteInput;
    }
  }
  // A direct float butterfly can overflow before normalization even when its
  // final result is finite. The fixed maximum experiment dimension permits a
  // small double scratch buffer and makes rejected over-range transforms leave
  // the caller's head unchanged.
  double scratch[256];
  for (std::size_t i = 0; i < dimension; ++i) {
    scratch[i] = values[i];
  }
  for (std::size_t width = 1; width < dimension; width *= 2U) {
    for (std::size_t start = 0; start < dimension; start += width * 2U) {
      for (std::size_t offset = 0; offset < width; ++offset) {
        const double a = scratch[start + offset];
        const double b = scratch[start + width + offset];
        scratch[start + offset] = a + b;
        scratch[start + width + offset] = a - b;
      }
    }
  }
  const double normalization = 1.0 / ::sqrt(static_cast<double>(dimension));
  for (std::size_t i = 0; i < dimension; ++i) {
    scratch[i] *= normalization;
    if (!::isfinite(scratch[i]) ||
        ::fabs(scratch[i]) > std::numeric_limits<float>::max()) {
      return Status::kOutOfRange;
    }
  }
  for (std::size_t i = 0; i < dimension; ++i) {
    values[i] = static_cast<float>(scratch[i]);
  }
  return Status::kOk;
}

}  // namespace uq

#undef UQ_HOST_DEVICE

#endif  // UQ_CODEC_CODEC_H_
