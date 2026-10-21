// CPU simulation of the exact extracted HIP validator bodies, not a HIP run.
// Build with UQ_VALIDATION_EXTRACTOR first to produce the two include files.
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

static std::string source(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

#ifdef UQ_VALIDATION_EXTRACTOR
static std::string body(const std::string& text) {
    const std::string signature = "__global__ void validate_read(";
    const size_t begin = text.find(signature);
    if (begin == std::string::npos || text.find(signature, begin + 1) != std::string::npos)
        throw std::runtime_error("expected exactly one v2 validate_read kernel");
    size_t end = text.find('{', begin);
    if (end == std::string::npos) throw std::runtime_error("missing validator body");
    unsigned nesting = 1;
    while (nesting && ++end < text.size()) {
        if (text[end] == '{') ++nesting;
        if (text[end] == '}') --nesting;
    }
    if (nesting) throw std::runtime_error("unclosed validator body");
    return text.substr(begin, end - begin + 1) + '\n';
}

int main(int argc, char** argv) {
    if (argc != 4) throw std::runtime_error("usage: extractor BASELINE CANDIDATE OUTPUT_PREFIX");
    for (int i = 1; i <= 2; ++i) {
        const std::string output = std::string(argv[3]) + (i == 1 ? ".baseline.inc" : ".candidate.inc");
        std::ofstream out(output);
        out << body(source(argv[i]));
        if (!out) throw std::runtime_error("cannot write " + output);
    }
}
#else
#include "../runtime/bf16_math.h"
#include "../runtime/uq.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

struct FakeDim { unsigned x = 0; };
static FakeDim blockIdx, threadIdx, blockDim;
static int atomicExch(int* p, int value) { int old = *p; *p = value; return old; }
#define __global__
namespace baseline {
#include UQ_VALIDATION_BASELINE_FILE
}
namespace candidate {
#include UQ_VALIDATION_CANDIDATE_FILE
}
#undef __global__

// Exact logical ends border inaccessible pages. Sparse mappings let the maximum
// slot be addressed without allocating/touching gigabytes of physical memory.
class Guarded {
    void* mapping_;
    size_t mapped_, page_, extent_;
public:
    uint8_t* data;
    const size_t bytes;
    explicit Guarded(size_t n) : bytes(n) {
        assert(n > 0);
        page_ = size_t(sysconf(_SC_PAGESIZE));
        extent_ = (n + page_ - 1) / page_ * page_;
        mapped_ = extent_ + 2 * page_;
        mapping_ = mmap(nullptr, mapped_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapping_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        data = static_cast<uint8_t*>(mapping_) + page_ + extent_ - n;
        writable(true);
    }
    ~Guarded() { munmap(mapping_, mapped_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    void writable(bool yes) {
        if (mprotect(static_cast<uint8_t*>(mapping_) + page_, extent_,
                     yes ? PROT_READ | PROT_WRITE : PROT_READ) != 0)
            throw std::runtime_error("mprotect failed");
    }
    template<class T> T* as() { return reinterpret_cast<T*>(data); }
};

using Kernel = decltype(&baseline::validate_read);
static size_t cases = 0;
static int simulate(Kernel kernel, unsigned threads, uq_bf16_cache_v2 c,
                    const uint16_t* q, const int64_t* indices, int count,
                    const int32_t* starts, const int32_t* lengths, int rows) {
    Guarded status(3 * sizeof(int));
    int* words = status.as<int>();
    words[0] = 0x12345678; words[1] = 0; words[2] = 0x23456789;
    blockDim.x = threads;
    for (int row = 0; row < rows; ++row) {
        blockIdx.x = unsigned(row);
        for (unsigned thread = 0; thread < threads; ++thread) {
            threadIdx.x = thread;
            kernel(c, q, indices, count, starts, lengths, words + 1);
        }
    }
    assert(words[0] == 0x12345678 && words[2] == 0x23456789);
    assert(words[1] == 0 || words[1] == 1);
    return words[1];
}

struct Fixture {
    Guarded k, v, q, indices, starts, lengths;
    uq_bf16_cache_v2 c{};
    int rows, count;
    Fixture(bool protect, int slots = 8, int queries = 1, int index_count = 1)
        : k(size_t(slots) * 2 * (protect ? 512 : 136)), v(k.bytes),
          q(size_t(queries) * 3072 * sizeof(uint16_t)),
          indices(size_t(index_count) * sizeof(int64_t)),
          starts(size_t(queries) * sizeof(int32_t)), lengths(starts.bytes),
          rows(queries), count(index_count) {
        c.k = k.data; c.v = v.data; c.slots = uint32_t(slots);
        c.bytes_per_buffer = k.bytes; c.protected_layer = protect;
        for (int i = 0; i < count; ++i) indices.as<int64_t>()[i] = slots - 1;
        for (int row = 0; row < rows; ++row) {
            starts.as<int32_t>()[row] = 0;
            lengths.as<int32_t>()[row] = count;
        }
        valid_slot(slots - 1);
    }
    void valid_slot(int slot) {
        for (Guarded* buffer : {&k, &v}) {
            if (c.protected_layer) {
                for (int i = 0; i < 512; ++i)
                    buffer->as<uint16_t>()[size_t(slot) * 512 + i] = (i & 1) ? 0xc480 : 0x4480;
            } else {
                for (int h = 0; h < 2; ++h) for (int g = 0; g < 8; ++g) {
                    uint8_t* group = buffer->data + (size_t(slot) * 2 + h) * 136 + g * 17;
                    for (int b = 0; b < 16; ++b) group[b] = uint8_t((b % 8) | ((15 - b % 8) << 4));
                    // Nibble 8 is excluded from canonical input.
                    for (int b = 0; b < 16; ++b) if ((group[b] >> 4) == 8) group[b] &= 15;
                    group[16] = buffer == &k ? 139 : 135;
                }
            }
        }
    }
    void expect(int want, bool null_inputs = false) {
        std::array<Guarded*, 6> buffers = {&k, &v, &q, &indices, &starts, &lengths};
        for (Guarded* b : buffers) b->writable(false);
        const uint16_t* query = null_inputs ? nullptr : q.as<uint16_t>();
        const int64_t* index = null_inputs ? nullptr : indices.as<int64_t>();
        const int before = simulate(baseline::validate_read, 32, c, query, index, count,
                                    starts.as<int32_t>(), lengths.as<int32_t>(), rows);
        const int after = simulate(candidate::validate_read, 256, c, query, index, count,
                                   starts.as<int32_t>(), lengths.as<int32_t>(), rows);
        assert(before == want && after == before);
        for (Guarded* b : buffers) b->writable(true);
        ++cases;
    }
};

static void malformed_spans_and_indices(bool protect) {
    Fixture f(protect, 8, 3, 5);
    f.valid_slot(1); f.valid_slot(3);
    const int64_t mixed[] = {1, 7, 1, 3, 7};
    std::memcpy(f.indices.data, mixed, sizeof mixed);
    f.starts.as<int32_t>()[0] = 0; f.lengths.as<int32_t>()[0] = 5;
    f.starts.as<int32_t>()[1] = 1; f.lengths.as<int32_t>()[1] = 3;
    f.starts.as<int32_t>()[2] = 4; f.lengths.as<int32_t>()[2] = 1;
    f.expect(0); // Duplicate indices are valid reads, including overlapping rows.
    // Unreferenced slots may be missing; validation covers exactly each span.
    std::memset(f.k.data, 255, f.c.protected_layer ? 1024 : 272);
    std::memset(f.v.data, 255, f.c.protected_layer ? 1024 : 272);
    f.expect(0);
    for (int position = 0; position < 5; ++position) {
        for (int64_t bad : {INT64_MIN, int64_t(-1), int64_t(0), int64_t(8), INT64_MAX}) {
            f.indices.as<int64_t>()[position] = bad; f.expect(1);
        }
        f.indices.as<int64_t>()[position] = mixed[position];
    }
    for (int row = 0; row < 3; ++row) {
        f.starts.as<int32_t>()[row] = 1; f.lengths.as<int32_t>()[row] = 3;
    }
    f.indices.as<int64_t>()[0] = 0; f.indices.as<int64_t>()[4] = INT64_MAX;
    f.expect(0); // Malformed indices outside all selected spans are ignored.
    std::memcpy(f.indices.data, mixed, sizeof mixed);
    f.starts.as<int32_t>()[0] = 0; f.lengths.as<int32_t>()[0] = 5;
    f.starts.as<int32_t>()[1] = 1; f.lengths.as<int32_t>()[1] = 3;
    f.starts.as<int32_t>()[2] = 4; f.lengths.as<int32_t>()[2] = 1;
    for (const auto& span : {std::array<int32_t, 2>{-1, 1}, {INT32_MIN, 1},
         {6, 1}, {INT32_MAX, 1}, {0, 0}, {0, -1}, {0, INT32_MIN},
         {0, 131073}, {0, INT32_MAX}, {5, 1}, {4, 2}, {INT32_MAX, INT32_MAX}}) {
        for (int row = 0; row < 3; ++row) {
            int32_t old_start = f.starts.as<int32_t>()[row], old_len = f.lengths.as<int32_t>()[row];
            f.starts.as<int32_t>()[row] = span[0]; f.lengths.as<int32_t>()[row] = span[1];
            f.expect(1);
            f.starts.as<int32_t>()[row] = old_start; f.lengths.as<int32_t>()[row] = old_len;
        }
    }
    for (int row = 0; row < 3; ++row) f.starts.as<int32_t>()[row] = -1;
    f.expect(1, true); // Every malformed row must return before touching Q/indices.
}

static void query_coverage() {
    Fixture f(false);
    for (int j = 0; j < 3072; ++j) {
        f.q.as<uint16_t>()[j] = 0x4481; f.expect(1); f.q.as<uint16_t>()[j] = 0;
    }
    for (uint16_t bits : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001),
                          uint16_t(0x4480), uint16_t(0xc480), uint16_t(0x4481),
                          uint16_t(0xc481), uint16_t(0x7f80), uint16_t(0xff80), uint16_t(0x7fc1)}) {
        for (int j : {0, 255, 256, 3071}) {
            f.q.as<uint16_t>()[j] = bits; f.expect(uq_bf16::input_ok(bits) ? 0 : 1);
            f.q.as<uint16_t>()[j] = 0;
        }
    }
}

static void protected_coverage() {
    Fixture f(true);
    f.expect(0);
    for (Guarded* buffer : {&f.k, &f.v}) for (int channel = 0; channel < 512; ++channel) {
        uint16_t* p = buffer->as<uint16_t>() + size_t(f.c.slots - 1) * 512 + channel;
        const uint16_t old = *p;
        *p = 0xc481; f.expect(1); *p = old;
    }
    for (uint16_t bad : {uint16_t(0x7f80), uint16_t(0xff80), uint16_t(0x7fc1)}) {
        for (Guarded* buffer : {&f.k, &f.v}) {
            uint16_t* p = buffer->as<uint16_t>() + size_t(f.c.slots) * 512 - 1;
            const uint16_t old = *p; *p = bad; f.expect(1); *p = old;
        }
    }
    std::memset(f.k.data, 255, f.k.bytes); std::memset(f.v.data, 255, f.v.bytes);
    f.expect(1);
}

static void packed_coverage() {
    Fixture f(false);
    f.expect(0);
    for (Guarded* buffer : {&f.k, &f.v}) {
        const int bound = buffer == &f.k ? 139 : 135;
        const size_t base = size_t(f.c.slots - 1) * 2 * 136;
        for (int head = 0; head < 2; ++head) for (int group = 0; group < 8; ++group) {
            uint8_t* p = buffer->data + base + head * 136 + group * 17;
            for (int byte = 0; byte < 16; ++byte) for (int half = 0; half < 2; ++half) {
                uint8_t old = p[byte];
                p[byte] = uint8_t((old & (half ? 15 : 240)) | (8 << (half * 4)));
                f.expect(1); p[byte] = old;
            }
            for (uint8_t bad : {uint8_t(bound + 1), uint8_t(255)}) {
                uint8_t old = p[16]; p[16] = bad; f.expect(1); p[16] = old;
            }
        }
        uint8_t* last_scale = buffer->data + buffer->bytes - 1;
        const uint8_t old = *last_scale;
        for (int scale = 0; scale < 256; ++scale) {
            *last_scale = uint8_t(scale); f.expect(scale <= bound ? 0 : 1);
        }
        *last_scale = old;
        uint8_t* last_byte = last_scale - 1;
        const uint8_t old_byte = *last_byte;
        for (int byte = 0; byte < 256; ++byte) {
            *last_byte = uint8_t(byte);
            f.expect(((byte & 15) == 8 || (byte >> 4) == 8) ? 1 : 0);
        }
        *last_byte = old_byte;
    }
    std::memset(f.k.data, 255, f.k.bytes); std::memset(f.v.data, 255, f.v.bytes);
    f.expect(1); // Entirely missing cache, not just a missing final group.
}

static void geometry_and_tails(const std::string& candidate_source) {
    std::string compact;
    for (unsigned char c : source(candidate_source)) if (!std::isspace(c)) compact += char(c);
    const std::string launch = "hipLaunchKernelGGL(validate_read,dim3(queries),dim3(256),0,stream,*c,q,indices,count,starts,lengths,status);";
    assert(compact.find(launch) != std::string::npos);
    const size_t launch_at = compact.find(launch);
    const size_t status_at = compact.find("intchecked=checked_status(status,stream);if(checked)returnchecked;", launch_at);
    const size_t read_at = compact.find("hipLaunchKernelGGL(read,dim3(queries,12),dim3(32)", launch_at);
    const size_t grouped_at = compact.find("hipLaunchKernelGGL(read_grouped,dim3(queries,2),dim3(192)", launch_at);
    assert(status_at != std::string::npos && read_at != std::string::npos && grouped_at != std::string::npos);
    assert(status_at < read_at && status_at < grouped_at);
    assert(compact.find("if(c->protected_layer){hipLaunchKernelGGL(read,", status_at) < read_at);
    assert(compact.find("}else{hipLaunchKernelGGL(read_grouped,", read_at) < grouped_at);
    static_assert(131072 * 512 < INT_MAX && 131072 * 256 < INT_MAX);
    static_assert(4095 * 3072 + 3071 < INT_MAX);
    for (bool protect : {false, true}) {
        for (int len : {1, 2, 3, 15, 16, 17, 31, 32, 33, 255, 256, 257}) {
            Fixture f(protect, 8, 1, len); f.expect(0);
            f.indices.as<int64_t>()[len - 1] = 0; f.expect(1);
        }
        Fixture last_row(protect, 8, 4096);
        last_row.expect(0);
        last_row.q.as<uint16_t>()[size_t(4096) * 3072 - 1] = 0x4481;
        last_row.expect(1);
        Fixture longest(protect, 1048576, 1, 1048576);
        longest.starts.as<int32_t>()[0] = 1048576 - 131072;
        longest.lengths.as<int32_t>()[0] = 131072;
        longest.expect(0); // Last legal slot, full context, last legal index.
        if (protect) {
            longest.v.as<uint16_t>()[size_t(1048576) * 512 - 1] = 0x7f80;
        } else {
            longest.v.data[longest.v.bytes - 1] = 255;
        }
        longest.expect(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("usage: validation_test CANDIDATE_SOURCE");
    malformed_spans_and_indices(false); malformed_spans_and_indices(true);
    query_coverage(); protected_coverage(); packed_coverage(); geometry_and_tails(argv[1]);
    std::cout << "Exact baseline/candidate validator bodies: " << cases
              << " CPU simulation cases PASS; read-only inputs, status canaries, guarded ends,"
                 " full scales/nibbles/channels/Q positions and supported-limit geometry."
                 " Native HIP execution and GPU performance were not tested.\n";
}
#endif
