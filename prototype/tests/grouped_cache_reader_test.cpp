// Single-thread cooperative execution of extracted source phases. This checks
// arithmetic and addressing, not GPU synchronization, compilation or timing.
#include "../runtime/bf16_math.h"
#include "../runtime/uq.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime_source {
float load(const void*, size_t, int, bool);
}
struct Dim { int x=0, y=0; };
struct Lane { float acc[8]={}, maximum=-INFINITY; };
struct Common {
    uq_bf16_cache_v2 c{};
    const uint16_t* q=nullptr;
    const int64_t* indices=nullptr;
    const int32_t* starts=nullptr;
    const int32_t* lengths=nullptr;
    float scale=0.0625f;
    uint16_t* output=nullptr;
    Dim blockIdx, threadIdx;
    size_t k_loads=0, v_loads=0;
    float load(const void* buffer, size_t record, int channel, bool protect) {
        (buffer==c.k ? k_loads : v_loads)++;
        return runtime_source::load(buffer,record,channel,protect);
    }
};
#include UQ_GROUPED_PHASES_FILE

static size_t comparisons=0, tokens=0, rows=0, grouped_loads=0, original_loads=0;
static void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}
static void equal(float a, float b, const char* stage) {
    uint32_t left, right;
    std::memcpy(&left,&a,sizeof left); std::memcpy(&right,&b,sizeof right);
    if (left!=right) throw std::runtime_error(std::string("FP32 mismatch: ")+stage);
    ++comparisons;
}

struct Fixture {
    static constexpr int slots=37;
    std::vector<uint8_t> k, v;
    std::vector<uint16_t> q;
    std::vector<int64_t> indices;
    std::vector<int32_t> starts, lengths;
    explicit Fixture(bool boundaries)
        : k(slots*272), v(slots*272),
          lengths{1,31,32,33,255,256,257,1024} {
        starts={13,7,11,9,5,3,1,17};
        indices.resize(1041);
        for (size_t i=0;i<indices.size();++i)
            indices[i]=1+int((i*17+i/5)%36); // Noncontiguous and repeated slots.
        q.resize(lengths.size()*3072);
        for (size_t row=0;row<lengths.size();++row)
            for (int head=0;head<12;++head) for (int j=0;j<256;++j) {
                const float value=float(int((j*13+head*31+row*7)%101)-50)/64.0f;
                q[(row*12+head)*256+j]=uq_bf16::to_bits(value);
            }
        constexpr uint8_t codes[]={0,1,2,3,4,5,6,7,9,10,11,12,13,14,15};
        constexpr uint8_t scales[]={0,1,2,63,125,127,134,135};
        for (int slot=0;slot<slots;++slot) for (int head=0;head<2;++head)
            for (int group=0;group<8;++group) for (int which=0;which<2;++which) {
                uint8_t* p=(which ? v : k).data()+(slot*2+head)*136+group*17;
                for (int byte=0;byte<16;++byte) {
                    int seed=slot*3+head*7+group*11+byte*2+which*5;
                    p[byte]=uint8_t(codes[seed%15] | (codes[(seed+1)%15]<<4));
                }
                p[16]=boundaries ? scales[(slot+head+group+which)%8]
                                 : uint8_t(122+(slot+head+group+which)%7);
                if (boundaries && group==7) p[16]=which ? 135 : 139;
                require(uq_bf16::record_ok((which ? v : k).data()+(slot*2+head)*136,
                                          which ? 135 : 139) || group<7,"canonical fixture");
            }
    }
    Common common(int row, int head, uint16_t* output) {
        Common c;
        c.c.k=k.data(); c.c.v=v.data(); c.c.slots=slots;
        c.c.bytes_per_buffer=k.size(); c.c.protected_layer=0;
        c.q=q.data(); c.indices=indices.data(); c.starts=starts.data(); c.lengths=lengths.data();
        c.output=output; c.blockIdx={row,head};
        return c;
    }
};

template<class Kernel>
static void phase(Kernel& kernel, void (Kernel::*fn)(int,int), int count, bool reverse, int t=0) {
    for (int step=0;step<count;++step) (kernel.*fn)(reverse ? count-1-step : step,t);
}

static void exercise(bool boundaries, bool reverse) {
    Fixture fixture(boundaries);
    const auto saved_k=fixture.k, saved_v=fixture.v;
    const auto saved_q=fixture.q;
    const auto saved_indices=fixture.indices;
    const auto saved_starts=fixture.starts, saved_lengths=fixture.lengths;
    constexpr size_t guard=19;
    constexpr uint16_t canary=0x7fad;
    std::vector<uint16_t> before(fixture.q.size()+2*guard,canary), after=before;
    for (size_t row=0;row<fixture.lengths.size();++row) for (int kvhead=0;kvhead<2;++kvhead) {
        Grouped grouped;
        static_cast<Common&>(grouped)=fixture.common(int(row),kvhead,after.data()+guard);
        std::array<Original,6> original;
        for (int head=0;head<6;++head)
            static_cast<Common&>(original[head])=fixture.common(int(row),kvhead*6+head,before.data()+guard);
        phase(grouped,&Grouped::query,192,reverse);
        phase(grouped,&Grouped::rotate,192,reverse);
        for (auto& head : original) {
            phase(head,&Original::query,32,reverse); phase(head,&Original::rotate,32,reverse);
        }
        for (int head=0;head<6;++head) for (int channel=0;channel<256;++channel)
            equal(grouped.rotated[head][channel],original[head].rotated[channel],"rotated query");
        for (int t=0;t<fixture.lengths[row];++t) {
            phase(grouped,&Grouped::decode,192,reverse,t);
            phase(grouped,&Grouped::product,192,reverse,t);
            for (auto& head : original) phase(head,&Original::product,32,reverse,t);
            for (int head=0;head<6;++head) for (int channel=0;channel<256;++channel)
                equal(grouped.products[head][channel],original[head].products[channel],"products");
            phase(grouped,&Grouped::score,192,reverse,t);
            for (auto& head : original) phase(head,&Original::score,32,reverse,t);
            for (int head=0;head<6;++head) {
                equal(grouped.states[head*32].maximum,original[head].states[0].maximum,"maximum");
                equal(grouped.alpha[head],original[head].alpha,"alpha");
                equal(grouped.weight[head],original[head].weight,"weight");
                equal(grouped.denominator[head],original[head].denominator,"denominator");
            }
            phase(grouped,&Grouped::accumulate,192,reverse,t);
            for (auto& head : original) phase(head,&Original::accumulate,32,reverse,t);
            for (int head=0;head<6;++head) for (int lane=0;lane<32;++lane) for (int i=0;i<8;++i)
                equal(grouped.states[head*32+lane].acc[i],original[head].states[lane].acc[i],"accumulators");
            ++tokens;
        }
        phase(grouped,&Grouped::emit_output,192,reverse);
        size_t old_k=0,old_v=0;
        for (auto& head : original) {
            phase(head,&Original::emit_output,32,reverse);
            old_k+=head.k_loads; old_v+=head.v_loads;
        }
        const size_t expected=size_t(fixture.lengths[row])*256;
        require(grouped.k_loads==expected && grouped.v_loads==expected,"once-per-KV decode counts");
        require(old_k==6*expected && old_v==6*expected,"six-fold decode reduction");
        grouped_loads+=grouped.k_loads+grouped.v_loads; original_loads+=old_k+old_v;
        ++rows;
    }
    require(before==after,"BF16 outputs differ");
    for (size_t i=0;i<guard;++i)
        require(after[i]==canary && after[after.size()-1-i]==canary,"output canary changed");
    for (size_t i=guard;i<after.size()-guard;++i) require(after[i]!=canary,"output channel missing");
    require(fixture.k==saved_k && fixture.v==saved_v && fixture.q==saved_q &&
            fixture.indices==saved_indices && fixture.starts==saved_starts && fixture.lengths==saved_lengths,
            "input changed");
}

int main() {
    try {
        for (bool boundaries : {false,true}) for (bool reverse : {false,true}) exercise(boundaries,reverse);
        std::cout << "Extracted grouped/original reader phases PASS: " << rows << " row/KV-head cases, "
                  << tokens << " token steps, " << comparisons << " exact FP32 comparisons; "
                  << grouped_loads << " grouped vs " << original_loads << " original scalar decodes (6x). "
                     "Full BF16 outputs, input snapshots, output canaries, ascending/reversed lane schedules. "
                     "CPU only; GPU synchronization, native equivalence and performance unqualified.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
