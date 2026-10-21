// Native fixture, run once per library in separate processes. No model loading.
// Compile with the authenticated hipcc, -O2 -std=c++17 -ffp-contract=off
// -fno-fast-math --offload-arch=gfx1201 and -ldl. The parent owns deadlines,
// device admission, source/library hashes, process cleanup and old/new comparison.
#include <hip/hip_runtime.h>
#include "../runtime/uq.h"
#include "../runtime/bf16_math.h"
#include "../runtime/cache_phase_profile.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
constexpr int kTokens = 3966, kSlots = 8192, kQueryValues = 3072;
constexpr std::array<int32_t, 9> kLengths = {1, 31, 32, 33, 255, 256, 257, 1024, 3966};
constexpr uint64_t kIdentity[4] = {0x123456789abcdef0, 0x1166992288334477, 3, 4};
constexpr const char* kTimingV2 = "warm-synchronized-output-v2";
size_t device_bytes=0, peak_device_bytes=0;
unsigned free_errors=0;
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void hip_check(hipError_t error, const char* what) {
  require(error == hipSuccess, std::string(what) + ": " + hipGetErrorString(error));
}
struct Device {
  void* p = nullptr; size_t bytes;
  explicit Device(size_t size) : bytes(size) {
    require(size<=64u*1024u*1024u-device_bytes,"explicit device buffer cap");
    hip_check(hipMalloc(&p, size), "hipMalloc");device_bytes+=size;
    peak_device_bytes=std::max(peak_device_bytes,device_bytes);
  }
  void release() {
    if(p) {if(hipFree(p)!=hipSuccess)++free_errors;p=nullptr;device_bytes-=bytes;}
  }
  ~Device() { release(); }
  Device(const Device&) = delete; Device& operator=(const Device&) = delete;
  template<class T> T* as() const { return static_cast<T*>(p); }
  void put(const void* source, size_t size, size_t offset = 0) const {
    require(offset <= bytes && size <= bytes-offset, "device upload bounds");
    hip_check(hipMemcpy(static_cast<uint8_t*>(p)+offset, source, size, hipMemcpyHostToDevice), "upload");
  }
  std::vector<uint8_t> get() const {
    std::vector<uint8_t> out(bytes);
    hip_check(hipMemcpy(out.data(), p, bytes, hipMemcpyDeviceToHost), "download"); return out;
  }
};
struct Output {
  int fd = -1; uint64_t total = 0; unsigned records = 0;
  explicit Output(const char* path) {
    require(path[0] == '/', "output must be absolute");
    fd = open(path, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0600);
    require(fd >= 0, "exclusive output create"); write("UQBFV2G1", 8);
  }
  ~Output() { if (fd >= 0) (void)close(fd); }
  void write(const void* data, size_t bytes) {
    require(total + bytes <= 32u*1024u*1024u, "output cap");
    auto p = static_cast<const uint8_t*>(data);
    for (size_t done=0; done<bytes;) {
      ssize_t n=::write(fd,p+done,bytes-done);
      if (n<0 && errno==EINTR) continue;
      require(n>0,"write failed"); done+=size_t(n);
    }
    total += bytes;
  }
  void integer(uint64_t number, unsigned bytes) {
    uint8_t little[8]; for (unsigned i=0;i<bytes;++i) little[i]=uint8_t(number>>(8*i));
    write(little,bytes);
  }
  void record(const std::string& name, const void* data, size_t bytes) {
    require(name.size()<=128,"record name"); integer(name.size(),4); integer(bytes,8);
    write(name.data(),name.size()); write(data,bytes); ++records;
  }
  void finish() {
    require(fsync(fd)==0,"fsync failed"); const int old=fd; fd=-1;
    require(close(old)==0,"close failed");
  }
};

// The previous implementation's exact generic decode path is the device oracle.
__global__ void decoder_bits(uint32_t* result) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=8192) return;
  const unsigned parity=i&1u, pair=i>>1u;
  const uint8_t code=uint8_t(pair&15u), scale=uint8_t(pair>>4u);
  uint8_t record[136] = {};
  record[0]=uint8_t(code<<(4u*parity)); record[16]=scale;
  const float actual=uq_bf16::unpack(record,int(parity));
  const float expected=uq_bf16::from_bits(uq_bf16::to_bits(uq::decode_fp4_scaled(code,scale)));
  __builtin_memcpy(result+2*i,&actual,4); __builtin_memcpy(result+2*i+1,&expected,4);
}

struct Library {
  void* handle = nullptr;
  decltype(&uq_bf16_append_v2) append = nullptr;
  decltype(&uq_bf16_attention_v2) attention = nullptr;
  int (*phase_last)(uq_cache_phase_profile_v1*,uint64_t) = nullptr;
  explicit Library(const char* path,bool phase_profile=false) {
    require(path[0]=='/',"library must be absolute");
    handle=dlopen(path,RTLD_NOW|RTLD_LOCAL); require(handle!=nullptr,"library load failed");
    append=reinterpret_cast<decltype(append)>(dlsym(handle,"uq_bf16_append_v2"));
    attention=reinterpret_cast<decltype(attention)>(dlsym(handle,"uq_bf16_attention_v2"));
    require(append && attention,"missing v2 exports");
    if(phase_profile) {
      phase_last=reinterpret_cast<decltype(phase_last)>(dlsym(handle,"uq_cache_phase_profile_last_v1"));
      require(phase_last!=nullptr,"missing phase profile export");
    }
  }
  uq_cache_phase_profile_v1 phase() const {
    uq_cache_phase_profile_v1 value{};
    require(phase_last && phase_last(&value,sizeof(value))==0,"phase profile getter failed");
    require(value.schema==1 && value.sequence>0 && value.complete==1 && value.api_result==0,
            "invalid phase profile snapshot");
    require(value.total_ns>0 && value.host_checks_ns<=value.total_ns &&
            value.validation_status_ns<=value.total_ns-value.host_checks_ns &&
            value.reader_sync_ns==value.total_ns-value.host_checks_ns-value.validation_status_ns,
            "invalid phase profile durations");
    return value;
  }
  // Keep one library loaded to process exit: HIP registration is never reused.
};

uint16_t sample(size_t i, unsigned salt) {
  constexpr uint16_t small[] = {0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080};
  if((i+salt)%29<8) return small[(i+salt)%8];
  return uq_bf16::to_bits(float(int((i*7+salt)%31)-15)/64.0f);
}

struct Fixture {
  Library& lib; Output& out; bool protect;
  Device k, v, new_k, new_v, locations, query, indices, starts, lengths, output, status;
  uq_bf16_cache_v2 cache{};
  std::vector<uint16_t> host_k, host_v, host_q;
  std::vector<int64_t> host_locations, host_indices;
  std::vector<int32_t> host_starts, host_lengths;
  unsigned refused=0, accepted=0;
  std::vector<std::array<double,3>> timing;
  std::vector<uq_cache_phase_profile_v1> phase_timing;
  Fixture(Library& l, Output& o, bool p) : lib(l),out(o),protect(p),
    k(size_t(kSlots)*2*(p?512:136)),v(k.bytes),new_k(size_t(kTokens)*1024),new_v(new_k.bytes),
    locations(size_t(kTokens)*8),query(kLengths.size()*6144),indices(size_t(kTokens)*8),
    starts(kLengths.size()*4),lengths(starts.bytes),output(query.bytes+128),status(12),
    host_k(size_t(kTokens)*512),host_v(host_k.size()),host_q(kLengths.size()*kQueryValues),
    host_locations(kTokens),host_indices(kTokens),host_starts(kLengths.size()),
    host_lengths(kLengths.begin(),kLengths.end()) {
    cache.k=k.p;cache.v=v.p;cache.bytes_per_buffer=k.bytes;cache.slots=kSlots;cache.protected_layer=p;
    std::copy(std::begin(kIdentity),std::end(kIdentity),cache.identity);
    for(size_t i=0;i<host_k.size();++i) {host_k[i]=sample(i,3);host_v[i]=sample(i,11);}
    for(size_t i=0;i<host_q.size();++i) host_q[i]=sample(i,17);
    for(int i=0;i<kTokens;++i) host_locations[i]=1+(i*137)%8191;
    for(int i=0;i<kTokens;++i) host_indices[i]=host_locations[kTokens-1-i];
    for(size_t i=0;i<kLengths.size();++i) host_starts[i]=(kTokens-kLengths[i])/2;
    hip_check(hipMemset(k.p,0xa5,k.bytes),"initialize K"); hip_check(hipMemset(v.p,0xa5,v.bytes),"initialize V");
    upload();
  }
  std::string prefix() const { return protect?"protected":"compact"; }
  void upload() {
    new_k.put(host_k.data(),new_k.bytes);new_v.put(host_v.data(),new_v.bytes);
    locations.put(host_locations.data(),locations.bytes);query.put(host_q.data(),query.bytes);
    indices.put(host_indices.data(),indices.bytes);starts.put(host_starts.data(),starts.bytes);
    lengths.put(host_lengths.data(),lengths.bytes);
  }
  void reset_status() {
    const uint32_t values[]={0x13579bdf,0x2468abcd,0x13579bdf};status.put(values,sizeof(values));
  }
  void check_status(bool value_refusal) {
    const auto bytes=status.get();uint32_t words[3];std::memcpy(words,bytes.data(),12);
    require(words[0]==0x13579bdf && words[2]==0x13579bdf,"status canary");
    require(words[1]==(value_refusal?1u:0u),"device status mismatch");
  }
  void reset_output() {hip_check(hipMemset(output.p,0xa5,output.bytes),"output canary fill");}
  int attention(int rows=int(kLengths.size()),float scale=0.0625f) {
    return lib.attention(&cache,kIdentity,query.as<uint16_t>(),uint64_t(rows)*6144,
      indices.as<int64_t>(),kTokens,starts.as<int32_t>(),lengths.as<int32_t>(),rows,scale,
      output.as<uint16_t>()+32,uint64_t(rows)*6144,status.as<int>()+1,nullptr);
  }
  std::vector<uint8_t> retain_output(const std::string& tag,int rows=int(kLengths.size())) {
    check_status(false);
    const auto bytes=output.get();const size_t used=size_t(rows)*6144;
    require(std::all_of(bytes.begin(),bytes.begin()+64,[](uint8_t x){return x==0xa5;}),"left output canary");
    require(std::all_of(bytes.begin()+64+used,bytes.end(),[](uint8_t x){return x==0xa5;}),"right output canary");
    for(size_t i=64;i<64+used;i+=2) {
      uint16_t value;std::memcpy(&value,bytes.data()+i,2);
      require((value&0x7f80u)!=0x7f80u,"nonfinite attention output");
    }
    out.record(prefix()+"/"+tag,bytes.data()+64,used);accepted+=unsigned(rows);
    return {bytes.begin()+64,bytes.begin()+64+used};
  }
  void success(const std::string& tag,int rows=int(kLengths.size())) {
    reset_status();reset_output();require(attention(rows)==0,"valid attention refused");
    (void)retain_output(tag,rows);
  }
  void synchronized_timings(int length,unsigned repeats,unsigned warmups) {
    const int32_t zero=0,n=length;starts.put(&zero,4);lengths.put(&n,4);
    const std::string tag="timing/"+std::to_string(length)+"/";
    // Retain a real attention output before explicit warmups and measurements.
    reset_status();reset_output();hip_check(hipDeviceSynchronize(),"reference preparation sync");
    require(attention(1)==0,"reference attention refused");
    hip_check(hipDeviceSynchronize(),"reference completion sync");
    const auto reference=retain_output(tag+"reference",1);
    for(unsigned r=0;r<warmups;++r) {
      reset_status();reset_output();hip_check(hipDeviceSynchronize(),"warmup preparation sync");
      require(attention(1)==0,"warmup attention refused");
      hip_check(hipDeviceSynchronize(),"warmup completion sync");
      require(retain_output(tag+"warmup-"+std::to_string(r),1)==reference,"warmup output changed");
    }
    for(unsigned r=0;r<repeats;++r) {
      reset_status();reset_output();hip_check(hipDeviceSynchronize(),"timing preparation sync");
      // Same-thread host-wall snapshots add no GPU work and stay outside the
      // enclosing call timer; these are not kernel-only or decoder timings.
      const auto previous_phase=lib.phase_last?lib.phase():uq_cache_phase_profile_v1{};
      const auto begin=std::chrono::steady_clock::now();
      const int result=attention(1);
      const hipError_t completion=hipDeviceSynchronize();
      const auto end=std::chrono::steady_clock::now();
      const double ms=std::chrono::duration<double,std::milli>(end-begin).count();
      require(result==0,"timed attention refused");hip_check(completion,"timing completion sync");
      require(std::isfinite(ms)&&ms>0,"invalid timing");
      if(lib.phase_last) {
        const auto phase=lib.phase();
        require(previous_phase.sequence<UINT64_MAX && phase.sequence==previous_phase.sequence+1,
                "stale or skipped phase profile sequence");
        const auto whole_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
        require(whole_ns>0 && phase.total_ns<=uint64_t(whole_ns),"phase profile exceeds enclosing call");
        phase_timing.push_back(phase);
      }
      require(retain_output(tag+"repetition-"+std::to_string(r),1)==reference,"timed output changed");
      timing.push_back({double(length),double(r),ms});
    }
  }
  void refusal(const std::string& label,bool device_refusal=true,float scale=0.0625f) {
    reset_status();reset_output();require(attention(1,scale)==1,"refusal accepted: "+label);
    const auto bytes=output.get();require(std::all_of(bytes.begin(),bytes.end(),[](uint8_t x){return x==0xa5;}),"refusal wrote output: "+label);
    if(device_refusal) check_status(true);
    else {
      const auto b=status.get();uint32_t w[3];std::memcpy(w,b.data(),12);
      require(w[0]==0x13579bdf && w[1]==0x2468abcd && w[2]==0x13579bdf,"host refusal touched status");
    }
    ++refused;
  }
  void run(unsigned repeats,const std::string& timing_protocol,unsigned warmups) {
    reset_status(); require(lib.append(&cache,kIdentity,new_k.as<uint16_t>(),new_v.as<uint16_t>(),
      new_k.bytes,locations.as<int64_t>(),kTokens,status.as<int>()+1,nullptr)==0,"valid append refused");check_status(false);
    const auto saved_k=k.get(),saved_v=v.get();
    out.record(prefix()+"/append-k",saved_k.data(),saved_k.size());out.record(prefix()+"/append-v",saved_v.data(),saved_v.size());
    success("mixed-lengths");
    // Each failure is isolated and the original input is restored before the next.
    const int32_t old_start=host_starts[0],old_length=host_lengths[0];
    for(int32_t bad:{-1,kTokens+1}) {starts.put(&bad,4);refusal("start");} starts.put(&old_start,4);
    for(int32_t bad:{0,-1,131073,kTokens+1}) {lengths.put(&bad,4);refusal("length");}lengths.put(&old_length,4);
    const size_t index_offset=size_t(old_start)*8; const int64_t good_slot=host_indices[size_t(old_start)];
    for(int64_t bad:{int64_t(0),int64_t(-1),int64_t(kSlots)}) {indices.put(&bad,8,index_offset);refusal("slot");}indices.put(&good_slot,8,index_offset);
    for(uint16_t bad:{uint16_t(0x7fc1),uint16_t(0x7f80),uint16_t(0x4481)}) {
      query.put(&bad,2,3071*2);refusal("query");
    }query.put(&host_q[3071],2,3071*2);
    refusal("scale-zero",false,0.0f);refusal("scale-nan",false,NAN);
    cache.identity[0]^=1;refusal("identity",false);cache.identity[0]^=1;
    for(Device* buffer:{&k,&v}) {
      const auto& original=(buffer==&k)?saved_k:saved_v;
      for(int head:{0,1}) {
        if(protect) {
          for(int channel:{0,31,32,255}) {
            const size_t offset=(size_t(good_slot)*512+size_t(head)*256+channel)*2;
            const uint16_t bad=0x7f80;buffer->put(&bad,2,offset);refusal("protected cache");buffer->put(original.data()+offset,2,offset);
          }
        } else {
          for(int group:{0,7}) {
            const size_t offset=(size_t(good_slot)*2+head)*136+group*17;
            for(uint8_t bad:{uint8_t(0x08),uint8_t(0x80)}) {
              buffer->put(&bad,1,offset+15);refusal("nibble");buffer->put(original.data()+offset+15,1,offset+15);
            }
            for(uint8_t bad:{uint8_t(buffer==&k?140:136),uint8_t(255)}) {
              buffer->put(&bad,1,offset+16);refusal("cache scale");buffer->put(original.data()+offset+16,1,offset+16);
            }
          }
        }
      }
    }
    // Actual writer refusal is transactional with respect to both cache buffers.
    for(int which=0;which<4;++which) {
      reset_status();const uint16_t bad_value=0x7fc1;const int64_t bad_slot=0;
      if(which==0)new_k.put(&bad_value,2);
      if(which==1)new_v.put(&bad_value,2,511*2);
      if(which==2)locations.put(&bad_slot,8);
      if(which==3)locations.put(&host_locations[0],8,8);
      require(lib.append(&cache,kIdentity,new_k.as<uint16_t>(),new_v.as<uint16_t>(),
        new_k.bytes,locations.as<int64_t>(),kTokens,status.as<int>()+1,nullptr)==1,"invalid append accepted");check_status(true);
      require(k.get()==saved_k && v.get()==saved_v,"refused append changed cache");++refused;upload();
    }
    require(k.get()==saved_k && v.get()==saved_v,"attention changed cache");
    success("after-refusals");
    if(!protect) {
      // Valid decoder boundary scales on both heads, including subnormal decode.
      std::vector<uint8_t> boundary_k=saved_k,boundary_v=saved_v;
      for(int head=0;head<2;++head) for(int group=0;group<8;++group) {
        const size_t base=(size_t(good_slot)*2+head)*136+group*17;
        constexpr uint8_t scales[]={0,1,2,126,127,128,134,135};
        for(int byte=0;byte<16;++byte) {boundary_k[base+byte]=0x21;boundary_v[base+byte]=0xba;}
        boundary_k[base+16]=group==7?139:scales[group];boundary_v[base+16]=scales[group];
      }
      k.put(boundary_k.data(),k.bytes);v.put(boundary_v.data(),v.bytes);success("boundary-scales",1);
      k.put(saved_k.data(),k.bytes);v.put(saved_v.data(),v.bytes);
    }
    // Timings follow this process's checks. The parent accepts them only after
    // all old/candidate correctness records match on both devices/cache modes.
    if(timing_protocol==kTimingV2) {
      for(int length:{128,1024,3966}) synchronized_timings(length,repeats,warmups);
    } else for(int length:{128,1024,3966}) for(unsigned r=0;r<repeats;++r) {
      const int32_t zero=0,n=length;starts.put(&zero,4);lengths.put(&n,4);reset_status();reset_output();
      const auto begin=std::chrono::steady_clock::now();require(attention(1)==0,"timed attention refused");
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();check_status(false);
      require(std::isfinite(ms)&&ms>0,"invalid timing");timing.push_back({double(length),double(r),ms});
    }
  }
};

int decimal(const char* text) {
  char* end=nullptr;errno=0;const long v=strtol(text,&end,10);
  require(errno==0&&end!=text&&*end=='\0'&&v>=0&&v<=16,"invalid numeric argument");return int(v);
}
} // namespace

int main(int argc,char** argv) {
  try {
    const char* library=nullptr;const char* output=nullptr;const char* bdf=nullptr;const char* uuid=nullptr;
    int device=-1;unsigned repeats=0,warmups=0,repeat_options=0;bool execute=false;
    bool protocol_seen=false,warmups_seen=false,phase_profile=false;std::string mode="both",timing_protocol="legacy-v1";
    for(int i=1;i<argc;++i) {
      const std::string a=argv[i];
      if(a=="--execute-gpu") {require(!execute,"duplicate execution flag");execute=true;continue;}
      if(a=="--phase-profile") {require(!phase_profile,"duplicate phase profile flag");phase_profile=true;continue;}
      require(i+1<argc,"missing argument");const char* value=argv[++i];
      if(a=="--library") {require(!library,"duplicate library");library=value;}
      else if(a=="--output") {require(!output,"duplicate output");output=value;}
      else if(a=="--expected-bdf") {require(!bdf,"duplicate BDF");bdf=value;}
      else if(a=="--expected-uuid") {require(!uuid,"duplicate UUID");uuid=value;}
      else if(a=="--device") {require(device<0,"duplicate device");device=decimal(value);}
      else if(a=="--timing-repeats") {require(repeats==0,"duplicate timing");++repeat_options;repeats=unsigned(decimal(value));require(repeats<=5,"timing cap");}
      else if(a=="--timing-protocol") {require(!protocol_seen,"duplicate timing protocol");protocol_seen=true;timing_protocol=value;}
      else if(a=="--timing-warmups") {require(!warmups_seen,"duplicate timing warmups");warmups_seen=true;
        require(std::string(value)=="0"||std::string(value)=="2","invalid timing warmups");warmups=unsigned(value[0]-'0');}
      else if(a=="--cache-mode") {mode=value;require(mode=="both"||mode=="bf16"||mode=="kv4","cache mode");}
      else throw std::runtime_error("unknown argument");
    }
    require(timing_protocol=="legacy-v1"||timing_protocol==kTimingV2,"unknown timing protocol");
    require(timing_protocol==kTimingV2?(warmups_seen&&warmups==2&&repeats==5&&repeat_options==1):warmups==0,"timing protocol/warmup/repeat binding");
    require(!phase_profile||timing_protocol==kTimingV2,"phase profile requires v2 timing");
    require(execute&&library&&output&&bdf&&uuid&&device>=0,"explicit execution, library, output, device, BDF and UUID required");
    require(library[0]=='/'&&output[0]=='/',"absolute paths required");
    int count=0;hip_check(hipGetDeviceCount(&count),"device count");require(count==2&&device<count,"exact two-device allocation required");
    hip_check(hipSetDevice(device),"set device");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,device),"device properties");
    require(std::strncmp(prop.gcnArchName,"gfx1201",7)==0 && (prop.gcnArchName[7]=='\0'||prop.gcnArchName[7]==':'),"architecture mismatch");
    char actual_bdf[64]{};hip_check(hipDeviceGetPCIBusId(actual_bdf,sizeof(actual_bdf),device),"device BDF");
    require(std::string(actual_bdf)==bdf,"device BDF mismatch");
    hipUUID native_uuid{};hip_check(hipDeviceGetUuid(&native_uuid,device),"device UUID");
    std::string actual_uuid;constexpr char hex[]="0123456789abcdef";
    for(unsigned i=0;i<16;++i) {
      if(i==4||i==6||i==8||i==10)actual_uuid+='-';
      const auto byte=static_cast<unsigned char>(native_uuid.bytes[i]);
      actual_uuid+=hex[byte>>4];actual_uuid+=hex[byte&15];
    }
    require(actual_uuid==uuid,"device UUID mismatch");
    Library lib(library,phase_profile);Output out(output);Device decode(8192*2*4);
    hipLaunchKernelGGL(decoder_bits,dim3(32),dim3(256),0,nullptr,decode.as<uint32_t>());
    hip_check(hipGetLastError(),"decoder launch");hip_check(hipDeviceSynchronize(),"decoder sync");
    const auto scalar=decode.get();
    for(size_t i=0;i<scalar.size();i+=8) require(std::memcmp(scalar.data()+i,scalar.data()+i+4,4)==0,"decoder differs from GPU generic oracle");
    out.record("decoder",scalar.data(),scalar.size());
    unsigned accepted=0,refused=0;
    std::vector<std::pair<bool,std::array<double,3>>> timing;
    std::vector<uq_cache_phase_profile_v1> phase_timing;
    for(bool protect:{true,false}) {
      if((protect&&mode=="kv4")||(!protect&&mode=="bf16")) continue;
      Fixture fixture(lib,out,protect);fixture.run(repeats,timing_protocol,warmups);accepted+=fixture.accepted;refused+=fixture.refused;
      for(const auto& row:fixture.timing)timing.push_back({protect,row});
      phase_timing.insert(phase_timing.end(),fixture.phase_timing.begin(),fixture.phase_timing.end());
    }
    hip_check(hipDeviceSynchronize(),"final sync");decode.release();
    require(free_errors==0 && device_bytes==0,"device buffer release failed");out.finish();
    std::cout<<"{\"schema\":1,\"status\":\"NATIVE_FIXTURE_PASSED\",\"qualification\":false,\"cache_mode\":\""<<mode<<"\",\"device\":"<<device
      <<",\"bdf\":\""<<actual_bdf<<"\",\"decoder_comparisons\":8192,\"attention_rows\":"<<accepted
      <<",\"uuid\":\""<<actual_uuid<<"\",\"visible_devices\":"<<count<<",\"architecture\":\""<<prop.gcnArchName<<"\""
      <<",\"refusals\":"<<refused<<",\"records\":"<<out.records<<",\"output_bytes\":"<<out.total
      <<",\"peak_explicit_device_buffer_bytes\":"<<peak_device_bytes<<",\"device_buffer_free_errors\":"<<free_errors
      <<",\"timing_repeats\":"<<repeats;
    if(timing_protocol==kTimingV2)std::cout<<",\"timing_protocol\":\""<<timing_protocol<<"\",\"timing_warmups\":"<<warmups;
    if(phase_profile)std::cout<<",\"phase_profile\":true";
    std::cout<<",\"timings\":[";
    for(size_t i=0;i<timing.size();++i) {
      if(i)std::cout<<',';const auto& row=timing[i];
      std::cout<<"{\"protected\":"<<(row.first?"true":"false")<<",\"length\":"<<int(row.second[0])
        <<",\"repetition\":"<<int(row.second[1])<<",\"synchronized_call_ms\":"<<std::setprecision(17)<<row.second[2];
      if(phase_profile) {
        const auto& phase=phase_timing.at(i);
        std::cout<<",\"phase_wall\":{\"schema\":1,\"sequence\":"<<phase.sequence
          <<",\"host_checks_ns\":"<<phase.host_checks_ns<<",\"validation_status_ns\":"<<phase.validation_status_ns
          <<",\"reader_sync_ns\":"<<phase.reader_sync_ns<<",\"total_ns\":"<<phase.total_ns<<'}';
      }
      std::cout<<'}';
    }
    std::cout<<"]}\n";return 0;
  } catch(const std::exception& e) {std::cerr<<"native cache fixture failed: "<<e.what()<<'\n';return 1;}
}
