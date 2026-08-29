#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

namespace {
constexpr int k_top_k = 6;
constexpr size_t k_slot_bytes = 7096320;
constexpr size_t k_gate_bytes = 2162688;
constexpr size_t k_up_bytes = 2162688;
constexpr size_t k_down_bytes = 2752512;
constexpr size_t k_gate_offset = 0;
constexpr size_t k_up_offset = k_gate_bytes;
constexpr size_t k_down_offset = k_gate_bytes + k_up_bytes;

struct BackendDeleter { void operator()(ggml_backend_t p) const { if (p) ggml_backend_free(p); } };
struct ModelDeleter { void operator()(llama_model * p) const { if (p) llama_model_free(p); } };
using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;

double ms(clock_type::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m=v.size()/2;
    return v.size()&1?v[m]:0.5*(v[m-1]+v[m]);
}

std::vector<uint8_t> read_bytes(const fs::path &p){
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());
    f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0);if(n<0)throw std::runtime_error("cannot stat "+p.string());
    std::vector<uint8_t>o(static_cast<size_t>(n));if(!o.empty()&&!f.read(reinterpret_cast<char*>(o.data()),static_cast<std::streamsize>(o.size())))throw std::runtime_error("short read "+p.string());return o;
}

std::vector<std::string> split_names(const std::string &s){
    std::vector<std::string>out;size_t b=0;while(b<=s.size()){size_t e=s.find(',',b);if(e==std::string::npos)e=s.size();out.push_back(s.substr(b,e-b));if(e==s.size())break;b=e+1;}return out;
}

struct Options { fs::path model,package,output; uint32_t l2_mib=8192; int repeats=3; };
Options parse(int argc,char**argv){Options o;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing value");std::string k=argv[i],v=argv[i+1];if(k=="--model")o.model=fs::absolute(v);else if(k=="--package")o.package=fs::absolute(v);else if(k=="--output-file")o.output=fs::absolute(v);else if(k=="--l2-mib")o.l2_mib=static_cast<uint32_t>(std::stoul(v));else if(k=="--repeats")o.repeats=std::stoi(v);else throw std::runtime_error("unknown arg "+k);}if(o.model.empty()||o.package.empty()||o.output.empty()||o.l2_mib<64||o.repeats<1||o.repeats>5||fs::exists(o.output))throw std::runtime_error("invalid request");return o;}

struct InfoSnapshot {
    ggml_siliangem_cache_info v{};
    explicit InfoSnapshot(ggml_backend_t cpu){v.struct_size=sizeof(v);if(!ggml_backend_cpu_siliangem_query(cpu,&v))throw std::runtime_error("L2 query failed");}
};

json delta_json(const ggml_siliangem_cache_info &a,const ggml_siliangem_cache_info &b){
    return {{"hits",b.hits-a.hits},{"misses",b.misses-a.misses},{"bytes_read",b.bytes_read-a.bytes_read},{"wait_calls",b.wait_calls-a.wait_calls},{"wait_ns",b.wait_ns-a.wait_ns},{"admissions",b.policy_admissions-a.policy_admissions},{"evictions",b.policy_evictions-a.policy_evictions},{"rejections",b.policy_rejections-a.policy_rejections}};
}

void prepare_blocking(ggml_backend_t cpu,uint32_t layer,const std::vector<int32_t>&ids){
    if(ids.empty())return;
    if(!ggml_backend_cpu_siliangem_prepare_experts(cpu,layer,ids.data(),static_cast<uint32_t>(ids.size())))throw std::runtime_error("blocking L2 prepare failed");
}

void fill_cache(ggml_backend_t cpu,const llama_siliang_expert_source &source,uint32_t target_slots,const std::set<uint32_t>&reserved_layers){
    uint32_t loaded=0;
    for(uint32_t layer=0;layer<source.n_layers&&loaded<target_slots;++layer){
        if(reserved_layers.count(layer))continue;
        const uint32_t count=std::min<uint32_t>(source.n_experts,target_slots-loaded);
        std::vector<int32_t>ids(count);for(uint32_t i=0;i<count;++i)ids[i]=static_cast<int32_t>(i);
        prepare_blocking(cpu,layer,ids);loaded+=count;
        std::fprintf(stderr,"l2-admission: fill layer=%u count=%u loaded=%u/%u\n",layer,count,loaded,target_slots);
    }
    uint32_t capacity=0,occupied=0;if(!ggml_backend_cpu_siliangem_cache_occupancy(cpu,&capacity,&occupied)||occupied!=target_slots)throw std::runtime_error("L2 fill did not reach requested occupancy");
}

void verify_layer0_package(ggml_backend_t cpu,const llama_siliang_expert_source &source,const std::array<int32_t,k_top_k>&ids,const std::vector<uint8_t>&package_experts){
    const auto names=split_names(source.part_names);if(names.size()!=source.n_parts)throw std::runtime_error("source part names invalid");
    for(int pos=0;pos<k_top_k;++pos){
        for(uint32_t part=0;part<source.n_parts;++part){
            const size_t source_index=part;const size_t bytes=source.part_bytes[source_index];size_t expected_offset=0,expected_bytes=0;
            const std::string &name=names[part];
            if(name.find("gate")!=std::string::npos){expected_offset=k_gate_offset;expected_bytes=k_gate_bytes;}
            else if(name.find("up")!=std::string::npos){expected_offset=k_up_offset;expected_bytes=k_up_bytes;}
            else if(name.find("down")!=std::string::npos){expected_offset=k_down_offset;expected_bytes=k_down_bytes;}
            else throw std::runtime_error("unknown source part name "+name);
            if(bytes!=expected_bytes)throw std::runtime_error("source/package part byte mismatch");
            std::vector<uint8_t>got(bytes);if(!ggml_backend_cpu_siliangem_copy_cached_part(cpu,0,static_cast<uint32_t>(ids[pos]),part,got.data(),got.size()))throw std::runtime_error("cached part copy failed");
            const uint8_t *expected=package_experts.data()+static_cast<size_t>(pos)*k_slot_bytes+expected_offset;
            if(std::memcmp(got.data(),expected,bytes)!=0)throw std::runtime_error("L2 expert bytes differ from frozen package");
        }
    }
}

} // namespace

int main(int argc,char**argv) try {
    const Options o=parse(argc,argv);
    llama_backend_init();
    llama_model_params mp=llama_model_default_params();mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.no_alloc=true;mp.load_mtp=false;
    ModelPtr model(llama_model_load_from_file(o.model.string().c_str(),mp));if(!model)throw std::runtime_error("metadata model load failed");
    const auto &source=model->siliang_expert_source;if(!source.valid()||source.kind!=llama_siliang_expert_source_kind::expert_major)throw std::runtime_error("DS4 expert-major source metadata unavailable");

    BackendPtr cpu(ggml_backend_cpu_init());if(!cpu)throw std::runtime_error("CPU backend init failed");
    ggml_siliangem_cache_config config={};config.struct_size=sizeof(config);config.enabled=1;config.capacity_mib=o.l2_mib;config.policy=GGML_SILIANGEM_CACHE_POLICY_LRU;config.deferred_io=1;config.verbose=0;config.memory_report=0;config.mmap_prefetch=0;
    ggml_siliangem_source_desc desc={};desc.struct_size=sizeof(desc);desc.kind=GGML_SILIANGEM_SOURCE_EXPERT_MAJOR;desc.path=source.path.c_str();desc.n_layers=source.n_layers;desc.n_experts=source.n_experts;desc.n_parts=source.n_parts;desc.base=source.base.data();desc.stride=source.stride.data();desc.part_offset=source.part_offset.data();desc.part_bytes=source.part_bytes.data();desc.part_names=source.part_names.c_str();
    if(!ggml_backend_cpu_siliangem_configure(cpu.get(),&config,&desc))throw std::runtime_error("L2 configure failed");
    InfoSnapshot configured(cpu.get());if(!configured.v.ready)throw std::runtime_error("L2 not ready after configure");

    auto package_manifest_bytes=read_bytes(o.package/"m03-work-unit-package.json");json package=json::parse(package_manifest_bytes.begin(),package_manifest_bytes.end());
    std::array<int32_t,k_top_k> route_ids{};for(int i=0;i<k_top_k;++i)route_ids[i]=package.at("work_unit").at("expert_ids").at(i).get<int32_t>();
    auto package_experts=read_bytes(o.package/"m03-selected-experts.bin");if(package_experts.size()!=k_top_k*k_slot_bytes)throw std::runtime_error("frozen package expert geometry mismatch");

    // Reserve layers used by all measurement rounds so the one-time full-cache
    // fill cannot accidentally seed a future target key.
    std::set<uint32_t>reserved_layers;for(int round=0;round<o.repeats;++round)for(int misses=0;misses<=6;++misses)reserved_layers.insert(static_cast<uint32_t>(round*7+(6-misses)));
    if(*reserved_layers.rbegin()>=source.n_layers)throw std::runtime_error("not enough routed layers for requested repeats");

    const auto fill_start=clock_type::now();fill_cache(cpu.get(),source,configured.v.capacity_slots,reserved_layers);const auto fill_end=clock_type::now();
    InfoSnapshot full(cpu.get());

    std::array<std::vector<double>,7>prepare_ms,wait_ms,total_ms;
    std::array<json,7>deltas;for(auto &d:deltas)d=json::array();
    json samples=json::array();bool package_verified=false;

    for(int round=0;round<o.repeats;++round){
        // Rotate order across rounds to reduce monotonic thermal/order bias.
        std::array<int,7>order = round%2==0 ? std::array<int,7>{6,5,4,3,2,1,0} : std::array<int,7>{0,1,2,3,4,5,6};
        for(int misses:order){
            const uint32_t layer=static_cast<uint32_t>(round*7+(6-misses));
            const int hits_expected=6-misses;
            if(hits_expected>0){std::vector<int32_t>hits(route_ids.begin(),route_ids.begin()+hits_expected);prepare_blocking(cpu.get(),layer,hits);}
            uint32_t cap=0,occ=0;if(!ggml_backend_cpu_siliangem_cache_occupancy(cpu.get(),&cap,&occ)||occ!=cap)throw std::runtime_error("L2 must remain full before timed request");

            InfoSnapshot before(cpu.get());
            std::array<int32_t,k_top_k>ordered{};uint32_t n_hits=0,n_misses=0,n_active=0;
            const auto t0=clock_type::now();
            if(!ggml_backend_cpu_siliangem_prepare_experts_async(cpu.get(),layer,route_ids.data(),k_top_k,ordered.data(),k_top_k,&n_hits,&n_misses,&n_active))throw std::runtime_error("timed prepare_async failed");
            const auto t1=clock_type::now();
            if(!ggml_backend_cpu_siliangem_wait_experts(cpu.get()))throw std::runtime_error("timed wait_experts failed");
            const auto t2=clock_type::now();
            InfoSnapshot after(cpu.get());
            if(n_active!=k_top_k||n_hits!=static_cast<uint32_t>(hits_expected)||n_misses!=static_cast<uint32_t>(misses))throw std::runtime_error("forced hit/miss geometry did not hold");

            const double pms=ms(t1-t0),wms=ms(t2-t1),tms=ms(t2-t0);prepare_ms[misses].push_back(pms);wait_ms[misses].push_back(wms);total_ms[misses].push_back(tms);json delta=delta_json(before.v,after.v);deltas[misses].push_back(delta);
            samples.push_back({{"round",round},{"layer",layer},{"expected_hits",hits_expected},{"expected_misses",misses},{"prepare_async_ms",pms},{"wait_ms",wms},{"total_admission_ms",tms},{"cache_delta",delta}});

            if(!package_verified&&round==0&&misses==6&&layer==0){verify_layer0_package(cpu.get(),source,route_ids,package_experts);package_verified=true;}
        }
    }

    json cells=json::array();
    for(int misses=0;misses<=6;++misses){uint64_t bytes=0,evictions=0;for(const auto &d:deltas[misses]){bytes+=d.at("bytes_read").get<uint64_t>();evictions+=d.at("evictions").get<uint64_t>();}
        cells.push_back({{"misses",misses},{"hits",6-misses},{"prepare_async_ms_median",median(prepare_ms[misses])},{"wait_ms_median",median(wait_ms[misses])},{"total_admission_ms_median",median(total_ms[misses])},{"total_admission_ms_first",total_ms[misses].empty()?0.0:total_ms[misses].front()},{"bytes_read_mean",deltas[misses].empty()?0.0:static_cast<double>(bytes)/deltas[misses].size()},{"evictions_mean",deltas[misses].empty()?0.0:static_cast<double>(evictions)/deltas[misses].size()}});
    }

    json result={{"schema","siliang-v013-ds4-full-l2-admission-v1"},{"complete",true},{"base_checkpoint","738c1804a88e9f99742714d7fbc354ecf7e0b279"},{"model",o.model.string()},{"source_kind","expert-major"},{"source_parts",source.part_names},{"l2_mib",o.l2_mib},{"capacity_slots",configured.v.capacity_slots},{"full_cache_fill_ms",ms(fill_end-fill_start)},{"full_cache_occupied_slots",full.v.occupied_slots},{"repeats",o.repeats},{"package_layer0_bytes_exact",package_verified},{"cells",cells},{"samples",samples}};
    std::ofstream out(o.output,std::ios::binary);if(!out)throw std::runtime_error("cannot create output");out<<result.dump(2)<<"\n";
    ggml_backend_cpu_siliangem_reset(cpu.get());llama_backend_free();std::fprintf(stderr,"ds4-l2-admission-assay: complete: %s\n",o.output.string().c_str());return 0;
} catch(const std::exception&e){std::fprintf(stderr,"ds4-l2-admission-assay: fatal: %s\n",e.what());return 2;}
