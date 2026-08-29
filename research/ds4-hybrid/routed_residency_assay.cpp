#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

namespace {
constexpr int64_t k_embd = 4096;
constexpr int64_t k_ff = 2048;
constexpr int k_top_k = 6;
constexpr size_t k_slot_bytes = 7096320;
constexpr size_t k_gate_bytes = 2162688;
constexpr size_t k_up_bytes = 2162688;
constexpr size_t k_down_bytes = 2752512;
constexpr size_t k_gate_offset = 0;
constexpr size_t k_up_offset = k_gate_bytes;
constexpr size_t k_down_offset = k_gate_bytes + k_up_bytes;

struct BackendDeleter { void operator()(ggml_backend_t p) const { if (p) ggml_backend_free(p); } };
struct BufferDeleter { void operator()(ggml_backend_buffer_t p) const { if (p) ggml_backend_buffer_free(p); } };
struct ContextDeleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
struct AllocDeleter { void operator()(ggml_gallocr_t p) const { if (p) ggml_gallocr_free(p); } };
using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using BufferPtr = std::unique_ptr<ggml_backend_buffer, BufferDeleter>;
using ContextPtr = std::unique_ptr<ggml_context, ContextDeleter>;
using AllocPtr = std::unique_ptr<ggml_gallocr, AllocDeleter>;

double ms(clock_type::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size()/2;
    return v.size() & 1 ? v[m] : 0.5*(v[m-1]+v[m]);
}

std::vector<uint8_t> read_bytes(const fs::path & p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p.string());
    f.seekg(0,std::ios::end); auto n=f.tellg(); f.seekg(0); if(n<0) throw std::runtime_error("cannot stat " + p.string());
    std::vector<uint8_t> out(static_cast<size_t>(n));
    if(!out.empty() && !f.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(out.size()))) throw std::runtime_error("short read " + p.string());
    return out;
}

ContextPtr make_context(size_t tensors=256) {
    ggml_init_params p={}; p.mem_size=tensors*ggml_tensor_overhead()+ggml_graph_overhead_custom(512,false)+32768; p.no_alloc=true;
    auto *ctx=ggml_init(p); if(!ctx) throw std::runtime_error("ggml_init failed"); return ContextPtr(ctx);
}

struct ExpertArena {
    ContextPtr ctx;
    BufferPtr buffer;
    ggml_tensor *carrier=nullptr,*gate=nullptr,*up=nullptr,*down=nullptr;
    uint8_t *base=nullptr;

    static ExpertArena create(ggml_backend_buffer_type_t buft) {
        ExpertArena a; a.ctx=make_context(16);
        a.buffer.reset(ggml_backend_buft_alloc_buffer(buft,k_top_k*k_slot_bytes));
        if(!a.buffer) throw std::runtime_error("expert arena allocation failed");
        ggml_backend_buffer_set_usage(a.buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        a.base=static_cast<uint8_t*>(ggml_backend_buffer_get_base(a.buffer.get())); if(!a.base) throw std::runtime_error("expert arena base missing");
        a.carrier=ggml_new_tensor_1d(a.ctx.get(),GGML_TYPE_I8,k_top_k*k_slot_bytes);
        a.gate=ggml_new_tensor_3d(a.ctx.get(),GGML_TYPE_IQ2_XXS,k_embd,k_ff,k_top_k);
        a.up=ggml_new_tensor_3d(a.ctx.get(),GGML_TYPE_IQ2_XXS,k_embd,k_ff,k_top_k);
        a.down=ggml_new_tensor_3d(a.ctx.get(),GGML_TYPE_Q2_K,k_ff,k_embd,k_top_k);
        for(auto *t:{a.gate,a.up,a.down}) { t->nb[2]=k_slot_bytes; t->nb[3]=k_top_k*k_slot_bytes; }
        if(ggml_backend_tensor_alloc(a.buffer.get(),a.carrier,a.base)!=GGML_STATUS_SUCCESS ||
           ggml_backend_tensor_alloc(a.buffer.get(),a.gate,a.base+k_gate_offset)!=GGML_STATUS_SUCCESS ||
           ggml_backend_tensor_alloc(a.buffer.get(),a.up,a.base+k_up_offset)!=GGML_STATUS_SUCCESS ||
           ggml_backend_tensor_alloc(a.buffer.get(),a.down,a.base+k_down_offset)!=GGML_STATUS_SUCCESS) throw std::runtime_error("expert arena tensor bind failed");
        return a;
    }
};

void require_cuda(ggml_backend_cuda_siliang_status s,const char *what){if(s!=GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS)throw std::runtime_error(std::string(what)+" status="+std::to_string(s));}

struct CopyEngine {
    ggml_backend_t cuda=nullptr;
    ggml_backend_cuda_siliang_stream_t stream=nullptr;
    ggml_backend_cuda_siliang_event_t done=nullptr;
    explicit CopyEngine(ggml_backend_t backend):cuda(backend){require_cuda(ggml_backend_cuda_siliang_stream_create(cuda,&stream),"stream create");require_cuda(ggml_backend_cuda_siliang_event_create(cuda,&done),"event create");}
    ~CopyEngine(){if(stream)(void)ggml_backend_cuda_siliang_stream_synchronize(stream);if(done)(void)ggml_backend_cuda_siliang_event_destroy(done);if(stream)(void)ggml_backend_cuda_siliang_stream_destroy(stream);}
    void submit_slots(ExpertArena &a,const uint8_t *source,const std::vector<int>&slots){
        for(int slot:slots){const size_t s=static_cast<size_t>(slot)*k_slot_bytes;
            require_cuda(ggml_backend_cuda_siliang_h2d_async(stream,a.carrier,source+s+k_gate_offset,s+k_gate_offset,k_gate_bytes),"gate H2D");
            require_cuda(ggml_backend_cuda_siliang_h2d_async(stream,a.carrier,source+s+k_up_offset,s+k_up_offset,k_up_bytes),"up H2D");
            require_cuda(ggml_backend_cuda_siliang_h2d_async(stream,a.carrier,source+s+k_down_offset,s+k_down_offset,k_down_bytes),"down H2D");
        }
        require_cuda(ggml_backend_cuda_siliang_event_record(stream,done),"copy record");
    }
    void wait_done(){require_cuda(ggml_backend_cuda_siliang_event_synchronize(done),"copy sync");}
    void main_wait(){require_cuda(ggml_backend_cuda_siliang_main_stream_wait_event(cuda,done),"main stream copy wait");}
    void copy_slots(ExpertArena &a,const uint8_t *source,const std::vector<int>&slots){submit_slots(a,source,slots);wait_done();}

    struct StageStats { double memcpy_ms=0.0; double enqueue_ms=0.0; };
    StageStats stage_and_submit_slots(ExpertArena &a,const uint8_t *pageable,uint8_t *stage,const std::vector<int>&slots){
        StageStats stats;
        for(int slot:slots){
            const size_t s=static_cast<size_t>(slot)*k_slot_bytes;
            const std::array<std::pair<size_t,size_t>,3> parts={{{k_gate_offset,k_gate_bytes},{k_up_offset,k_up_bytes},{k_down_offset,k_down_bytes}}};
            for(const auto &part:parts){
                const auto m0=clock_type::now();
                std::memcpy(stage+s+part.first,pageable+s+part.first,part.second);
                const auto m1=clock_type::now();
                stats.memcpy_ms+=ms(m1-m0);
                const auto e0=clock_type::now();
                require_cuda(ggml_backend_cuda_siliang_h2d_async(stream,a.carrier,stage+s+part.first,s+part.first,part.second),"staged H2D");
                const auto e1=clock_type::now();
                stats.enqueue_ms+=ms(e1-e0);
            }
        }
        require_cuda(ggml_backend_cuda_siliang_event_record(stream,done),"staged copy record");
        return stats;
    }
};

struct RoutedGraph {
    ContextPtr ctx;
    AllocPtr alloc;
    ggml_cgraph *graph=nullptr;
    ggml_tensor *input=nullptr,*ids=nullptr,*weights=nullptr,*output=nullptr;
    int count=0;

    static RoutedGraph build(ggml_backend_t backend,ExpertArena &arena,const std::vector<int32_t>&ids_v,const std::vector<float>&weights_v,float clamp){
        if(ids_v.empty()||ids_v.size()!=weights_v.size()) throw std::runtime_error("invalid routed subset");
        RoutedGraph r; r.count=static_cast<int>(ids_v.size()); r.ctx=make_context(256);
        r.input=ggml_new_tensor_3d(r.ctx.get(),GGML_TYPE_F32,k_embd,1,1);ggml_set_input(r.input);
        r.ids=ggml_new_tensor_2d(r.ctx.get(),GGML_TYPE_I32,r.count,1);ggml_set_input(r.ids);
        r.weights=ggml_new_tensor_3d(r.ctx.get(),GGML_TYPE_F32,1,r.count,1);ggml_set_input(r.weights);
        auto *up=ggml_mul_mat_id(r.ctx.get(),arena.up,r.input,r.ids);
        auto *gate=ggml_mul_mat_id(r.ctx.get(),arena.gate,r.input,r.ids);
        if(clamp>1e-6f){up=ggml_clamp(r.ctx.get(),up,-clamp,clamp);gate=ggml_clamp(r.ctx.get(),gate,-INFINITY,clamp);}
        auto *act=ggml_swiglu_split(r.ctx.get(),gate,up);
        auto *rows=ggml_mul_mat_id(r.ctx.get(),arena.down,act,r.ids);
        rows=ggml_mul(r.ctx.get(),rows,r.weights);
        r.output=ggml_view_2d(r.ctx.get(),rows,k_embd,1,rows->nb[2],0);
        for(int i=1;i<r.count;++i){auto *v=ggml_view_2d(r.ctx.get(),rows,k_embd,1,rows->nb[2],static_cast<size_t>(i)*rows->nb[1]);r.output=ggml_add(r.ctx.get(),r.output,v);}
        ggml_set_output(r.output);
        r.graph=ggml_new_graph_custom(r.ctx.get(),256,false);ggml_build_forward_expand(r.graph,r.output);
        r.alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if(!r.alloc||!ggml_gallocr_alloc_graph(r.alloc.get(),r.graph)) throw std::runtime_error("routed graph allocation failed");
        ggml_backend_tensor_set(r.ids,ids_v.data(),0,ids_v.size()*sizeof(int32_t));
        ggml_backend_tensor_set(r.weights,weights_v.data(),0,weights_v.size()*sizeof(float));
        return r;
    }
};

std::array<float,k_embd> read_output(ggml_tensor *t){std::array<float,k_embd>v{};ggml_backend_tensor_get(t,v.data(),0,sizeof(v));return v;}
float max_abs(const std::array<float,k_embd>&a,const std::array<float,k_embd>&b){float m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,std::fabs(a[i]-b[i]));return m;}
std::array<float,k_embd> add(const std::array<float,k_embd>&a,const std::array<float,k_embd>&b){std::array<float,k_embd>o{};for(size_t i=0;i<o.size();++i)o[i]=a[i]+b[i];return o;}

class CpuWorker {
public:
    CpuWorker(ggml_backend_t cpu,RoutedGraph *graph,int affinity_cpu):cpu_(cpu),graph_(graph),affinity_cpu_(affinity_cpu),thread_(&CpuWorker::loop,this){}
    ~CpuWorker(){{std::lock_guard<std::mutex>l(m_);stop_=true;}cv_.notify_all();if(thread_.joinable())thread_.join();}
    uint64_t submit(){std::lock_guard<std::mutex>l(m_);if(pending_)throw std::runtime_error("CPU job already pending");++gen_;pending_=true;done_=false;cv_.notify_all();return gen_;}
    std::pair<clock_type::time_point,clock_type::time_point> wait(uint64_t g){std::unique_lock<std::mutex>l(m_);cv_.wait(l,[&]{return stop_||(done_&&done_gen_==g);});if(stop_)throw std::runtime_error("CPU worker failed");return{start_,end_};}
private:
    void loop(){
        if(affinity_cpu_>=0&&affinity_cpu_<64){DWORD_PTR mask=static_cast<DWORD_PTR>(1ull<<affinity_cpu_);(void)SetThreadAffinityMask(GetCurrentThread(),mask);}
        for(;;){uint64_t g=0;{std::unique_lock<std::mutex>l(m_);cv_.wait(l,[&]{return stop_||pending_;});if(stop_)return;g=gen_;pending_=false;}
            auto s=clock_type::now();auto status=ggml_backend_graph_compute(cpu_,graph_->graph);auto e=clock_type::now();
            if(status!=GGML_STATUS_SUCCESS){std::lock_guard<std::mutex>l(m_);stop_=true;cv_.notify_all();return;}
            {std::lock_guard<std::mutex>l(m_);start_=s;end_=e;done_=true;done_gen_=g;}cv_.notify_all();}
    }
    ggml_backend_t cpu_;RoutedGraph *graph_;int affinity_cpu_;std::thread thread_;std::mutex m_;std::condition_variable cv_;bool stop_=false,pending_=false,done_=false;uint64_t gen_=0,done_gen_=0;clock_type::time_point start_{},end_{};
};

struct Options{fs::path package,output;int repeats=21,warmups=6,cpu_threads=12,worker_cpu=-1;};
Options parse(int argc,char**argv){Options o;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing value");std::string k=argv[i],v=argv[i+1];if(k=="--package")o.package=fs::absolute(v);else if(k=="--output-file")o.output=fs::absolute(v);else if(k=="--repeats")o.repeats=std::stoi(v);else if(k=="--warmups")o.warmups=std::stoi(v);else if(k=="--cpu-threads")o.cpu_threads=std::stoi(v);else if(k=="--worker-cpu")o.worker_cpu=std::stoi(v);else throw std::runtime_error("unknown arg "+k);}if(o.package.empty()||o.output.empty()||o.repeats<5||o.cpu_threads<1||o.cpu_threads>12||fs::exists(o.output))throw std::runtime_error("invalid request");return o;}

} // namespace

int main(int argc,char**argv) try {
    const Options o=parse(argc,argv);ggml_backend_load_all();BackendPtr cpu(ggml_backend_cpu_init());BackendPtr cuda(ggml_backend_cuda_init(0));if(!cpu||!cuda)throw std::runtime_error("backend init failed");ggml_backend_cpu_set_n_threads(cpu.get(),o.cpu_threads);
    auto manifest_bytes=read_bytes(o.package/"m03-work-unit-package.json");json pkg=json::parse(manifest_bytes.begin(),manifest_bytes.end());
    const float clamp=pkg.at("routed_clamp").get<float>();
    auto input_bytes=read_bytes(o.package/"m03-ffn-input-f32le.bin");auto expert_bytes=read_bytes(o.package/"m03-selected-experts.bin");if(input_bytes.size()!=k_embd*sizeof(float)||expert_bytes.size()!=k_top_k*k_slot_bytes)throw std::runtime_error("package geometry mismatch");
    std::array<float,k_embd> input{};std::memcpy(input.data(),input_bytes.data(),input_bytes.size());
    std::array<float,k_top_k> route_weights{};for(int i=0;i<k_top_k;++i)route_weights[i]=pkg.at("work_unit").at("expert_weights").at(i).get<float>();

    ExpertArena cpu_arena=ExpertArena::create(ggml_backend_cpu_buffer_type());std::memcpy(cpu_arena.base,expert_bytes.data(),expert_bytes.size());
    ExpertArena gpu_arena=ExpertArena::create(ggml_backend_get_default_buffer_type(cuda.get()));
    BufferPtr pinned(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(),expert_bytes.size()));if(!pinned)throw std::runtime_error("pinned source allocation failed");auto*pinned_base=static_cast<uint8_t*>(ggml_backend_buffer_get_base(pinned.get()));std::memcpy(pinned_base,expert_bytes.data(),expert_bytes.size());CopyEngine copy(cuda.get());copy.copy_slots(gpu_arena,pinned_base,{0,1,2,3,4,5});

    std::vector<int32_t> all_ids={0,1,2,3,4,5};std::vector<float> all_weights(route_weights.begin(),route_weights.end());
    auto cpu_all=RoutedGraph::build(cpu.get(),cpu_arena,all_ids,all_weights,clamp);auto gpu_all=RoutedGraph::build(cuda.get(),gpu_arena,all_ids,all_weights,clamp);ggml_backend_tensor_set(cpu_all.input,input.data(),0,sizeof(input));ggml_backend_tensor_set(gpu_all.input,input.data(),0,sizeof(input));
    if(ggml_backend_graph_compute(cpu.get(),cpu_all.graph)!=GGML_STATUS_SUCCESS||ggml_backend_graph_compute(cuda.get(),gpu_all.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("endpoint correctness compute failed");ggml_backend_synchronize(cuda.get());auto ref_cpu=read_output(cpu_all.output),ref_gpu=read_output(gpu_all.output);

    json cells=json::array();
    for(int cpu_count=0;cpu_count<=6;++cpu_count){const int gpu_count=6-cpu_count;std::vector<int32_t>cpu_ids,gpu_ids;std::vector<float>cpu_w,gpu_w;for(int i=0;i<6;++i){if(i<cpu_count){cpu_ids.push_back(i);cpu_w.push_back(route_weights[i]);}else{gpu_ids.push_back(i);gpu_w.push_back(route_weights[i]);}}
        std::unique_ptr<RoutedGraph>cg,gg;if(cpu_count)cg=std::make_unique<RoutedGraph>(RoutedGraph::build(cpu.get(),cpu_arena,cpu_ids,cpu_w,clamp));if(gpu_count)gg=std::make_unique<RoutedGraph>(RoutedGraph::build(cuda.get(),gpu_arena,gpu_ids,gpu_w,clamp));if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
        std::unique_ptr<CpuWorker>worker;if(cg)worker=std::make_unique<CpuWorker>(cpu.get(),cg.get(),o.worker_cpu);
        for(int w=0;w<o.warmups;++w){
            if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
            if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
            if(worker){auto g=worker->submit();if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU warmup failed");ggml_backend_synchronize(cuda.get());}worker->wait(g);}else if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU warmup failed");ggml_backend_synchronize(cuda.get());}}
        std::vector<double>wall,cpu_ms,gpu_ms;for(int r=0;r<o.repeats;++r){
            if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
            if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
            auto ws=clock_type::now();uint64_t gen=0;if(worker)gen=worker->submit();auto gs=clock_type::now();if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU compute failed");ggml_backend_synchronize(cuda.get());}auto ge=clock_type::now();std::pair<clock_type::time_point,clock_type::time_point>ci={ws,ws};if(worker)ci=worker->wait(gen);auto we=std::max(ge,ci.second);wall.push_back(ms(we-ws));cpu_ms.push_back(worker?ms(ci.second-ci.first):0.0);gpu_ms.push_back(gg?ms(ge-gs):0.0);}
        // One clean replay after the timed loop for correctness, matching the
        // frozen M03 convention of reloading graph inputs before every run.
        if(cg){ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));if(ggml_backend_graph_compute(cpu.get(),cg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU correctness replay failed");}
        if(gg){ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU correctness replay failed");ggml_backend_synchronize(cuda.get());}
        std::array<float,k_embd>mixed{};if(cg){auto v=read_output(cg->output);for(size_t i=0;i<mixed.size();++i)mixed[i]+=v[i];}if(gg){auto v=read_output(gg->output);for(size_t i=0;i<mixed.size();++i)mixed[i]+=v[i];}
        cells.push_back({{"cpu_l2_experts",cpu_count},{"gpu_l1_experts",gpu_count},{"h2d_bytes_timed",0},{"critical_wall_ms_median",median(wall)},{"cpu_branch_ms_median",median(cpu_ms)},{"gpu_branch_ms_median",median(gpu_ms)},{"mixed_vs_current_cpu_endpoint_max_abs",max_abs(mixed,ref_cpu)},{"mixed_vs_current_gpu_endpoint_max_abs",max_abs(mixed,ref_gpu)}});
    }

    // L2-hit -> L1/R promotion economics.  For each route composition, the
    // first n_l2 selected experts start as RAM/L2 residents and the remaining
    // experts are already K/L1 hits.  promote_count of those L2 residents are
    // copied through a bounded pinned stage and executed on GPU; the rest stay
    // on CPU.  This isolates the actual promotion decision without storage I/O.
    json promotion_cells=json::array();
    for(int n_l2=1;n_l2<=6;++n_l2){
        for(int promote_count=0;promote_count<=n_l2;++promote_count){
            std::vector<int> promote_slots;
            std::vector<int32_t> cpu_ids,gpu_ids;
            std::vector<float> cpu_w,gpu_w;
            for(int i=0;i<6;++i){
                if(i<n_l2){
                    if(i<promote_count){promote_slots.push_back(i);gpu_ids.push_back(i);gpu_w.push_back(route_weights[i]);}
                    else{cpu_ids.push_back(i);cpu_w.push_back(route_weights[i]);}
                }else{gpu_ids.push_back(i);gpu_w.push_back(route_weights[i]);}
            }
            std::unique_ptr<RoutedGraph>cg,gg;
            if(!cpu_ids.empty())cg=std::make_unique<RoutedGraph>(RoutedGraph::build(cpu.get(),cpu_arena,cpu_ids,cpu_w,clamp));
            if(!gpu_ids.empty())gg=std::make_unique<RoutedGraph>(RoutedGraph::build(cuda.get(),gpu_arena,gpu_ids,gpu_w,clamp));
            std::unique_ptr<CpuWorker>worker;if(cg)worker=std::make_unique<CpuWorker>(cpu.get(),cg.get(),o.worker_cpu);

            // Warm both paths and the private copy stream.  Source bytes in
            // expert_bytes model ordinary pageable L2 ownership; pinned_base
            // is the bounded P transport window.
            for(int w=0;w<o.warmups;++w){
                if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                if(!promote_slots.empty()){
                    (void)copy.stage_and_submit_slots(gpu_arena,expert_bytes.data(),pinned_base,promote_slots);
                    copy.main_wait();
                }
                uint64_t gen=0;if(worker)gen=worker->submit();
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("promotion GPU warmup failed");ggml_backend_synchronize(cuda.get());}
                else if(!promote_slots.empty())copy.wait_done();
                if(worker)worker->wait(gen);
            }

            std::vector<double>solo_gpu_path,solo_memcpy,solo_enqueue;
            std::vector<double>critical,cpu_branch,gpu_path,concurrent_memcpy,concurrent_enqueue;
            for(int r=0;r<o.repeats;++r){
                // GPU promotion path alone: source->P memcpy + H2D + GPU work.
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                const auto s0=clock_type::now();CopyEngine::StageStats ss;
                if(!promote_slots.empty()){ss=copy.stage_and_submit_slots(gpu_arena,expert_bytes.data(),pinned_base,promote_slots);copy.main_wait();}
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("promotion solo GPU compute failed");ggml_backend_synchronize(cuda.get());}
                else if(!promote_slots.empty())copy.wait_done();
                const auto s1=clock_type::now();solo_gpu_path.push_back(ms(s1-s0));solo_memcpy.push_back(ss.memcpy_ms);solo_enqueue.push_back(ss.enqueue_ms);

                // Same path with the CPU-resident routed subset executing in
                // parallel.  The private H2D stream is ordered into the main
                // CUDA stream by one event, while CPU compute runs independently.
                if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                const auto w0=clock_type::now();uint64_t gen=0;if(worker)gen=worker->submit();
                CopyEngine::StageStats cs;
                if(!promote_slots.empty()){cs=copy.stage_and_submit_slots(gpu_arena,expert_bytes.data(),pinned_base,promote_slots);copy.main_wait();}
                const auto g0=clock_type::now();
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("promotion concurrent GPU compute failed");ggml_backend_synchronize(cuda.get());}
                else if(!promote_slots.empty())copy.wait_done();
                const auto g1=clock_type::now();
                std::pair<clock_type::time_point,clock_type::time_point>ci={w0,w0};if(worker)ci=worker->wait(gen);
                const auto w1=std::max(g1,ci.second);
                critical.push_back(ms(w1-w0));cpu_branch.push_back(worker?ms(ci.second-ci.first):0.0);gpu_path.push_back(ms(g1-w0));concurrent_memcpy.push_back(cs.memcpy_ms);concurrent_enqueue.push_back(cs.enqueue_ms);
            }

            // Clean correctness replay after the timed samples.
            if(!promote_slots.empty()){(void)copy.stage_and_submit_slots(gpu_arena,expert_bytes.data(),pinned_base,promote_slots);copy.main_wait();}
            if(cg){ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));if(ggml_backend_graph_compute(cpu.get(),cg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("promotion CPU replay failed");}
            if(gg){ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("promotion GPU replay failed");ggml_backend_synchronize(cuda.get());}
            else if(!promote_slots.empty())copy.wait_done();
            std::array<float,k_embd>mixed{};if(cg){auto v=read_output(cg->output);for(size_t i=0;i<mixed.size();++i)mixed[i]+=v[i];}if(gg){auto v=read_output(gg->output);for(size_t i=0;i<mixed.size();++i)mixed[i]+=v[i];}

            promotion_cells.push_back({
                {"selected_l2_hits",n_l2},
                {"selected_existing_l1_hits",6-n_l2},
                {"promoted_l2_to_gpu",promote_count},
                {"kept_l2_cpu",n_l2-promote_count},
                {"gpu_compute_experts",6-n_l2+promote_count},
                {"timed_h2d_bytes",static_cast<uint64_t>(promote_count)*(k_gate_bytes+k_up_bytes+k_down_bytes)},
                {"gpu_path_solo_ms_median",median(solo_gpu_path)},
                {"source_to_p_memcpy_solo_ms_median",median(solo_memcpy)},
                {"h2d_enqueue_solo_ms_median",median(solo_enqueue)},
                {"critical_wall_concurrent_ms_median",median(critical)},
                {"cpu_branch_concurrent_ms_median",median(cpu_branch)},
                {"gpu_path_concurrent_ms_median",median(gpu_path)},
                {"source_to_p_memcpy_concurrent_ms_median",median(concurrent_memcpy)},
                {"h2d_enqueue_concurrent_ms_median",median(concurrent_enqueue)},
                {"gpu_path_slowdown_ratio",median(solo_gpu_path)>0.0?median(gpu_path)/median(solo_gpu_path):1.0},
                {"mixed_vs_current_cpu_endpoint_max_abs",max_abs(mixed,ref_cpu)},
                {"mixed_vs_current_gpu_endpoint_max_abs",max_abs(mixed,ref_gpu)}
            });
        }
    }
    // Direct-registered upper-bound sweep.  This is the same post-router split
    // as above, except the selected L2-resident source is assumed to be already
    // CUDA-registered/pinned.  Therefore there is no L2->P memcpy: H2D reads
    // directly from pinned_base into the transient GPU slots.  This models an
    // R-style same-use execution without claiming persistent K admission.
    json direct_registered_cells=json::array();
    for(int n_l2=1;n_l2<=6;++n_l2){
        for(int move_count=0;move_count<=n_l2;++move_count){
            std::vector<int> move_slots;
            std::vector<int32_t> cpu_ids,gpu_ids;
            std::vector<float> cpu_w,gpu_w;
            for(int i=0;i<6;++i){
                if(i<n_l2){
                    if(i<move_count){move_slots.push_back(i);gpu_ids.push_back(i);gpu_w.push_back(route_weights[i]);}
                    else{cpu_ids.push_back(i);cpu_w.push_back(route_weights[i]);}
                }else{gpu_ids.push_back(i);gpu_w.push_back(route_weights[i]);}
            }
            std::unique_ptr<RoutedGraph>cg,gg;
            if(!cpu_ids.empty())cg=std::make_unique<RoutedGraph>(RoutedGraph::build(cpu.get(),cpu_arena,cpu_ids,cpu_w,clamp));
            if(!gpu_ids.empty())gg=std::make_unique<RoutedGraph>(RoutedGraph::build(cuda.get(),gpu_arena,gpu_ids,gpu_w,clamp));
            std::unique_ptr<CpuWorker>worker;if(cg)worker=std::make_unique<CpuWorker>(cpu.get(),cg.get(),o.worker_cpu);
            for(int w=0;w<o.warmups;++w){
                if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                if(!move_slots.empty()){copy.submit_slots(gpu_arena,pinned_base,move_slots);copy.main_wait();}
                uint64_t gen=0;if(worker)gen=worker->submit();
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("direct-register warmup GPU compute failed");ggml_backend_synchronize(cuda.get());}
                else if(!move_slots.empty())copy.wait_done();
                if(worker)worker->wait(gen);
            }
            std::vector<double>critical,cpu_branch,gpu_path,solo_gpu_path;
            for(int r=0;r<o.repeats;++r){
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                const auto s0=clock_type::now();
                if(!move_slots.empty()){copy.submit_slots(gpu_arena,pinned_base,move_slots);copy.main_wait();}
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("direct-register solo GPU compute failed");ggml_backend_synchronize(cuda.get());}
                else if(!move_slots.empty())copy.wait_done();
                solo_gpu_path.push_back(ms(clock_type::now()-s0));

                if(cg)ggml_backend_tensor_set(cg->input,input.data(),0,sizeof(input));
                if(gg)ggml_backend_tensor_set(gg->input,input.data(),0,sizeof(input));
                const auto w0=clock_type::now();uint64_t gen=0;if(worker)gen=worker->submit();
                if(!move_slots.empty()){copy.submit_slots(gpu_arena,pinned_base,move_slots);copy.main_wait();}
                const auto g0=clock_type::now();
                if(gg){if(ggml_backend_graph_compute(cuda.get(),gg->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("direct-register concurrent GPU compute failed");ggml_backend_synchronize(cuda.get());}
                else if(!move_slots.empty())copy.wait_done();
                const auto g1=clock_type::now();
                std::pair<clock_type::time_point,clock_type::time_point>ci={w0,w0};if(worker)ci=worker->wait(gen);
                const auto w1=std::max(g1,ci.second);
                critical.push_back(ms(w1-w0));cpu_branch.push_back(worker?ms(ci.second-ci.first):0.0);gpu_path.push_back(ms(g1-w0));
            }
            direct_registered_cells.push_back({
                {"selected_l2_hits",n_l2},
                {"selected_existing_l1_hits",6-n_l2},
                {"moved_l2_to_transient_gpu",move_count},
                {"kept_l2_cpu",n_l2-move_count},
                {"gpu_compute_experts",6-n_l2+move_count},
                {"timed_h2d_bytes",static_cast<uint64_t>(move_count)*(k_gate_bytes+k_up_bytes+k_down_bytes)},
                {"source_to_p_memcpy_ms",0.0},
                {"gpu_path_solo_ms_median",median(solo_gpu_path)},
                {"critical_wall_concurrent_ms_median",median(critical)},
                {"cpu_branch_concurrent_ms_median",median(cpu_branch)},
                {"gpu_path_concurrent_ms_median",median(gpu_path)}
            });
        }
    }

    json result={{"schema","siliang-v013-ds4-residency-and-promotion-v2"},{"complete",true},{"base_checkpoint","738c1804a88e9f99742714d7fbc354ecf7e0b279"},{"cpu_threads",o.cpu_threads},{"worker_logical_cpu",o.worker_cpu},{"repeats",o.repeats},{"warmups",o.warmups},{"current_cpu_vs_gpu_endpoint_max_abs",max_abs(ref_cpu,ref_gpu)},{"all_hit_memory_only",cells},{"l2_to_gpu_promotion",promotion_cells},{"direct_registered_transient_gpu",direct_registered_cells}};
    std::ofstream f(o.output,std::ios::binary);if(!f)throw std::runtime_error("cannot create output");f<<result.dump(2)<<"\n";std::fprintf(stderr,"ds4-routed-residency-assay: complete: %s\n",o.output.string().c_str());return 0;
} catch(const std::exception&e){std::fprintf(stderr,"ds4-routed-residency-assay: fatal: %s\n",e.what());return 2;}
