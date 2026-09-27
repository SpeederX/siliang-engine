#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Minimal CUDA synchronization and H2D bridge used by the Siliang MoE arena.
// Stream and event handles are private to the CUDA backend and must not be mixed across devices.
enum ggml_backend_cuda_siliang_status {
    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS = 0,
    GGML_BACKEND_CUDA_SILIANG_STATUS_INVALID_ARGUMENT = 1,
    GGML_BACKEND_CUDA_SILIANG_STATUS_WRONG_BACKEND = 2,
    GGML_BACKEND_CUDA_SILIANG_STATUS_WRONG_BUFFER = 3,
    GGML_BACKEND_CUDA_SILIANG_STATUS_WRONG_DEVICE = 4,
    GGML_BACKEND_CUDA_SILIANG_STATUS_RANGE = 5,
    GGML_BACKEND_CUDA_SILIANG_STATUS_CUDA_ERROR = 6,
    GGML_BACKEND_CUDA_SILIANG_STATUS_NOT_READY = 7,
};

typedef void * ggml_backend_cuda_siliang_stream_t;
typedef void * ggml_backend_cuda_siliang_event_t;

GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_stream_create(
        ggml_backend_t backend,
        ggml_backend_cuda_siliang_stream_t * out_stream);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_stream_destroy(
        ggml_backend_cuda_siliang_stream_t stream);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_stream_synchronize(
        ggml_backend_cuda_siliang_stream_t stream);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_create(
        ggml_backend_t backend,
        ggml_backend_cuda_siliang_event_t * out_event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_destroy(
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_synchronize(
        ggml_backend_cuda_siliang_event_t event);
// Non-blocking: SUCCESS when all work recorded before the event has completed, NOT_READY otherwise.
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_query(
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_record(
        ggml_backend_cuda_siliang_stream_t stream,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_main_stream_event_record(
        ggml_backend_t backend,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_all_streams_event_record(
        ggml_backend_t backend,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_stream_wait_event(
        ggml_backend_cuda_siliang_stream_t stream,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_main_stream_wait_event(
        ggml_backend_t backend,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_h2d_async(
        ggml_backend_cuda_siliang_stream_t stream,
        struct ggml_tensor * tensor,
        const void * source,
        size_t offset,
        size_t size);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_d2h_async(
        ggml_backend_cuda_siliang_stream_t stream,
        struct ggml_tensor * tensor,
        void * destination,
        size_t offset,
        size_t size);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_d2d_async(
        ggml_backend_cuda_siliang_stream_t stream,
        struct ggml_tensor * tensor,
        size_t destination_offset,
        size_t source_offset,
        size_t size);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_host_register_readonly(
        ggml_backend_t backend,
        void * buffer,
        size_t size);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_host_unregister(
        ggml_backend_t backend,
        void * buffer);

// Siliang dynamic K (CUDA VMM). A compute buffer type whose last bytes are one physical allocation
// per entry of tail_bytes (rounded up to the VMM granularity); NULL without VMM support.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_siliang_tail_buffer_type(
        ggml_backend_t backend,
        const size_t * tail_bytes,
        size_t tail_count);
// The live buffer of a tail buffer type: base, total size, head bytes (the tail starts at base + head),
// and a generation that changes whenever the buffer is reallocated. NOT_READY while none is allocated.
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_tail_query(
        ggml_backend_buffer_type_t buft,
        void ** out_base,
        size_t * out_size,
        size_t * out_head,
        uint64_t * out_generation);
// A CUDA buffer assembled from consecutive VMM segments: sources[i] == 0 maps new device memory of sizes[i]
// bytes, otherwise the allocation mapped at device address sources[i] is mapped again (sizes[i] must be its size).
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_siliang_alias_buffer(
        ggml_backend_t backend,
        const uint64_t * sources,
        const size_t * sizes,
        size_t count);
// VMM mapping granularity of the backend's device, 0 without VMM support.
GGML_BACKEND_API size_t ggml_backend_cuda_siliang_vmm_granularity(ggml_backend_t backend);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_memset_async(
        ggml_backend_cuda_siliang_stream_t stream,
        struct ggml_tensor * tensor,
        size_t offset,
        size_t size,
        int value);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
