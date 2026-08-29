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
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_event_record(
        ggml_backend_cuda_siliang_stream_t stream,
        ggml_backend_cuda_siliang_event_t event);
GGML_BACKEND_API enum ggml_backend_cuda_siliang_status ggml_backend_cuda_siliang_main_stream_event_record(
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

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
