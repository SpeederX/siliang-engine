if (NOT WIN32)
    message(FATAL_ERROR "SILIANG_DS4_HYBRID_RESEARCH currently requires Windows")
endif()

if (NOT TARGET ggml-cuda)
    message(FATAL_ERROR "SILIANG_DS4_HYBRID_RESEARCH requires GGML_CUDA=ON")
endif()

add_executable(ds4-shared-cpu-overlap-assay
    shared_overlap_assay.cpp
)

target_include_directories(ds4-shared-cpu-overlap-assay PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor
)

target_compile_features(ds4-shared-cpu-overlap-assay PRIVATE cxx_std_17)
target_link_libraries(ds4-shared-cpu-overlap-assay PRIVATE llama ggml-cpu ggml-cuda)
