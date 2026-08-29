cmake_minimum_required(VERSION 3.20)
project(siliang_ds4_hybrid_assays LANGUAGES CXX)

set(SILIANG_SOURCE_ROOT "" CACHE PATH "Siliang source tree used for headers")
set(SILIANG_PREBUILT_ROOT "" CACHE PATH "Qualified 738c1804a CUDA build tree")
if (NOT SILIANG_SOURCE_ROOT OR NOT SILIANG_PREBUILT_ROOT)
    message(FATAL_ERROR "Set SILIANG_SOURCE_ROOT and SILIANG_PREBUILT_ROOT")
endif()

function(add_siliang_assay target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_include_directories(${target} PRIVATE
        "${SILIANG_SOURCE_ROOT}/ggml/include"
        "${SILIANG_SOURCE_ROOT}/ggml/src"
        "${SILIANG_SOURCE_ROOT}/src"
        "${SILIANG_SOURCE_ROOT}/vendor"
    )
    target_link_libraries(${target} PRIVATE
        "${SILIANG_PREBUILT_ROOT}/ggml/src/Release/ggml.lib"
        "${SILIANG_PREBUILT_ROOT}/ggml/src/Release/ggml-base.lib"
        "${SILIANG_PREBUILT_ROOT}/ggml/src/Release/ggml-cpu.lib"
        "${SILIANG_PREBUILT_ROOT}/ggml/src/ggml-cuda/Release/ggml-cuda.lib"
    )
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SILIANG_PREBUILT_ROOT}/bin/Release/ggml.dll"
            "${SILIANG_PREBUILT_ROOT}/bin/Release/ggml-base.dll"
            "${SILIANG_PREBUILT_ROOT}/bin/Release/ggml-cpu.dll"
            "${SILIANG_PREBUILT_ROOT}/bin/Release/ggml-cuda.dll"
            "$<TARGET_FILE_DIR:${target}>"
    )
endfunction()

add_siliang_assay(ds4-shared-cpu-overlap-assay ../shared_overlap_assay.cpp)
add_siliang_assay(ds4-routed-residency-assay ../routed_residency_assay.cpp)
add_siliang_assay(ds4-l2-admission-assay ../l2_admission_assay.cpp)
target_include_directories(ds4-l2-admission-assay PRIVATE "${SILIANG_SOURCE_ROOT}/include")
target_link_libraries(ds4-l2-admission-assay PRIVATE "${SILIANG_PREBUILT_ROOT}/src/Release/llama.lib")
add_custom_command(TARGET ds4-l2-admission-assay POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SILIANG_PREBUILT_ROOT}/bin/Release/llama.dll"
        "$<TARGET_FILE_DIR:ds4-l2-admission-assay>"
)
