set(PERF_SRCS "")
set(PERF_HDRS "")
list(APPEND PERF_SRCS main.cc)
list(APPEND PERF_SRCS onnx_trt.cc)
list(APPEND PERF_HDRS utils.h)
list(APPEND PERF_HDRS buffer.h)
list(APPEND PERF_HDRS onnx_trt.h)
list(APPEND PERF_HDRS onnx_trt_config.h)
list(APPEND PERF_HDRS record.h)

add_executable(
  main
  ${PERF_SRCS}
  ${PERF_HDRS}
)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/src/lego)

target_include_directories(main PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(main PRIVATE ${LINK_LIBS})
target_link_libraries(main PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(main PRIVATE ${PLUGIN_LIBS})