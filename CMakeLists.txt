cmake_minimum_required(VERSION 3.12)

project(lego-profiler)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)
message(STATUS "Using CUDA ${CUDA_VERSION}")
set(CUDA_NVCC_FLAGS -std=c++11)

add_subdirectory("3rd/rapidjson")
include("build/3rd/rapidjson/RapidJSONConfig.cmake")
list(APPEND LINK_LIBS rapidjson)
include_directories("3rd/rapidjson/include")

list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/src/lego)
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(src/lego)