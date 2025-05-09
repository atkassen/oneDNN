#===============================================================================
# Copyright 2016-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if (NOT DNNL_BUILD_TESTS)
    return()
endif()

include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/third_party)

# propagate TEST specific flags
append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")

set(CMAKE_TEST_CCXX_NOWARN_FLAGS)

# propagate no warning flags
append(CMAKE_TEST_CCXX_NOWARN_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# allow tests to include internal header files with, e.g.
# include "src/common/dnnl_thread.hpp" ...
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR})
# ... and allow the included files transitively include other headers files
# e.g. include "common/dnnl_thread_parallel_nd.hpp" from "dnnl_thread.hpp"
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/src)

if(WIN32 AND NOT MINGW)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # 2415: unused variable
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/Qdiag-disable:2415")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # c4244: conversion with possible loss of data
        # c4996: unsafe / deprecated functions
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/wd4996 /wd4244")
        if ("${DNNL_CPU_THREADING_RUNTIME}" STREQUAL "THREADPOOL" AND Eigen3_FOUND)
            # c4267: conversion with possible loss of data
            append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/wd4267")
        endif()
    endif()
endif()

append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE" AND
   NOT DNNL_ENABLE_STACK_CHECKER AND
   # TBB threading runtime requires explicit finalization for tests used
   # under CTest. Since TBB doesn't provide C API to perform finalization
   # we have to skip C tests.
   NOT DNNL_CPU_RUNTIME STREQUAL "TBB" AND
   NOT DNNL_CPU_SYCL)
    find_libm(LIBM)
    register_exe(api-c api.c "test" "${LIBM}")
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang" AND (UNIX OR MINGW))
    get_directory_property(include_dirs INCLUDE_DIRECTORIES)
    set(test_c_symbols "${CMAKE_CURRENT_BINARY_DIR}/test_c_symbols.c")
    add_custom_command(
        OUTPUT ${test_c_symbols}
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_c_symbols_refs.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/.. ${test_c_symbols} ${include_dirs}
    )
    register_exe(test_c_symbols-c ${test_c_symbols} "test")
# elseif(WIN32)
# No Windows support for: test_c_symbols.c
endif()

add_definitions_with_host_compiler(-DNOMINMAX) # to allow std::max on Windows with parentheses
add_definitions_with_host_compiler(-DWIN32_LEAN_AND_MEAN) # to speedup up Windows compilation

if(NOT DNNL_ENABLE_PRIMITIVE_CACHE)
    add_definitions_with_host_compiler(-DDNNL_DISABLE_PRIMITIVE_CACHE)
endif()

set(TEST_THREAD ${CMAKE_CURRENT_SOURCE_DIR}/test_thread.cpp)

# Switch on threading layer for GPU only configurations to speed up testing.
# For non-dppcp compilers OpenMP threading will be used which is handled in
# OpenMP.cmake.
if(DNNL_WITH_SYCL AND DNNL_CPU_RUNTIME STREQUAL "NONE")
    list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
    find_package_tbb(REQUIRED)
    list(REMOVE_AT CMAKE_MODULE_PATH 0)
    handle_tbb_target()
endif()

add_subdirectory(gtests)
add_subdirectory(benchdnn)

if(NOT DNNL_WITH_SYCL AND NOT DNNL_ENABLE_STACK_CHECKER)
    if(UNIX OR MINGW)
        add_subdirectory(noexcept)
    endif()
endif()
