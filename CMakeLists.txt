# =============================================================================
# Quantum Geometric Tensor Library - CMake Build System
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(quantum_geometric_tensor VERSION 1.0.0 LANGUAGES C)

# C Standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(QGT_BUILD_TESTS "Build test executables" ON)
option(QGT_BUILD_EXAMPLES "Build example executables" ON)
option(QGT_ENABLE_CUDA "Enable CUDA support" OFF)
option(QGT_BUILD_SHARED "Build shared library" ON)
option(QGT_BUILD_STATIC "Build static library" ON)

# Auto-detect MPI if available
find_package(MPI QUIET)
if(MPI_FOUND)
    option(QGT_ENABLE_MPI "Enable MPI support" ON)
else()
    option(QGT_ENABLE_MPI "Enable MPI support" OFF)
endif()

# =============================================================================
# Platform Detection and Configuration
# =============================================================================
# UNIFIED NAMING: All platform macros use QGT_ prefix

if(APPLE)
    set(QGT_PLATFORM_MACOS ON)
    add_definitions(-DQGT_PLATFORM_MACOS=1)

    # Apple Accelerate framework
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

    # Auto-detect Metal support
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    if(METAL_FRAMEWORK AND METALPERFORMANCESHADERS_FRAMEWORK)
        option(QGT_ENABLE_METAL "Enable Metal support (macOS only)" ON)
    else()
        option(QGT_ENABLE_METAL "Enable Metal support (macOS only)" OFF)
    endif()

    # Check for ARM (Apple Silicon)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(QGT_ARCH_ARM64 ON)
        add_definitions(-DQGT_ARCH_ARM64=1 -DQGT_SIMD_NEON=1)
    else()
        set(QGT_ARCH_X86_64 ON)
        add_definitions(-DQGT_ARCH_X86_64=1)
    endif()
elseif(UNIX AND NOT APPLE)
    set(QGT_PLATFORM_LINUX ON)
    add_definitions(-DQGT_PLATFORM_LINUX=1)
    option(QGT_ENABLE_METAL "Enable Metal support (macOS only)" OFF)

    # Look for OpenBLAS or Intel MKL
    find_package(BLAS)
    find_package(LAPACK)

    # Architecture detection with proper SIMD flags
    include(CheckCCompilerFlag)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(QGT_ARCH_X86_64 ON)
        add_definitions(-DQGT_ARCH_X86_64=1)

        # Check for SIMD support - check all levels
        check_c_compiler_flag("-mavx512f" QGT_HAS_AVX512F)
        check_c_compiler_flag("-mavx2" QGT_HAS_AVX2)
        check_c_compiler_flag("-mavx" QGT_HAS_AVX)
        check_c_compiler_flag("-mfma" QGT_HAS_FMA)

        if(QGT_HAS_AVX512F)
            add_definitions(-DQGT_SIMD_AVX512=1)
            add_compile_options(-mavx512f -mavx512vl -mavx512dq)
        elseif(QGT_HAS_AVX2)
            add_definitions(-DQGT_SIMD_AVX2=1)
            add_compile_options(-mavx2)
        elseif(QGT_HAS_AVX)
            add_definitions(-DQGT_SIMD_AVX=1)
            add_compile_options(-mavx)
        endif()

        if(QGT_HAS_FMA)
            add_definitions(-DQGT_SIMD_FMA=1)
            add_compile_options(-mfma)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(QGT_ARCH_ARM64 ON)
        add_definitions(-DQGT_ARCH_ARM64=1 -DQGT_SIMD_NEON=1)
    endif()
elseif(WIN32)
    set(QGT_PLATFORM_WINDOWS ON)
    add_definitions(-DQGT_PLATFORM_WINDOWS=1)
    option(QGT_ENABLE_METAL "Enable Metal support (macOS only)" OFF)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        set(QGT_ARCH_X86_64 ON)
        add_definitions(-DQGT_ARCH_X86_64=1)
        if(MSVC)
            add_compile_options(/arch:AVX2)
            add_definitions(-DQGT_SIMD_AVX2=1)
        endif()
    endif()
endif()

# pthread support
find_package(Threads REQUIRED)

# MPI support configuration
if(QGT_ENABLE_MPI)
    if(NOT MPI_FOUND)
        find_package(MPI REQUIRED)
    endif()
    add_definitions(-DQGT_HAS_MPI -DHAS_MPI)
else()
    add_definitions(-DNO_MPI)
endif()

# hwloc support configuration (optional)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(HWLOC QUIET hwloc)
endif()
if(NOT HWLOC_FOUND)
    # Try Homebrew paths on macOS (arm64 and x86_64)
    find_path(HWLOC_INCLUDE_DIRS hwloc.h
        PATHS /opt/homebrew/include /usr/local/include /usr/include)
    find_library(HWLOC_LIBRARIES hwloc
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
    if(HWLOC_INCLUDE_DIRS AND HWLOC_LIBRARIES)
        set(HWLOC_FOUND TRUE)
    endif()
endif()
if(HWLOC_FOUND)
    add_definitions(-DQGT_HAS_HWLOC)
    include_directories(${HWLOC_INCLUDE_DIRS})
    link_directories(${HWLOC_LIBRARY_DIRS})
    message(STATUS "hwloc found: ${HWLOC_LIBRARIES} (dirs: ${HWLOC_LIBRARY_DIRS})")
else()
    add_definitions(-DNO_HWLOC)
    message(STATUS "hwloc not found - workload balancer will use basic topology detection")
endif()

# zlib support (for dataset compression)
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    message(STATUS "zlib found: ${ZLIB_LIBRARIES}")
else()
    message(STATUS "zlib not found - some dataset loading may be limited")
endif()

# json-c support configuration (for D-Wave backend)
if(PkgConfig_FOUND)
    pkg_check_modules(JSON_C QUIET json-c)
endif()
if(NOT JSON_C_FOUND)
    # Try Homebrew path on macOS
    find_path(JSON_C_INCLUDE_DIRS json-c/json.h
        PATHS /opt/homebrew/include /usr/local/include /usr/include)
    find_library(JSON_C_LIBRARIES json-c
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
    if(JSON_C_INCLUDE_DIRS AND JSON_C_LIBRARIES)
        set(JSON_C_FOUND TRUE)
    endif()
endif()
if(JSON_C_FOUND)
    add_definitions(-DQGT_HAS_JSON_C)
    include_directories(${JSON_C_INCLUDE_DIRS})
    link_directories(${JSON_C_LIBRARY_DIRS})
    message(STATUS "json-c found: ${JSON_C_LIBRARIES} (dirs: ${JSON_C_LIBRARY_DIRS})")
else()
    add_definitions(-DNO_JSON_C)
    message(STATUS "json-c not found - D-Wave backend will use stub implementation")
endif()

# libcurl support configuration (for cloud backends)
find_package(CURL QUIET)
if(NOT CURL_FOUND AND PkgConfig_FOUND)
    pkg_check_modules(CURL QUIET libcurl)
endif()
if(CURL_FOUND)
    add_definitions(-DQGT_HAS_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    message(STATUS "libcurl found: ${CURL_LIBRARIES}")
else()
    add_definitions(-DNO_CURL)
    message(STATUS "libcurl not found - cloud backends will use stub implementation")
endif()

# Metal support configuration
if(QGT_ENABLE_METAL)
    add_definitions(-DQGT_HAS_METAL=1)
endif()

# CUDA support configuration
if(QGT_ENABLE_CUDA)
    add_definitions(-DQGT_HAS_CUDA=1)
else()
    add_definitions(-DNO_CUDA)
endif()

# =============================================================================
# Compiler Flags
# =============================================================================
# Portable build option (for distribution)
option(QGT_PORTABLE_BUILD "Build without -march=native for portable binaries" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-function)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -ffast-math)
        if(NOT QGT_PORTABLE_BUILD)
            if(QGT_ARCH_ARM64)
                add_compile_options(-mcpu=native)
            elseif(QGT_ARCH_X86_64)
                add_compile_options(-march=native)
            endif()
        endif()
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -DDEBUG)
    endif()
endif()

# =============================================================================
# Include Directories
# =============================================================================
set(QGT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${QGT_INCLUDE_DIR})

# =============================================================================
# Source Files Collection
# =============================================================================

# Core module sources
file(GLOB QGT_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantum_geometric/core/*.c
)

# Distributed module sources
file(GLOB QGT_DISTRIBUTED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantum_geometric/distributed/*.c
)
# Exclude files with incomplete implementations (require MPI/hwloc at compile time or missing headers)
list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "differential_geometry\\.c$")
list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "quantum_distributed_operations\\.c$")
list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "quantum_distributed_system\\.c$")
list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "distributed_training\\.c$")
list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "pipeline_parallel\\.c$")

# Conditionally exclude files that require hwloc
if(NOT HWLOC_FOUND)
    list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "workload_balancer\\.c$")
    list(FILTER QGT_DISTRIBUTED_SOURCES EXCLUDE REGEX "communication_optimizer\\.c$")
endif()

# Hardware module sources
file(GLOB QGT_HARDWARE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/hardware/*.c
)

# Conditionally exclude files that require json-c
if(NOT JSON_C_FOUND)
    list(FILTER QGT_HARDWARE_SOURCES EXCLUDE REGEX "quantum_dwave_backend\\.c$")
    list(FILTER QGT_HARDWARE_SOURCES EXCLUDE REGEX "quantum_dwave_backend_optimized\\.c$")
    list(FILTER QGT_HARDWARE_SOURCES EXCLUDE REGEX "quantum_rigetti_backend\\.c$")
    list(FILTER QGT_HARDWARE_SOURCES EXCLUDE REGEX "quantum_rigetti_backend_optimized\\.c$")
endif()

# Hybrid module sources
file(GLOB QGT_HYBRID_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/hybrid/*.c
)

# Learning module sources
file(GLOB QGT_LEARNING_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/learning/*.c
)

# Physics module sources
file(GLOB QGT_PHYSICS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/physics/*.c
)

# AI module sources
file(GLOB QGT_AI_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/ai/*.c
)

# Monitoring module sources
file(GLOB QGT_MONITORING_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/monitoring/*.c
)

# Config module sources
file(GLOB QGT_CONFIG_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/config/*.c
)

# Supercomputer module sources
file(GLOB QGT_SUPERCOMPUTER_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/supercomputer/*.c
)

# Combine all sources
set(QGT_ALL_SOURCES
    ${QGT_CORE_SOURCES}
    ${QGT_DISTRIBUTED_SOURCES}
    ${QGT_HARDWARE_SOURCES}
    ${QGT_HYBRID_SOURCES}
    ${QGT_LEARNING_SOURCES}
    ${QGT_PHYSICS_SOURCES}
    ${QGT_AI_SOURCES}
    ${QGT_MONITORING_SOURCES}
    ${QGT_CONFIG_SOURCES}
    ${QGT_SUPERCOMPUTER_SOURCES}
)

# Platform-specific exclusions
if(NOT APPLE)
    # Remove macOS-specific files on non-Apple platforms
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*_macos\\.c$")
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*accelerate.*\\.c$")
else()
    # On Apple: remove generic versions when platform-specific exists
    # memory_optimization.c has a _macos.c variant
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*/memory_optimization\\.c$")
endif()

if(NOT UNIX OR APPLE)
    # Remove Linux-specific files on non-Linux platforms
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*_linux\\.c$")
endif()

if(UNIX AND NOT APPLE)
    # On Linux: remove generic versions when platform-specific exists
    # memory_optimization.c has a _linux.c variant
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*/memory_optimization\\.c$")
endif()

# Remove CUDA/Metal sources if not enabled
if(NOT QGT_ENABLE_CUDA)
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*_cuda\\.c$")
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*_gpu\\.c$")
endif()

if(NOT QGT_ENABLE_METAL)
    list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*_metal\\.c$")
endif()

# Remove assembly files for now (handled separately)
list(FILTER QGT_ALL_SOURCES EXCLUDE REGEX ".*\\.s$")

# =============================================================================
# Library Targets
# =============================================================================

# Static library
if(QGT_BUILD_STATIC)
    add_library(quantum_geometric_static STATIC ${QGT_ALL_SOURCES})
    target_include_directories(quantum_geometric_static PUBLIC ${QGT_INCLUDE_DIR})
    target_link_libraries(quantum_geometric_static PUBLIC Threads::Threads m)

    if(APPLE)
        target_link_libraries(quantum_geometric_static PUBLIC ${ACCELERATE_FRAMEWORK})
        # IOKit for GPU monitoring
        find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
        target_link_libraries(quantum_geometric_static PUBLIC ${IOKIT_FRAMEWORK})
        if(QGT_ENABLE_METAL)
            target_link_libraries(quantum_geometric_static PUBLIC ${METAL_FRAMEWORK} ${METALPERFORMANCESHADERS_FRAMEWORK})
            find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
            target_link_libraries(quantum_geometric_static PUBLIC ${FOUNDATION_FRAMEWORK})
        endif()
    elseif(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()

    if(QGT_ENABLE_MPI)
        target_link_libraries(quantum_geometric_static PUBLIC MPI::MPI_C)
    endif()

    if(HWLOC_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${HWLOC_LIBRARIES})
    endif()

    if(JSON_C_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${JSON_C_LIBRARIES})
    endif()

    if(CURL_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${CURL_LIBRARIES})
    endif()

    if(ZLIB_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${ZLIB_LIBRARIES})
    endif()

    set_target_properties(quantum_geometric_static PROPERTIES
        OUTPUT_NAME quantum_geometric
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# Shared library
if(QGT_BUILD_SHARED)
    add_library(quantum_geometric_shared SHARED ${QGT_ALL_SOURCES})
    target_include_directories(quantum_geometric_shared PUBLIC ${QGT_INCLUDE_DIR})
    target_link_libraries(quantum_geometric_shared PUBLIC Threads::Threads m)

    if(APPLE)
        target_link_libraries(quantum_geometric_shared PUBLIC ${ACCELERATE_FRAMEWORK})
        # IOKit for GPU monitoring
        target_link_libraries(quantum_geometric_shared PUBLIC ${IOKIT_FRAMEWORK})
        if(QGT_ENABLE_METAL)
            target_link_libraries(quantum_geometric_shared PUBLIC ${METAL_FRAMEWORK} ${METALPERFORMANCESHADERS_FRAMEWORK})
            find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
            target_link_libraries(quantum_geometric_shared PUBLIC ${FOUNDATION_FRAMEWORK})
        endif()
    elseif(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()

    if(QGT_ENABLE_MPI)
        target_link_libraries(quantum_geometric_shared PUBLIC MPI::MPI_C)
    endif()

    if(HWLOC_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${HWLOC_LIBRARIES})
    endif()

    if(JSON_C_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${JSON_C_LIBRARIES})
    endif()

    if(CURL_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${CURL_LIBRARIES})
    endif()

    if(ZLIB_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${ZLIB_LIBRARIES})
    endif()

    set_target_properties(quantum_geometric_shared PROPERTIES
        OUTPUT_NAME quantum_geometric
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
endif()

# Alias for easier usage
if(QGT_BUILD_STATIC)
    add_library(quantum_geometric ALIAS quantum_geometric_static)
else()
    add_library(quantum_geometric ALIAS quantum_geometric_shared)
endif()

# =============================================================================
# Test Targets
# =============================================================================
if(QGT_BUILD_TESTS)
    enable_testing()

    # Collect all test sources
    file(GLOB QGT_TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/test_*.c)

    # Create test executables
    foreach(TEST_SOURCE ${QGT_TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE quantum_geometric)
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

# Install libraries
if(QGT_BUILD_STATIC)
    install(TARGETS quantum_geometric_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(QGT_BUILD_SHARED)
    install(TARGETS quantum_geometric_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install headers
install(DIRECTORY ${QGT_INCLUDE_DIR}/quantum_geometric
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== Quantum Geometric Tensor Library Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Build tests: ${QGT_BUILD_TESTS}")
message(STATUS "  Build shared lib: ${QGT_BUILD_SHARED}")
message(STATUS "  Build static lib: ${QGT_BUILD_STATIC}")
message(STATUS "  MPI support: ${QGT_ENABLE_MPI}")
message(STATUS "  CUDA support: ${QGT_ENABLE_CUDA}")
message(STATUS "  Metal support: ${QGT_ENABLE_METAL}")
message(STATUS "  hwloc support: ${HWLOC_FOUND}")
message(STATUS "")
if(APPLE)
    message(STATUS "Frameworks: Accelerate")
endif()
message(STATUS "")
message(STATUS "Source files: ${QGT_ALL_SOURCES}")
message(STATUS "")
