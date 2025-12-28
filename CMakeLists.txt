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

    # Apple Accelerate framework with new LAPACK interface to avoid deprecation warnings
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    add_definitions(-DACCELERATE_NEW_LAPACK)

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

# =============================================================================
# Quantum Format Parser Libraries (optional)
# =============================================================================

# qe-qasm support (OpenQASM parser from IBM)
# Repository: https://github.com/openqasm/qe-qasm
option(QGT_ENABLE_QEQASM "Enable qe-qasm OpenQASM parser library" OFF)
if(QGT_ENABLE_QEQASM)
    find_path(QEQASM_INCLUDE_DIRS qasm/QasmParser.h
        PATHS /opt/homebrew/include /usr/local/include /usr/include
              ${CMAKE_CURRENT_SOURCE_DIR}/external/qe-qasm/include)
    find_library(QEQASM_PARSER_LIB qasmParser
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
              ${CMAKE_CURRENT_SOURCE_DIR}/external/qe-qasm/lib)
    find_library(QEQASM_AST_LIB qasmAST
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
              ${CMAKE_CURRENT_SOURCE_DIR}/external/qe-qasm/lib)
    find_library(QEQASM_FRONTEND_LIB qasmFrontend
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
              ${CMAKE_CURRENT_SOURCE_DIR}/external/qe-qasm/lib)
    if(QEQASM_INCLUDE_DIRS AND QEQASM_PARSER_LIB AND QEQASM_AST_LIB AND QEQASM_FRONTEND_LIB)
        set(QEQASM_FOUND TRUE)
        set(QEQASM_LIBRARIES ${QEQASM_PARSER_LIB} ${QEQASM_AST_LIB} ${QEQASM_FRONTEND_LIB})
        add_definitions(-DQGT_HAS_QEQASM)
        include_directories(${QEQASM_INCLUDE_DIRS})
        message(STATUS "qe-qasm found: ${QEQASM_LIBRARIES}")
    else()
        set(QEQASM_FOUND FALSE)
        message(WARNING "qe-qasm enabled but not found - QASM parsing will use fallback")
    endif()
else()
    set(QEQASM_FOUND FALSE)
    message(STATUS "qe-qasm disabled - QASM parsing will use fallback implementation")
endif()

# libquil support (Quil parser from Rigetti)
# Repository: https://github.com/rigetti/libquil
option(QGT_ENABLE_LIBQUIL "Enable libquil Quil parser library" OFF)
if(QGT_ENABLE_LIBQUIL)
    find_path(LIBQUIL_INCLUDE_DIRS libquil.h
        PATHS /opt/homebrew/include /usr/local/include /usr/include
              ${CMAKE_CURRENT_SOURCE_DIR}/external/libquil/include
              $ENV{HOME}/.local/include)
    find_library(LIBQUIL_LIBRARIES quil
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
              ${CMAKE_CURRENT_SOURCE_DIR}/external/libquil/lib
              $ENV{HOME}/.local/lib)
    if(LIBQUIL_INCLUDE_DIRS AND LIBQUIL_LIBRARIES)
        set(LIBQUIL_FOUND TRUE)
        add_definitions(-DQGT_HAS_LIBQUIL)
        include_directories(${LIBQUIL_INCLUDE_DIRS})
        message(STATUS "libquil found: ${LIBQUIL_LIBRARIES}")
    else()
        set(LIBQUIL_FOUND FALSE)
        message(WARNING "libquil enabled but not found - Quil parsing will use fallback")
    endif()
else()
    set(LIBQUIL_FOUND FALSE)
    message(STATUS "libquil disabled - Quil parsing will use fallback implementation")
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
include_directories(${CMAKE_BINARY_DIR}/include)

# =============================================================================
# Feature Detection Header Generation
# =============================================================================
# Set feature detection variables for qgtl_features.h

# Platform detection for feature header (QGTL_ prefix for template)
# Set defaults
set(QGTL_PLATFORM_MACOS 0)
set(QGTL_PLATFORM_LINUX 0)
set(QGTL_PLATFORM_WINDOWS 0)

if(QGT_PLATFORM_MACOS)
    set(QGTL_PLATFORM_MACOS 1)
elseif(QGT_PLATFORM_LINUX)
    set(QGTL_PLATFORM_LINUX 1)
elseif(QGT_PLATFORM_WINDOWS)
    set(QGTL_PLATFORM_WINDOWS 1)
endif()

# Architecture for feature header
set(QGTL_ARCH_ARM64 0)
set(QGTL_ARCH_X86_64 0)

if(QGT_ARCH_ARM64)
    set(QGTL_ARCH_ARM64 1)
elseif(QGT_ARCH_X86_64)
    set(QGTL_ARCH_X86_64 1)
endif()

# SIMD capabilities for feature header
set(QGTL_SIMD_NEON 0)
set(QGTL_SIMD_AVX 0)
set(QGTL_SIMD_AVX2 0)
set(QGTL_SIMD_AVX512 0)
set(QGTL_SIMD_FMA 0)

# Set SIMD from detected capabilities
if(QGT_HAS_AVX512F)
    set(QGTL_SIMD_AVX512 1)
endif()
if(QGT_HAS_AVX2)
    set(QGTL_SIMD_AVX2 1)
endif()
if(QGT_HAS_AVX)
    set(QGTL_SIMD_AVX 1)
endif()
if(QGT_HAS_FMA)
    set(QGTL_SIMD_FMA 1)
endif()

# Apple Silicon NEON
if(QGT_ARCH_ARM64)
    set(QGTL_SIMD_NEON 1)
endif()

# GPU features
if(QGT_ENABLE_METAL)
    set(QGTL_HAS_METAL 1)
else()
    set(QGTL_HAS_METAL 0)
endif()

if(QGT_ENABLE_CUDA)
    set(QGTL_HAS_CUDA 1)
else()
    set(QGTL_HAS_CUDA 0)
endif()

# Distributed features
if(QGT_ENABLE_MPI)
    set(QGTL_HAS_MPI 1)
else()
    set(QGTL_HAS_MPI 0)
endif()

if(HWLOC_FOUND)
    set(QGTL_HAS_HWLOC 1)
else()
    set(QGTL_HAS_HWLOC 0)
endif()

# External libraries
if(CURL_FOUND)
    set(QGTL_HAS_CURL 1)
else()
    set(QGTL_HAS_CURL 0)
endif()

if(JSON_C_FOUND)
    set(QGTL_HAS_JSON_C 1)
else()
    set(QGTL_HAS_JSON_C 0)
endif()

if(ZLIB_FOUND)
    set(QGTL_HAS_ZLIB 1)
else()
    set(QGTL_HAS_ZLIB 0)
endif()

if(BLAS_FOUND)
    set(QGTL_HAS_BLAS 1)
else()
    set(QGTL_HAS_BLAS 0)
endif()

if(LAPACK_FOUND)
    set(QGTL_HAS_LAPACK 1)
else()
    set(QGTL_HAS_LAPACK 0)
endif()

# On Apple, Accelerate provides BLAS/LAPACK
if(APPLE)
    set(QGTL_HAS_BLAS 1)
    set(QGTL_HAS_LAPACK 1)
endif()

# Format parsers
if(QEQASM_FOUND)
    set(QGTL_HAS_QEQASM 1)
else()
    set(QGTL_HAS_QEQASM 0)
endif()

if(LIBQUIL_FOUND)
    set(QGTL_HAS_LIBQUIL 1)
else()
    set(QGTL_HAS_LIBQUIL 0)
endif()

# Build configuration
if(QGT_BUILD_SHARED)
    set(QGTL_BUILD_SHARED 1)
else()
    set(QGTL_BUILD_SHARED 0)
endif()

if(QGT_BUILD_STATIC)
    set(QGTL_BUILD_STATIC 1)
else()
    set(QGTL_BUILD_STATIC 0)
endif()

# Generate the features header
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/quantum_geometric/core)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/quantum_geometric/core/qgtl_features.h.in
    ${CMAKE_BINARY_DIR}/include/quantum_geometric/core/qgtl_features.h
    @ONLY
)

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
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/supercomputer/backends/*.c
)

# Metal sources (Apple platforms)
if(APPLE)
    file(GLOB QGT_METAL_MM_SOURCES
        ${CMAKE_SOURCE_DIR}/src/quantum_geometric/supercomputer/backends/*.mm
        ${CMAKE_SOURCE_DIR}/src/metal/*.mm
    )
    list(APPEND QGT_SUPERCOMPUTER_SOURCES ${QGT_METAL_MM_SOURCES})
endif()

# CUDA backend sources
if(QGT_ENABLE_CUDA)
    enable_language(CUDA)
    file(GLOB QGT_CUDA_SOURCES
        ${CMAKE_SOURCE_DIR}/src/quantum_geometric/supercomputer/backends/*.cu
        ${CMAKE_SOURCE_DIR}/src/cuda/*.cu
    )
    list(APPEND QGT_SUPERCOMPUTER_SOURCES ${QGT_CUDA_SOURCES})
endif()

# Algorithms module sources (QAOA, Grover, VQE, etc.)
file(GLOB QGT_ALGORITHMS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/quantum_geometric/algorithms/*.c
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
    ${QGT_ALGORITHMS_SOURCES}
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
# Modular Build System (Optional)
# =============================================================================
# When QGT_MODULAR_BUILD is ON, builds separate libraries for each module
# that can be linked independently. Otherwise, builds a single monolithic library.

option(QGT_MODULAR_BUILD "Build separate library modules" OFF)

if(QGT_MODULAR_BUILD)
    message(STATUS "")
    message(STATUS "Building modular libraries...")

    # Helper function to apply platform-specific exclusions
    function(qgt_filter_platform_sources SOURCES_VAR)
        if(NOT APPLE)
            list(FILTER ${SOURCES_VAR} EXCLUDE REGEX ".*_macos\\.c$")
            list(FILTER ${SOURCES_VAR} EXCLUDE REGEX ".*accelerate.*\\.c$")
        else()
            list(FILTER ${SOURCES_VAR} EXCLUDE REGEX ".*/memory_optimization\\.c$")
        endif()
        if(NOT UNIX OR APPLE)
            list(FILTER ${SOURCES_VAR} EXCLUDE REGEX ".*_linux\\.c$")
        endif()
        if(UNIX AND NOT APPLE)
            list(FILTER ${SOURCES_VAR} EXCLUDE REGEX ".*/memory_optimization\\.c$")
        endif()
        set(${SOURCES_VAR} ${${SOURCES_VAR}} PARENT_SCOPE)
    endfunction()

    # Helper function to create a module library
    function(qgt_add_module MODULE_NAME SOURCES)
        if(QGT_BUILD_STATIC)
            add_library(qgtl_${MODULE_NAME} STATIC ${SOURCES})
            target_include_directories(qgtl_${MODULE_NAME} PUBLIC ${QGT_INCLUDE_DIR})
            target_include_directories(qgtl_${MODULE_NAME} PUBLIC ${CMAKE_BINARY_DIR}/include)
            set_target_properties(qgtl_${MODULE_NAME} PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/modules
            )
        endif()
    endfunction()

    # Core module - basic tensor operations, memory, no external deps
    set(QGT_CORE_MINIMAL_SOURCES ${QGT_CORE_SOURCES})
    list(FILTER QGT_CORE_MINIMAL_SOURCES EXCLUDE REGEX ".*_gpu\\.c$")
    list(FILTER QGT_CORE_MINIMAL_SOURCES EXCLUDE REGEX ".*_cuda\\.c$")
    list(FILTER QGT_CORE_MINIMAL_SOURCES EXCLUDE REGEX ".*_metal\\.c$")
    qgt_filter_platform_sources(QGT_CORE_MINIMAL_SOURCES)
    qgt_add_module(core "${QGT_CORE_MINIMAL_SOURCES}")
    if(TARGET qgtl_core)
        target_link_libraries(qgtl_core PUBLIC Threads::Threads m)
        if(APPLE)
            target_link_libraries(qgtl_core PUBLIC ${ACCELERATE_FRAMEWORK})
        elseif(BLAS_FOUND AND LAPACK_FOUND)
            target_link_libraries(qgtl_core PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        endif()
    endif()

    # Physics module - quantum physics operations
    qgt_add_module(physics "${QGT_PHYSICS_SOURCES}")
    if(TARGET qgtl_physics)
        target_link_libraries(qgtl_physics PUBLIC qgtl_core)
    endif()

    # Algorithms module - QAOA, Grover, etc.
    qgt_add_module(algorithms "${QGT_ALGORITHMS_SOURCES}")
    if(TARGET qgtl_algorithms)
        target_link_libraries(qgtl_algorithms PUBLIC qgtl_core)
    endif()

    # Learning module - ML/training pipeline
    qgt_add_module(learning "${QGT_LEARNING_SOURCES}")
    if(TARGET qgtl_learning)
        target_link_libraries(qgtl_learning PUBLIC qgtl_core)
    endif()

    # Hybrid module - quantum-classical integration
    qgt_add_module(hybrid "${QGT_HYBRID_SOURCES}")
    if(TARGET qgtl_hybrid)
        target_link_libraries(qgtl_hybrid PUBLIC qgtl_core qgtl_learning)
    endif()

    # Simulator backend - always available, no cloud deps
    set(QGT_HARDWARE_SIM_SOURCES ${QGT_HARDWARE_SOURCES})
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_ibm.*\\.c$")
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_dwave.*\\.c$")
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_rigetti.*\\.c$")
    # Exclude files that require MPI
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_distributed_execution\\.c$")
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_error_correction\\.c$")
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_error_mitigation\\.c$")
    # Exclude GPU-specific files (these have missing type definitions)
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX "quantum_field_gpu\\.c$")
    list(FILTER QGT_HARDWARE_SIM_SOURCES EXCLUDE REGEX ".*_gpu\\.c$")
    qgt_filter_platform_sources(QGT_HARDWARE_SIM_SOURCES)
    qgt_add_module(hardware_sim "${QGT_HARDWARE_SIM_SOURCES}")
    if(TARGET qgtl_hardware_sim)
        target_link_libraries(qgtl_hardware_sim PUBLIC qgtl_core)
    endif()

    # IBM Quantum backend - requires CURL
    if(CURL_FOUND)
        file(GLOB QGT_IBM_SOURCES "${CMAKE_SOURCE_DIR}/src/quantum_geometric/hardware/quantum_ibm*.c")
        if(QGT_IBM_SOURCES)
            qgt_add_module(hardware_ibm "${QGT_IBM_SOURCES}")
            if(TARGET qgtl_hardware_ibm)
                target_link_libraries(qgtl_hardware_ibm PUBLIC qgtl_core qgtl_hardware_sim ${CURL_LIBRARIES})
            endif()
        endif()
    endif()

    # D-Wave backend - requires CURL and JSON-C
    if(CURL_FOUND AND JSON_C_FOUND)
        file(GLOB QGT_DWAVE_SOURCES "${CMAKE_SOURCE_DIR}/src/quantum_geometric/hardware/quantum_dwave*.c")
        if(QGT_DWAVE_SOURCES)
            qgt_add_module(hardware_dwave "${QGT_DWAVE_SOURCES}")
            if(TARGET qgtl_hardware_dwave)
                target_link_libraries(qgtl_hardware_dwave PUBLIC qgtl_core ${CURL_LIBRARIES} ${JSON_C_LIBRARIES})
            endif()
        endif()

        # Rigetti backend - requires CURL and JSON-C
        file(GLOB QGT_RIGETTI_SOURCES "${CMAKE_SOURCE_DIR}/src/quantum_geometric/hardware/quantum_rigetti*.c")
        if(QGT_RIGETTI_SOURCES)
            qgt_add_module(hardware_rigetti "${QGT_RIGETTI_SOURCES}")
            if(TARGET qgtl_hardware_rigetti)
                target_link_libraries(qgtl_hardware_rigetti PUBLIC qgtl_core ${CURL_LIBRARIES} ${JSON_C_LIBRARIES})
            endif()
        endif()
    endif()

    # Distributed module - requires MPI
    if(QGT_ENABLE_MPI)
        set(QGT_DISTRIBUTED_MODULE_SOURCES ${QGT_DISTRIBUTED_SOURCES})
        qgt_filter_platform_sources(QGT_DISTRIBUTED_MODULE_SOURCES)
        qgt_add_module(distributed "${QGT_DISTRIBUTED_MODULE_SOURCES}")
        if(TARGET qgtl_distributed)
            target_link_libraries(qgtl_distributed PUBLIC qgtl_core MPI::MPI_C)
            if(HWLOC_FOUND)
                target_link_libraries(qgtl_distributed PUBLIC ${HWLOC_LIBRARIES})
            endif()
        endif()
    endif()

    # List all built modules
    message(STATUS "Module libraries built in: ${CMAKE_BINARY_DIR}/lib/modules/")
    message(STATUS "")
endif()

# =============================================================================
# Library Targets (Monolithic)
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

    if(QEQASM_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${QEQASM_LIBRARIES})
    endif()

    if(LIBQUIL_FOUND)
        target_link_libraries(quantum_geometric_static PUBLIC ${LIBQUIL_LIBRARIES})
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

    if(QEQASM_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${QEQASM_LIBRARIES})
    endif()

    if(LIBQUIL_FOUND)
        target_link_libraries(quantum_geometric_shared PUBLIC ${LIBQUIL_LIBRARIES})
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
    PATTERN "*.h.in" EXCLUDE
)

# Install generated feature header
install(FILES ${CMAKE_BINARY_DIR}/include/quantum_geometric/core/qgtl_features.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantum_geometric/core
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "===============================================================")
message(STATUS "  Quantum Geometric Tensor Library (QGTL) Configuration")
message(STATUS "===============================================================")
message(STATUS "")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Static library:     ${QGT_BUILD_STATIC}")
message(STATUS "  Shared library:     ${QGT_BUILD_SHARED}")
message(STATUS "  Modular build:      ${QGT_MODULAR_BUILD}")
message(STATUS "  Tests:              ${QGT_BUILD_TESTS}")
message(STATUS "")
message(STATUS "GPU Acceleration:")
message(STATUS "  CUDA:               ${QGT_ENABLE_CUDA}")
message(STATUS "  Metal:              ${QGT_ENABLE_METAL}")
message(STATUS "")
message(STATUS "Distributed Computing:")
message(STATUS "  MPI:                ${QGT_ENABLE_MPI}")
message(STATUS "  hwloc:              ${HWLOC_FOUND}")
message(STATUS "")
message(STATUS "Hardware Backends:")
if(CURL_FOUND)
    message(STATUS "  IBM Quantum:        YES (CURL found)")
else()
    message(STATUS "  IBM Quantum:        NO (requires CURL)")
endif()
if(CURL_FOUND AND JSON_C_FOUND)
    message(STATUS "  D-Wave:             YES (CURL + JSON-C found)")
    message(STATUS "  Rigetti:            YES (CURL + JSON-C found)")
else()
    message(STATUS "  D-Wave:             NO (requires CURL + JSON-C)")
    message(STATUS "  Rigetti:            NO (requires CURL + JSON-C)")
endif()
message(STATUS "  Simulator:          YES (always available)")
message(STATUS "")
message(STATUS "External Libraries:")
message(STATUS "  CURL:               ${CURL_FOUND}")
message(STATUS "  JSON-C:             ${JSON_C_FOUND}")
message(STATUS "  zlib:               ${ZLIB_FOUND}")
if(APPLE)
    message(STATUS "  BLAS/LAPACK:        YES (Accelerate framework)")
else()
    message(STATUS "  BLAS:               ${BLAS_FOUND}")
    message(STATUS "  LAPACK:             ${LAPACK_FOUND}")
endif()
message(STATUS "")
message(STATUS "Feature header generated at:")
message(STATUS "  ${CMAKE_BINARY_DIR}/include/quantum_geometric/core/qgtl_features.h")
message(STATUS "")
message(STATUS "===============================================================")
message(STATUS "")
