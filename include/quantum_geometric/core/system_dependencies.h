/**
 * @file system_dependencies.h
 * @brief Platform-specific system dependencies
 *
 * UNIFIED NAMING: All macros use QGT_ prefix for consistency across the library.
 * This header provides compile-time platform detection that complements
 * the CMake-based detection (which defines the same macros).
 */

#ifndef SYSTEM_DEPENDENCIES_H
#define SYSTEM_DEPENDENCIES_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// =============================================================================
// Platform Detection (compile-time, complements CMake definitions)
// =============================================================================
#if defined(__linux__)
    #ifndef QGT_PLATFORM_LINUX
        #define QGT_PLATFORM_LINUX 1
    #endif
#elif defined(__APPLE__) && defined(__MACH__)
    #ifndef QGT_PLATFORM_MACOS
        #define QGT_PLATFORM_MACOS 1
    #endif
#elif defined(_WIN32) || defined(_WIN64)
    #ifndef QGT_PLATFORM_WINDOWS
        #define QGT_PLATFORM_WINDOWS 1
    #endif
#endif

// =============================================================================
// Architecture Detection
// =============================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    #ifndef QGT_ARCH_X86_64
        #define QGT_ARCH_X86_64 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #ifndef QGT_ARCH_ARM64
        #define QGT_ARCH_ARM64 1
    #endif
#endif

// =============================================================================
// NUMA Support (Linux only)
// =============================================================================
#ifdef QGT_PLATFORM_LINUX
    #ifdef __has_include
        #if __has_include(<numa.h>)
            #include <numa.h>
            #define QGT_HAS_NUMA 1
        #endif
    #endif
#endif

// Include common system headers
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifndef QGT_PLATFORM_WINDOWS
    #include <sys/mman.h>
#endif
#include <math.h>

// =============================================================================
// Platform-specific memory page size
// =============================================================================
#ifndef QGT_PAGE_SIZE
#ifdef QGT_PLATFORM_LINUX
    #include <unistd.h>
    #define QGT_PAGE_SIZE sysconf(_SC_PAGESIZE)
#elif defined(QGT_PLATFORM_MACOS)
    #include <mach/vm_statistics.h>
    #define QGT_PAGE_SIZE vm_page_size
#elif defined(QGT_PLATFORM_WINDOWS)
    #include <windows.h>
    static inline size_t qgt_get_page_size(void) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwPageSize;
    }
    #define QGT_PAGE_SIZE qgt_get_page_size()
#else
    #define QGT_PAGE_SIZE 4096
#endif
#endif // QGT_PAGE_SIZE

// =============================================================================
// Platform-specific huge pages
// =============================================================================
#ifdef QGT_PLATFORM_LINUX
    #define QGT_HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages
    #define QGT_HAS_HUGE_PAGES 1
#endif

// =============================================================================
// Platform-specific memory barriers
// =============================================================================
#if defined(__GNUC__) || defined(__clang__)
    #define memory_barrier() __sync_synchronize()
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define memory_barrier() _ReadWriteBarrier()
#else
    #define memory_barrier() do {} while(0)
#endif

// =============================================================================
// Platform-specific thread priorities
// =============================================================================
#ifdef QGT_PLATFORM_LINUX
    #include <sched.h>
    #define set_thread_high_priority() do { \
        struct sched_param param; \
        param.sched_priority = sched_get_priority_max(SCHED_FIFO); \
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param); \
    } while(0)
#elif defined(QGT_PLATFORM_MACOS)
    #include <pthread.h>
    #define set_thread_high_priority() do { \
        struct sched_param param; \
        param.sched_priority = sched_get_priority_max(SCHED_FIFO); \
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param); \
    } while(0)
#elif defined(QGT_PLATFORM_WINDOWS)
    #define set_thread_high_priority() SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)
#else
    #define set_thread_high_priority() do {} while(0)
#endif

// =============================================================================
// Platform-specific CPU affinity
// =============================================================================
#ifdef QGT_PLATFORM_LINUX
    #define set_cpu_affinity(cpu) do { \
        cpu_set_t cpuset; \
        CPU_ZERO(&cpuset); \
        CPU_SET(cpu, &cpuset); \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); \
    } while(0)
#elif defined(QGT_PLATFORM_WINDOWS)
    #define set_cpu_affinity(cpu) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)(1ULL << (cpu)))
#else
    #define set_cpu_affinity(cpu) do { (void)(cpu); } while(0)
#endif

// =============================================================================
// SIMD Detection (compile-time, complements CMake definitions)
// =============================================================================
#if defined(QGT_ARCH_X86_64)
    #include <immintrin.h>
    #ifndef QGT_SIMD_AVX
        #if defined(__AVX__)
            #define QGT_SIMD_AVX 1
        #endif
    #endif
    #ifndef QGT_SIMD_AVX2
        #if defined(__AVX2__)
            #define QGT_SIMD_AVX2 1
        #endif
    #endif
    #ifndef QGT_SIMD_AVX512
        #if defined(__AVX512F__)
            #define QGT_SIMD_AVX512 1
        #endif
    #endif
    #ifndef QGT_SIMD_FMA
        #if defined(__FMA__)
            #define QGT_SIMD_FMA 1
        #endif
    #endif
#elif defined(QGT_ARCH_ARM64)
    #include <arm_neon.h>
    #ifndef QGT_SIMD_NEON
        #define QGT_SIMD_NEON 1
    #endif
#endif

// Platform-specific endianness
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define QG_BIG_ENDIAN
#else
    #define QG_LITTLE_ENDIAN
#endif

// Platform-specific inline hints
#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline
#endif

// Platform-specific branch hints
#if defined(__GNUC__) || defined(__clang__)
    #define likely(x)   __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
#else
    #define likely(x)   (x)
    #define unlikely(x) (x)
#endif

// Platform-specific cache line size
#ifdef QGT_PLATFORM_LINUX
    #define CACHE_LINE_SIZE 64
#else
    #define CACHE_LINE_SIZE 64
#endif

// Platform-specific alignment
#define ALIGN_TO(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

// Platform-specific prefetch hints
#if defined(__GNUC__) || defined(__clang__)
    #define prefetch(addr) __builtin_prefetch(addr)
    #define prefetch_write(addr) __builtin_prefetch(addr, 1)
#else
    #define prefetch(addr)
    #define prefetch_write(addr)
#endif

// Platform-specific thread local storage
#if defined(__GNUC__) || defined(__clang__)
    #define THREAD_LOCAL __thread
#else
    #define THREAD_LOCAL __declspec(thread)
#endif

#endif // SYSTEM_DEPENDENCIES_H
