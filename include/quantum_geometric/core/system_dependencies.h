/**
 * @file system_dependencies.h
 * @brief Platform-specific system dependencies
 */

#ifndef SYSTEM_DEPENDENCIES_H
#define SYSTEM_DEPENDENCIES_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// Platform detection
#if defined(__linux__)
    #define QG_PLATFORM_LINUX
#elif defined(__APPLE__)
    #define QG_PLATFORM_MACOS
#else
    #error "Unsupported platform"
#endif

// NUMA support
#ifdef QG_PLATFORM_LINUX
    #include <numa.h>
    #define QG_HAS_NUMA
#endif

// Include common system headers
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <math.h>

// Platform-specific memory page size
#ifdef QG_PLATFORM_LINUX
    #include <unistd.h>
    #define QG_PAGE_SIZE sysconf(_SC_PAGESIZE)
#else
    #include <mach/vm_statistics.h>
    #define QG_PAGE_SIZE vm_page_size
#endif

// Platform-specific huge pages
#ifdef QG_PLATFORM_LINUX
    #define QG_HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages
    #define QG_HAS_HUGE_PAGES
#endif

// Platform-specific memory barriers
#ifdef QG_PLATFORM_LINUX
    #define memory_barrier() __sync_synchronize()
#else
    #define memory_barrier() __sync_synchronize()
#endif

// Platform-specific thread priorities
#ifdef QG_PLATFORM_LINUX
    #include <sched.h>
    #define set_thread_high_priority() do { \
        struct sched_param param; \
        param.sched_priority = sched_get_priority_max(SCHED_FIFO); \
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param); \
    } while(0)
#else
    #include <pthread.h>
    #define set_thread_high_priority() do { \
        struct sched_param param; \
        param.sched_priority = sched_get_priority_max(SCHED_FIFO); \
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param); \
    } while(0)
#endif

// Platform-specific CPU affinity
#ifdef QG_PLATFORM_LINUX
    #define set_cpu_affinity(cpu) do { \
        cpu_set_t cpuset; \
        CPU_ZERO(&cpuset); \
        CPU_SET(cpu, &cpuset); \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); \
    } while(0)
#else
    #define set_cpu_affinity(cpu) do {} while(0)
#endif

// Platform-specific SIMD detection
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define QG_HAS_AVX
    #define QG_HAS_AVX2
    #if defined(__AVX512F__)
        #define QG_HAS_AVX512
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define QG_HAS_NEON
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
#ifdef QG_PLATFORM_LINUX
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
