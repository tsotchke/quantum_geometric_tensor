#ifndef OPTIMIZATION_FLAGS_H
#define OPTIMIZATION_FLAGS_H

#include <stdint.h>

// General optimization flags
#define QG_OPT_NONE                   0x00000000  // No optimizations
#define QG_OPT_ALL                    0xFFFFFFFF  // All optimizations

// Performance optimization flags
#define QG_OPT_PERF_NONE             0x00000000  // No performance optimizations
#define QG_OPT_PERF_SIMD             0x00000001  // Enable SIMD operations
#define QG_OPT_PERF_THREADING        0x00000002  // Enable multi-threading
#define QG_OPT_PERF_GPU              0x00000004  // Enable GPU acceleration
#define QG_OPT_PERF_DISTRIBUTED      0x00000008  // Enable distributed computing
#define QG_OPT_PERF_VECTORIZATION    0x00000010  // Enable vectorization
#define QG_OPT_PERF_PREFETCH         0x00000020  // Enable memory prefetching
#define QG_OPT_PERF_CACHE_ALIGN      0x00000040  // Enable cache alignment
#define QG_OPT_PERF_BRANCH_PREDICT   0x00000080  // Enable branch prediction
#define QG_OPT_PERF_ALL              0x000000FF  // All performance optimizations

// Memory optimization flags
#define QG_OPT_MEM_NONE              0x00000000  // No memory optimizations
#define QG_OPT_MEM_POOL              0x00000100  // Enable memory pooling
#define QG_OPT_MEM_ARENA             0x00000200  // Enable memory arena
#define QG_OPT_MEM_ALIGN             0x00000400  // Enable memory alignment
#define QG_OPT_MEM_COMPRESS          0x00000800  // Enable memory compression
#define QG_OPT_MEM_NUMA              0x00001000  // Enable NUMA awareness
#define QG_OPT_MEM_HUGE_PAGES        0x00002000  // Enable huge pages
#define QG_OPT_MEM_CACHE_COHERENT    0x00004000  // Enable cache coherency
#define QG_OPT_MEM_ZERO_COPY         0x00008000  // Enable zero-copy transfers
#define QG_OPT_MEM_ALL               0x0000FF00  // All memory optimizations

// Quantum optimization flags
#define QG_OPT_QUANTUM_NONE          0x00000000  // No quantum optimizations
#define QG_OPT_QUANTUM_ERROR_CORR    0x00010000  // Enable error correction
#define QG_OPT_QUANTUM_DECOHERENCE   0x00020000  // Enable decoherence handling
#define QG_OPT_QUANTUM_STABILIZER    0x00040000  // Enable stabilizer operations
#define QG_OPT_QUANTUM_MEASUREMENT   0x00080000  // Enable measurement optimization
#define QG_OPT_QUANTUM_CIRCUIT       0x00100000  // Enable circuit optimization
#define QG_OPT_QUANTUM_TOPOLOGY      0x00200000  // Enable topology optimization
#define QG_OPT_QUANTUM_SCHEDULING    0x00400000  // Enable quantum scheduling
#define QG_OPT_QUANTUM_ROUTING       0x00800000  // Enable qubit routing
#define QG_OPT_QUANTUM_ALL           0x00FF0000  // All quantum optimizations

// Algorithm optimization flags
#define QG_OPT_ALGO_NONE             0x00000000  // No algorithm optimizations
#define QG_OPT_ALGO_LOOP_UNROLL      0x01000000  // Enable loop unrolling
#define QG_OPT_ALGO_FUSION           0x02000000  // Enable operation fusion
#define QG_OPT_ALGO_REORDER          0x04000000  // Enable operation reordering
#define QG_OPT_ALGO_ELIMINATE        0x08000000  // Enable dead code elimination
#define QG_OPT_ALGO_INLINE           0x10000000  // Enable function inlining
#define QG_OPT_ALGO_CONSTANT_FOLD    0x20000000  // Enable constant folding
#define QG_OPT_ALGO_STRENGTH_REDUCE  0x40000000  // Enable strength reduction
#define QG_OPT_ALGO_VECTORIZE        0x80000000  // Enable algorithm vectorization
#define QG_OPT_ALGO_ALL              0xFF000000  // All algorithm optimizations

// Optimization levels
typedef enum {
    QG_OPT_LEVEL_NONE = 0,          // No optimizations
    QG_OPT_LEVEL_BASIC = 1,         // Basic optimizations
    QG_OPT_LEVEL_STANDARD = 2,      // Standard optimizations
    QG_OPT_LEVEL_AGGRESSIVE = 3,    // Aggressive optimizations
    QG_OPT_LEVEL_MAX = 4           // Maximum optimizations
} optimization_level_t;

// Optimization configuration
typedef struct {
    uint32_t flags;                 // Optimization flags
    optimization_level_t level;      // Optimization level
    bool enable_profiling;          // Enable performance profiling
    bool enable_validation;         // Enable optimization validation
    bool enable_fallback;          // Enable fallback mechanisms
    void* config_data;            // Additional configuration data
} optimization_config_t;

// Flag manipulation functions
static inline bool has_flag(uint32_t flags, uint32_t flag) {
    return (flags & flag) == flag;
}

static inline uint32_t add_flag(uint32_t flags, uint32_t flag) {
    return flags | flag;
}

static inline uint32_t remove_flag(uint32_t flags, uint32_t flag) {
    return flags & ~flag;
}

static inline uint32_t toggle_flag(uint32_t flags, uint32_t flag) {
    return flags ^ flag;
}

// Level-based flag generation
static inline uint32_t get_level_flags(optimization_level_t level) {
    switch (level) {
        case QG_OPT_LEVEL_NONE:
            return QG_OPT_NONE;
        
        case QG_OPT_LEVEL_BASIC:
            return QG_OPT_PERF_SIMD | QG_OPT_MEM_POOL | 
                   QG_OPT_QUANTUM_ERROR_CORR | QG_OPT_ALGO_CONSTANT_FOLD;
        
        case QG_OPT_LEVEL_STANDARD:
            return QG_OPT_PERF_SIMD | QG_OPT_PERF_THREADING | 
                   QG_OPT_MEM_POOL | QG_OPT_MEM_ALIGN |
                   QG_OPT_QUANTUM_ERROR_CORR | QG_OPT_QUANTUM_CIRCUIT |
                   QG_OPT_ALGO_CONSTANT_FOLD | QG_OPT_ALGO_INLINE;
        
        case QG_OPT_LEVEL_AGGRESSIVE:
            return QG_OPT_PERF_ALL | QG_OPT_MEM_ALL |
                   QG_OPT_QUANTUM_ALL | QG_OPT_ALGO_ALL;
        
        case QG_OPT_LEVEL_MAX:
            return QG_OPT_ALL;
            
        default:
            return QG_OPT_NONE;
    }
}

#endif // OPTIMIZATION_FLAGS_H
