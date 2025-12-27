/**
 * quantum_supercomputer_core.h - Supercomputer-scale quantum computing
 *
 * High-level interface for executing quantum operations across
 * supercomputer-class distributed systems with CUDA GPUs.
 *
 * This module provides:
 * - MPI-based distributed execution across nodes
 * - Multi-GPU support with CUDA, cuBLAS, cuSPARSE
 * - Automatic execution planning and resource allocation
 * - Performance monitoring and optimization
 */

#ifndef QUANTUM_SUPERCOMPUTER_CORE_H
#define QUANTUM_SUPERCOMPUTER_CORE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Feature Detection
// ============================================================================

// Check for CUDA availability
#if defined(__CUDACC__) || defined(CUDA_VERSION) || defined(HAVE_CUDA)
#define QSC_HAS_CUDA 1
#else
#define QSC_HAS_CUDA 0
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

typedef struct SupercomputerContext SupercomputerContext;
typedef struct SupercomputerConfig SupercomputerConfig;
typedef struct QuantumOperation QuantumOperation;
typedef struct OperationProfile OperationProfile;
typedef struct ExecutionPlan ExecutionPlan;
typedef struct OperationResult OperationResult;
typedef struct PerformanceMonitor PerformanceMonitor;

// ============================================================================
// Configuration
// ============================================================================

/**
 * Supercomputer configuration.
 */
struct SupercomputerConfig {
    // Cluster topology
    int num_nodes;              // Number of compute nodes
    int nodes_per_group;        // Nodes per group (for hierarchical)

    // Node resources
    int gpus_per_node;          // GPUs per node
    int cores_per_node;         // CPU cores per node
    size_t memory_per_node;     // Memory per node in bytes

    // Network parameters
    double network_bandwidth;   // Bandwidth in GB/s
    double network_latency;     // Latency in microseconds

    // Execution parameters
    int num_streams_per_gpu;    // CUDA streams per GPU
    bool enable_prefetch;       // Enable memory prefetching
    bool enable_overlap;        // Enable compute-comm overlap

    // Monitor configuration
    void* monitor_config;       // Performance monitor config
};

// ============================================================================
// Public API
// ============================================================================

#if QSC_HAS_CUDA

/**
 * Initialize supercomputer context.
 * Sets up MPI communicators, allocates GPU resources, and initializes
 * performance monitoring.
 *
 * @param config Supercomputer configuration
 * @return Initialized context, or NULL on failure
 */
SupercomputerContext* init_supercomputer(const SupercomputerConfig* config);

/**
 * Execute a quantum operation across the supercomputer.
 * Automatically plans distribution, executes across nodes, and gathers results.
 *
 * @param ctx Supercomputer context
 * @param op Quantum operation to execute
 * @return 0 on success, -1 on failure
 */
int execute_quantum_operation(SupercomputerContext* ctx,
                              const QuantumOperation* op);

/**
 * Clean up supercomputer context.
 * Releases all GPU resources, memory, and MPI communicators.
 *
 * @param ctx Context to clean up
 */
void cleanup_supercomputer(SupercomputerContext* ctx);

#else // !QSC_HAS_CUDA

// Stubs for non-CUDA builds
static inline SupercomputerContext* init_supercomputer(
    const SupercomputerConfig* config) {
    (void)config;
    return NULL;
}

static inline int execute_quantum_operation(SupercomputerContext* ctx,
                                            const QuantumOperation* op) {
    (void)ctx;
    (void)op;
    return -1;
}

static inline void cleanup_supercomputer(SupercomputerContext* ctx) {
    (void)ctx;
}

#endif // QSC_HAS_CUDA

// ============================================================================
// Configuration Helpers
// ============================================================================

/**
 * Initialize configuration with reasonable defaults.
 */
static inline void supercomputer_config_init_default(SupercomputerConfig* config) {
    if (!config) return;

    config->num_nodes = 1;
    config->nodes_per_group = 1;
    config->gpus_per_node = 1;
    config->cores_per_node = 8;
    config->memory_per_node = 16ULL * 1024 * 1024 * 1024;  // 16GB
    config->network_bandwidth = 100.0;  // 100 GB/s
    config->network_latency = 1.0;      // 1 microsecond
    config->num_streams_per_gpu = 4;
    config->enable_prefetch = true;
    config->enable_overlap = true;
    config->monitor_config = NULL;
}

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_SUPERCOMPUTER_CORE_H
