/**
 * @file tensor_network_optimizer.h
 * @brief Tensor network optimization for quantum transformers
 *
 * Provides optimization strategies for tensor network contractions
 * including quantum-inspired and geometric approaches.
 */

#ifndef TENSOR_NETWORK_OPTIMIZER_H
#define TENSOR_NETWORK_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/tensor_network_operations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimization strategies
typedef enum {
    STRATEGY_NONE = 0,
    STRATEGY_QUANTUM_INSPIRED = 1,  // Quantum-inspired optimization
    STRATEGY_GEOMETRIC = 2,         // Geometric optimization
    STRATEGY_HYBRID = 3,            // Hybrid quantum-geometric
    STRATEGY_ADAPTIVE = 4,          // Adaptive strategy selection
    STRATEGY_GREEDY = 5,            // Greedy contraction order
    STRATEGY_EXHAUSTIVE = 6,        // Exhaustive search (small networks only)
    STRATEGY_SVD = 7                // SVD-based decomposition
} optimization_strategy_t;

// Type alias for CamelCase compatibility
typedef optimization_strategy_t OptimizationStrategy;

// Optimizer configuration
typedef struct tensor_optimizer_config {
    optimization_strategy_t default_strategy;  // Default optimization strategy
    size_t max_iterations;                     // Maximum optimization iterations
    double convergence_threshold;              // Convergence threshold
    double learning_rate;                      // Learning rate for gradient-based
    bool use_caching;                          // Enable result caching
    bool use_parallel;                         // Enable parallel optimization
    size_t num_threads;                        // Thread count for parallel
    double memory_limit;                       // Memory limit in bytes
    bool track_metrics;                        // Enable performance tracking
} tensor_optimizer_config_t;

// Optimizer metrics
typedef struct tensor_optimizer_metrics {
    size_t total_optimizations;          // Total optimizations performed
    size_t cache_hits;                   // Cache hits
    size_t cache_misses;                 // Cache misses
    double total_time;                   // Total optimization time
    double average_improvement;          // Average optimization improvement
    double best_improvement;             // Best optimization improvement
    size_t iterations_used;              // Iterations used in last optimization
    double memory_used;                  // Memory used
} tensor_optimizer_metrics_t;

// Contraction path element
typedef struct contraction_step {
    size_t node1_id;                     // First node to contract
    size_t node2_id;                     // Second node to contract
    size_t result_id;                    // Resulting node ID
    double cost;                         // Estimated cost of this step
    double memory_cost;                  // Memory required
} contraction_step_t;

// Contraction path
typedef struct contraction_path {
    contraction_step_t* steps;           // Steps in the path
    size_t num_steps;                    // Number of steps
    double total_cost;                   // Total estimated cost
    double total_memory;                 // Total memory requirement
    bool is_optimal;                     // Whether path is optimal
} contraction_path_t;

// Forward declarations for optimizer internals
struct contraction_cache;
struct tensor_memory_pool;

// Tensor network optimizer
typedef struct tensor_network_optimizer {
    tensor_optimizer_config_t config;    // Configuration
    tensor_optimizer_metrics_t metrics;  // Performance metrics

    // Caching structures
    contraction_path_t** path_cache;     // Cached contraction paths
    size_t cache_size;                   // Current cache size
    size_t cache_capacity;               // Maximum cache capacity

    // SIMD and acceleration state
    bool simd_enabled;                   // SIMD operations enabled
    struct contraction_cache* contraction_cache;  // Contraction results cache
    struct tensor_memory_pool* memory_pool;       // Memory pool for tensors

    // Internal state
    bool initialized;                    // Initialization flag
    void* internal_state;                // Optimizer-specific state
} tensor_network_optimizer_t;

// CamelCase type alias for compatibility
typedef tensor_network_optimizer_t TensorNetworkOptimizer;

// ============================================================================
// Core Functions
// ============================================================================

/**
 * @brief Initialize tensor network optimizer with default settings
 * @return Pointer to initialized optimizer or NULL on failure
 */
tensor_network_optimizer_t* init_tensor_optimizer(void);

/**
 * @brief Create tensor network optimizer with configuration
 * @param config Optimizer configuration
 * @return Pointer to initialized optimizer or NULL on failure
 */
tensor_network_optimizer_t* create_tensor_optimizer(
    const tensor_optimizer_config_t* config);

/**
 * @brief Destroy tensor network optimizer
 * @param optimizer Optimizer to destroy
 */
void cleanup_tensor_optimizer(tensor_network_optimizer_t* optimizer);

/**
 * @brief Destroy tensor network optimizer (alias)
 * @param optimizer Optimizer to destroy
 */
void destroy_tensor_optimizer(tensor_network_optimizer_t* optimizer);

// ============================================================================
// Optimization Functions
// ============================================================================

/**
 * @brief Optimize tensor network with specified strategy
 * @param optimizer Tensor network optimizer
 * @param network Tensor network to optimize
 * @param strategy Optimization strategy
 * @return true on success, false on failure
 */
bool optimize_tensor_network(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network,
    optimization_strategy_t strategy);

/**
 * @brief Find optimal contraction path for network
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @param strategy Optimization strategy
 * @param path Output contraction path
 * @return true on success
 */
bool find_optimal_path(
    tensor_network_optimizer_t* optimizer,
    const tensor_network_t* network,
    optimization_strategy_t strategy,
    contraction_path_t** path);

/**
 * @brief Execute contraction path on network
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @param path Contraction path to execute
 * @param result Output result data
 * @param result_size Output result size
 * @return true on success
 */
bool execute_contraction_path(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network,
    const contraction_path_t* path,
    ComplexFloat** result,
    size_t* result_size);

// ============================================================================
// Network Creation Functions
// ============================================================================

/**
 * @brief Create projection network for attention
 * @param optimizer Tensor network optimizer
 * @param input_dim Input dimension
 * @param output_dim Output dimension
 * @return Pointer to created network or NULL on failure
 */
tensor_network_t* create_projection_network(
    tensor_network_optimizer_t* optimizer,
    size_t input_dim,
    size_t output_dim);

/**
 * @brief Create feed-forward network
 * @param optimizer Tensor network optimizer
 * @param input_dim Input dimension
 * @param output_dim Output dimension
 * @return Pointer to created network or NULL on failure
 */
tensor_network_t* create_feed_forward_network(
    tensor_network_optimizer_t* optimizer,
    size_t input_dim,
    size_t output_dim);

/**
 * @brief Create attention network
 * @param optimizer Tensor network optimizer
 * @param hidden_dim Hidden dimension
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @return Pointer to created network or NULL on failure
 */
tensor_network_t* create_attention_network(
    tensor_network_optimizer_t* optimizer,
    size_t hidden_dim,
    size_t num_heads,
    size_t head_dim);

/**
 * @brief Cleanup tensor network (wrapper)
 * @param network Network to cleanup
 */
void cleanup_tensor_network(tensor_network_t* network);

// ============================================================================
// Strategy-Specific Functions
// ============================================================================

/**
 * @brief Perform quantum-inspired optimization
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @return true on success
 */
bool optimize_quantum_inspired(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network);

/**
 * @brief Perform geometric optimization
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @return true on success
 */
bool optimize_geometric(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network);

/**
 * @brief Perform hybrid optimization
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @return true on success
 */
bool optimize_hybrid(
    tensor_network_optimizer_t* optimizer,
    tensor_network_t* network);

/**
 * @brief Select best strategy for network
 * @param optimizer Tensor network optimizer
 * @param network Tensor network
 * @return Recommended optimization strategy
 */
optimization_strategy_t select_best_strategy(
    const tensor_network_optimizer_t* optimizer,
    const tensor_network_t* network);

// ============================================================================
// Cost Estimation Functions
// ============================================================================

/**
 * @brief Estimate contraction cost
 * @param network Tensor network
 * @param node1_id First node ID
 * @param node2_id Second node ID
 * @return Estimated cost
 */
double estimate_contraction_cost(
    const tensor_network_t* network,
    size_t node1_id,
    size_t node2_id);

/**
 * @brief Estimate path cost
 * @param network Tensor network
 * @param path Contraction path
 * @return Total estimated cost
 */
double estimate_path_cost(
    const tensor_network_t* network,
    const contraction_path_t* path);

/**
 * @brief Estimate memory requirement for path
 * @param network Tensor network
 * @param path Contraction path
 * @return Memory requirement in bytes
 */
size_t estimate_path_memory(
    const tensor_network_t* network,
    const contraction_path_t* path);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get optimizer metrics
 * @param optimizer Tensor network optimizer
 * @param metrics Output metrics structure
 * @return true on success
 */
bool get_optimizer_metrics(
    const tensor_network_optimizer_t* optimizer,
    tensor_optimizer_metrics_t* metrics);

/**
 * @brief Reset optimizer metrics
 * @param optimizer Tensor network optimizer
 */
void reset_optimizer_metrics(tensor_network_optimizer_t* optimizer);

/**
 * @brief Clear optimizer cache
 * @param optimizer Tensor network optimizer
 */
void clear_optimizer_cache(tensor_network_optimizer_t* optimizer);

/**
 * @brief Free contraction path
 * @param path Path to free
 */
void free_contraction_path(contraction_path_t* path);

/**
 * @brief Validate optimizer configuration
 * @param config Configuration to validate
 * @return true if valid
 */
bool validate_optimizer_config(const tensor_optimizer_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_NETWORK_OPTIMIZER_H
