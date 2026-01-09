#ifndef OPTIMIZATION_VERIFIER_H
#define OPTIMIZATION_VERIFIER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include "quantum_geometric/core/optimization_flags.h"
#include "quantum_geometric/core/error_codes.h"

// Verification types
typedef enum {
    VERIFY_CORRECTNESS,      // Verify optimization correctness
    VERIFY_PERFORMANCE,      // Verify performance improvements
    VERIFY_RESOURCE,         // Verify resource usage
    VERIFY_QUANTUM,          // Verify quantum properties
    VERIFY_STABILITY        // Verify numerical stability
} verification_type_t;

// Verification modes
typedef enum {
    MODE_STATIC,            // Static verification
    MODE_DYNAMIC,           // Dynamic verification
    MODE_HYBRID,            // Hybrid verification
    MODE_QUANTUM,           // Quantum verification
    MODE_CONTINUOUS        // Continuous verification
} verification_mode_t;

// Verification levels
typedef enum {
    LEVEL_BASIC,           // Basic verification
    LEVEL_STANDARD,        // Standard verification
    LEVEL_THOROUGH,        // Thorough verification
    LEVEL_EXHAUSTIVE,      // Exhaustive verification
    LEVEL_QUANTUM         // Quantum-specific verification
} verification_level_t;

// Verification metrics
typedef struct {
    double correctness;            // Correctness score
    double performance_gain;       // Performance improvement
    double resource_efficiency;    // Resource efficiency
    double numerical_stability;    // Numerical stability
    double quantum_fidelity;      // Quantum fidelity
    size_t verification_time;     // Time spent verifying
} verification_metrics_t;

// Verification configuration
typedef struct {
    verification_mode_t mode;      // Verification mode
    verification_level_t level;    // Verification level
    bool enable_profiling;         // Enable profiling
    bool enable_assertions;        // Enable assertions
    double tolerance;              // General verification tolerance

    // Specific thresholds for different verification types
    double correctness_threshold;   // Minimum correctness score (default: 0.99)
    double performance_threshold;   // Minimum performance gain ratio (default: 0.0)
    double efficiency_threshold;    // Minimum resource efficiency (default: 0.5)
    double fidelity_threshold;      // Minimum quantum fidelity (default: 0.99)
    double stability_threshold;     // Minimum numerical stability (default: 0.9)

    // Workload parameters for verification tests
    size_t resource_test_size;      // Elements for resource efficiency test (default: 10000)
    size_t quantum_test_qubits;     // Qubits for fidelity test (default: 4)
    size_t stability_matrix_size;   // Matrix size for stability test (default: 16)

    void* config_data;             // Additional config data
} verification_config_t;

// Verification results
typedef struct {
    bool success;                  // Overall success
    verification_metrics_t metrics; // Verification metrics
    char* failure_reason;          // Failure description
    struct timespec timestamp;     // Verification time
    void* result_data;           // Additional result data
} verification_result_t;

// Opaque verifier handle
typedef struct optimization_verifier_t optimization_verifier_t;

// Core functions
optimization_verifier_t* create_optimization_verifier(const verification_config_t* config);
void destroy_optimization_verifier(optimization_verifier_t* verifier);

// Verification functions
qgt_error_t verify_optimization(optimization_verifier_t* verifier,
                              uint32_t opt_flags,
                              verification_type_t type,
                              verification_result_t* result);
qgt_error_t verify_optimization_level(optimization_verifier_t* verifier,
                                    optimization_level_t level,
                                    verification_result_t* result);
qgt_error_t verify_optimization_chain(optimization_verifier_t* verifier,
                                    const uint32_t* opt_flags,
                                    size_t num_flags,
                                    verification_result_t* results);

// Analysis functions
qgt_error_t analyze_optimization_impact(optimization_verifier_t* verifier,
                                      uint32_t opt_flags,
                                      verification_metrics_t* metrics);
qgt_error_t analyze_performance_gain(optimization_verifier_t* verifier,
                                   uint32_t opt_flags,
                                   double* gain);
qgt_error_t analyze_resource_usage(optimization_verifier_t* verifier,
                                 uint32_t opt_flags,
                                 double* efficiency);

// Validation functions
qgt_error_t validate_optimization_flags(optimization_verifier_t* verifier,
                                      uint32_t opt_flags);
qgt_error_t validate_optimization_chain(optimization_verifier_t* verifier,
                                      const uint32_t* opt_flags,
                                      size_t num_flags);
qgt_error_t validate_optimization_config(optimization_verifier_t* verifier,
                                       const optimization_config_t* config);

// Quantum-specific functions
qgt_error_t verify_quantum_optimization(optimization_verifier_t* verifier,
                                      uint32_t opt_flags,
                                      verification_result_t* result);
qgt_error_t analyze_quantum_impact(optimization_verifier_t* verifier,
                                 uint32_t opt_flags,
                                 verification_metrics_t* metrics);
qgt_error_t validate_quantum_properties(optimization_verifier_t* verifier,
                                      uint32_t opt_flags);

// Monitoring functions
qgt_error_t start_verification_monitoring(optimization_verifier_t* verifier);
qgt_error_t stop_verification_monitoring(optimization_verifier_t* verifier);
qgt_error_t get_verification_stats(const optimization_verifier_t* verifier,
                                 verification_metrics_t* metrics);

// Reporting functions
qgt_error_t generate_verification_report(const optimization_verifier_t* verifier,
                                       const verification_result_t* result,
                                       char** report);
qgt_error_t export_verification_data(const optimization_verifier_t* verifier,
                                   const char* filename);
qgt_error_t import_verification_data(optimization_verifier_t* verifier,
                                   const char* filename);

// Utility functions
void free_verification_result(verification_result_t* result);
const char* get_verification_error(qgt_error_t error);
bool is_optimization_valid(uint32_t opt_flags);

// ===========================================================================
// Simplified API for Testing - Complexity and Resource Verification
// ===========================================================================

// Operation types for complexity verification
typedef enum {
    FIELD_COUPLING = 0,
    TENSOR_CONTRACTION = 1,
    GEOMETRIC_TRANSFORM = 2,
    MEMORY_ACCESS = 3,
    COMMUNICATION = 4
} operation_type_t;

// Target efficiency thresholds
#define TARGET_MEMORY_EFFICIENCY 0.85
#define TARGET_GPU_EFFICIENCY 0.80

// Memory metrics structure
typedef struct MemoryMetrics {
    double pool_utilization;       // Memory pool utilization (0-1)
    double fragmentation_ratio;    // Memory fragmentation ratio (0-1)
    size_t cache_hits;             // Number of cache hits
    size_t cache_misses;           // Number of cache misses
    size_t current_allocated;      // Currently allocated bytes
    size_t peak_allocated;         // Peak allocated bytes
    size_t total_allocations;      // Total allocation count
    size_t total_frees;            // Total free count
} MemoryMetrics;

// GPU metrics structure
typedef struct GPUMetrics {
    double compute_utilization;    // GPU compute utilization (0-1)
    double memory_utilization;     // GPU memory utilization (0-1)
    double bandwidth_utilization;  // Memory bandwidth utilization (0-1)
    size_t memory_used;            // GPU memory used in bytes
    size_t memory_total;           // Total GPU memory in bytes
    double temperature;            // GPU temperature in Celsius
    double power_usage;            // GPU power usage in watts
    double clock_speed;            // Current clock speed in MHz
} GPUMetrics;

// Optimization report structure
typedef struct OptimizationReport {
    // Complexity optimization flags
    bool field_coupling_optimized;
    bool tensor_contraction_optimized;
    bool geometric_transform_optimized;
    bool memory_access_optimized;
    bool communication_optimized;

    // Efficiency metrics
    double memory_efficiency;
    double gpu_efficiency;
    double overall_efficiency;

    // Performance metrics
    double speedup_factor;
    double throughput_improvement;

    // Resource usage
    MemoryMetrics memory_metrics;
    GPUMetrics gpu_metrics;

    // Timing
    double verification_time;
    struct timespec timestamp;
} OptimizationReport;

// Verify operation has O(log n) complexity
bool verify_log_complexity(operation_type_t operation);

// Measure current memory usage metrics
MemoryMetrics measure_memory_usage(void);

// Measure current GPU utilization metrics
GPUMetrics measure_gpu_utilization(void);

// Verify all optimizations and generate report
OptimizationReport verify_all_optimizations(void);

// Print optimization report to stdout
void print_optimization_report(const OptimizationReport* report);

// Calculate field hierarchically (optimized O(log n) version)
double calculate_field_hierarchical(const double* field, size_t size, size_t index);

// Contract tensors using Strassen algorithm
void contract_tensors_strassen(double* result, const double* a, const double* b,
                               size_t m, size_t n, size_t k);

// Fast geometric transform
void transform_geometric_fast(double* output, const double* input, size_t dimension);

#endif // OPTIMIZATION_VERIFIER_H
