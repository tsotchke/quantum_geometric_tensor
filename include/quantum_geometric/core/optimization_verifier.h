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
    double tolerance;              // Verification tolerance
    void* config_data;           // Additional config data
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

#endif // OPTIMIZATION_VERIFIER_H
