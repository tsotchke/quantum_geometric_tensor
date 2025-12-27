/**
 * @file full_topological_protection.h
 * @brief Full topological protection system for quantum error correction
 *
 * Provides a complete system for protecting quantum states using
 * topological error correction, including error tracking, verification,
 * and anyon manipulation.
 */

#ifndef FULL_TOPOLOGICAL_PROTECTION_H
#define FULL_TOPOLOGICAL_PROTECTION_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/physics/surface_code.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_ANYONS 256
#define MAX_CORRECTION_STEPS 1024

// ============================================================================
// Error Codes for Topological Protection
// ============================================================================
// TopologicalErrorCode is defined in quantum_types.h. Additional aliases
// for backward compatibility with this module:
#define TOPO_ERROR_SUCCESS TOPO_NO_ERROR
#define TOPO_ERROR_INVALID_PARAMETERS TOPO_ERROR_INVALID_STATE
#define TOPO_ERROR_INITIALIZATION_FAILED TOPO_ERROR_OUT_OF_MEMORY
#define TOPO_ERROR_MEMORY_ALLOCATION_FAILED TOPO_ERROR_OUT_OF_MEMORY
#define TOPO_ERROR_MEASUREMENT_FAILED TOPO_ERROR_DETECTED
#define TOPO_ERROR_VERIFICATION_FAILED TOPO_ERROR_DETECTED

// ============================================================================
// Hardware Configuration
// ============================================================================

/**
 * Hardware-specific configuration for error correction
 */
typedef struct HardwareConfig {
    size_t num_qubits;              // Number of qubits
    double gate_error_rate;         // Gate error rate
    double measurement_error_rate;  // Measurement error rate
    double coherence_time;          // T2 coherence time
    double t1_time;                 // T1 relaxation time
    double readout_fidelity;        // Measurement readout fidelity
    bool supports_mid_circuit;      // Supports mid-circuit measurement
} HardwareConfig;

// ============================================================================
// Error Tracker
// ============================================================================

/**
 * Tracks errors and correlations over time
 */
typedef struct ErrorTracker {
    double* error_rates;            // Per-qubit error rates
    double* error_correlations;     // Error correlations
    size_t* error_locations;        // Recent error locations
    size_t capacity;                // Array capacity
    size_t num_errors;              // Current error count
    double total_weight;            // Total error weight
} ErrorTracker;

// ============================================================================
// Verification System
// ============================================================================

/**
 * Verification system for quantum state quality
 */
typedef struct VerificationSystem {
    double* stability_metrics;      // Stability metrics per qubit
    double* coherence_metrics;      // Coherence metrics per qubit
    double* fidelity_metrics;       // Fidelity metrics per qubit
    size_t num_metrics;             // Number of metrics
    double threshold_stability;     // Stability threshold
    double threshold_coherence;     // Coherence threshold
    double threshold_fidelity;      // Fidelity threshold
} VerificationSystem;

// ============================================================================
// Protection System
// ============================================================================

/**
 * Main protection system structure
 */
typedef struct ProtectionSystem {
    HardwareConfig* config;         // Hardware configuration
    ErrorTracker* error_tracker;    // Error tracker
    VerificationSystem* verifier;   // Verification system
    double fast_cycle_interval;     // Fast cycle interval
    double medium_cycle_interval;   // Medium cycle interval
    double slow_cycle_interval;     // Slow cycle interval
    bool active;                    // Protection active flag
} ProtectionSystem;

// ============================================================================
// Anyon Set
// ============================================================================

/**
 * Collection of detected anyons
 */
typedef struct AnyonSet {
    size_t* positions;              // Anyon positions
    double* charges;                // Anyon charges
    double* energies;               // Anyon energies
    size_t num_anyons;              // Number of anyons
    size_t capacity;                // Array capacity
} AnyonSet;

// ============================================================================
// Correction Pattern
// ============================================================================

/**
 * Pattern for error correction
 */
typedef struct CorrectionPattern {
    size_t* qubit_indices;          // Qubits to correct
    int* correction_types;          // Type of correction (X, Y, Z)
    size_t num_corrections;         // Number of corrections
    size_t capacity;                // Array capacity
    double expected_fidelity;       // Expected fidelity after correction
} CorrectionPattern;

// ============================================================================
// Correction System
// ============================================================================

/**
 * System for managing corrections
 */
typedef struct CorrectionSystem {
    CorrectionPattern* pattern;     // Current correction pattern
    HardwareConfig* config;         // Hardware configuration
    size_t max_iterations;          // Maximum correction iterations
    double convergence_threshold;   // Convergence threshold
} CorrectionSystem;

// ============================================================================
// Stabilizer Result and Surface Code Configuration
// ============================================================================
// StabilizerResult, SurfaceConfig, and surface_code_type_t are defined in
// surface_code.h which is included above. Use those types for consistency.

// ============================================================================
// Initialization and Cleanup Functions
// ============================================================================

ErrorTracker* init_error_tracker(quantum_state_t* state, const HardwareConfig* config);
void free_error_tracker(ErrorTracker* tracker);

VerificationSystem* init_verification_system(quantum_state_t* state, const HardwareConfig* config);
void free_verification_system(VerificationSystem* system);

ProtectionSystem* init_protection_system(quantum_state_t* state, const HardwareConfig* config);
void free_protection_system(ProtectionSystem* system);

// ============================================================================
// Error Detection and Correction
// ============================================================================

TopologicalErrorCode detect_topological_errors(quantum_state_t* state, const HardwareConfig* config);
void correct_topological_errors(quantum_state_t* state, const HardwareConfig* config);
void protect_topological_state(quantum_state_t* state, const HardwareConfig* config);
bool verify_topological_state(quantum_state_t* state, const HardwareConfig* config);

// ============================================================================
// Anyon Operations
// ============================================================================

AnyonSet* detect_mitigated_anyons(quantum_state_t* state, CorrectionSystem* system);
void free_anyon_set(AnyonSet* anyons);
CorrectionPattern* optimize_correction_pattern(AnyonSet* anyons, CorrectionSystem* system);
void free_correction_pattern(CorrectionPattern* pattern);
void apply_mitigated_correction(quantum_state_t* state, CorrectionPattern* pattern, CorrectionSystem* system);

// ============================================================================
// Topological Surface Code Operations
// ============================================================================
// Note: init_surface_code, cleanup_surface_code, and measure_stabilizers
// are defined in surface_code.h with different signatures. Use those for
// general surface code operations. The functions below are for the
// topological protection system's internal surface code management.

bool topo_init_surface_code(const SurfaceConfig* config);
void topo_cleanup_surface_code(void);
size_t topo_measure_stabilizers(StabilizerResult* results);

// ============================================================================
// Helper Functions
// ============================================================================

bool should_run_fast_cycle(ProtectionSystem* system);
bool should_run_medium_cycle(ProtectionSystem* system);
bool should_run_slow_cycle(ProtectionSystem* system);
void wait_protection_interval(ProtectionSystem* system);
void mark_error_location(quantum_state_t* state, size_t location);
void log_correction_failure(quantum_state_t* state, CorrectionSystem* system);

#ifdef __cplusplus
}
#endif

#endif // FULL_TOPOLOGICAL_PROTECTION_H
