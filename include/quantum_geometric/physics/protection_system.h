/**
 * @file protection_system.h
 * @brief Quantum state protection and error correction system
 */

#ifndef QUANTUM_GEOMETRIC_PROTECTION_SYSTEM_H
#define QUANTUM_GEOMETRIC_PROTECTION_SYSTEM_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include <stdbool.h>

// Forward declarations
typedef struct ErrorTracker ErrorTracker;
typedef struct ProtectionVerifier ProtectionVerifier;

// Protection system structure
typedef struct ProtectionSystem {
    ErrorTracker* error_tracker;           // Tracks error patterns
    ProtectionVerifier* verifier;          // Verifies protection status
    double protection_threshold;           // Protection quality threshold
    size_t fast_cycle_interval;           // Fast cycle interval
    size_t medium_cycle_interval;         // Medium cycle interval
    size_t slow_cycle_interval;           // Slow cycle interval
    size_t current_cycle;                 // Current protection cycle
    void* protection_data;                // Additional protection data
} ProtectionSystem;

// Initialize protection system
ProtectionSystem* init_protection_system(const quantum_state* state,
                                       const IBMConfig* config);

// Clean up protection system
void cleanup_protection_system(ProtectionSystem* system);

// Protection cycle management
bool should_run_fast_cycle(const ProtectionSystem* system);
bool should_run_medium_cycle(const ProtectionSystem* system);
bool should_run_slow_cycle(const ProtectionSystem* system);

// Error detection and correction
bool detect_topological_errors(const quantum_state* state,
                             const IBMConfig* config);
bool correct_topological_errors(quantum_state* state,
                              const IBMConfig* config);
bool verify_topological_state(const quantum_state* state,
                            const IBMConfig* config);

// Error logging and recovery
void log_correction_failure(const quantum_state* state,
                          const char* message);

// Enhanced error detection
AnyonSet* detect_mitigated_anyons(const quantum_state* state,
                                 const void* config);

// Correction pattern optimization
CorrectionPattern* optimize_correction_pattern(const AnyonSet* anyons,
                                             const void* config);

// Apply mitigated correction
bool apply_mitigated_correction(quantum_state* state,
                              const CorrectionPattern* pattern,
                              const void* config);

// Memory management
void free_correction_pattern(CorrectionPattern* pattern);
void free_anyon_set(AnyonSet* anyons);

// Protection cycle timing
void wait_protection_interval(const ProtectionSystem* system);

#endif // QUANTUM_GEOMETRIC_PROTECTION_SYSTEM_H
