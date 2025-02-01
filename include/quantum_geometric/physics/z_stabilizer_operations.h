#ifndef Z_STABILIZER_OPERATIONS_H
#define Z_STABILIZER_OPERATIONS_H

#include "quantum_geometric/physics/quantum_stabilizer.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdbool.h>

// Z stabilizer measurement result
typedef struct {
    int value;                  // +1 or -1 measurement result
    double confidence;          // Measurement confidence [0,1]
    double error_rate;         // Error rate for this measurement
    bool needs_correction;     // Whether correction is needed
    void* auxiliary_data;      // Additional measurement data
} z_measurement_t;

// Z stabilizer hardware configuration
typedef struct {
    double phase_calibration;        // Phase calibration factor
    double z_gate_fidelity;         // Z gate fidelity
    double measurement_fidelity;    // Measurement fidelity
    size_t echo_sequence_length;    // Echo sequence length
    bool dynamic_phase_correction;  // Enable dynamic phase correction
} ZHardwareConfig;

// Z stabilizer configuration
typedef struct {
    size_t repetition_count;         // Number of measurement repetitions
    double error_threshold;          // Error detection threshold
    double confidence_threshold;     // Confidence threshold
    size_t history_capacity;         // Measurement history capacity
    double phase_calibration;        // Phase calibration factor
    size_t echo_sequence_length;     // Echo sequence length
    bool dynamic_phase_correction;   // Enable dynamic phase correction
    bool enable_z_optimization;      // Enable Z-specific optimizations
} ZStabilizerConfig;

// Z stabilizer state
typedef struct {
    ZStabilizerConfig config;           // Configuration parameters
    double* phase_correlations;         // Phase correlation tracking
    double* measurement_confidences;    // Measurement confidence values
    double* measurement_history;        // History of measurements
    double* stabilizer_values;          // Current stabilizer values
    size_t history_size;               // Current history size
    double phase_error_rate;           // Current phase error rate
} ZStabilizerState;

// Z stabilizer results
typedef struct {
    double average_fidelity;          // Average measurement fidelity
    double phase_stability;           // Phase stability metric
    double correlation_strength;      // Correlation strength metric
    size_t measurement_count;         // Total measurements performed
    double error_suppression_factor;  // Error suppression factor
} ZStabilizerResults;

// Initialize Z stabilizer measurement
qgt_error_t z_measurement_create(z_measurement_t** measurement);

// Destroy Z stabilizer measurement
void z_measurement_destroy(z_measurement_t* measurement);

// Perform Z stabilizer measurement
qgt_error_t z_stabilizer_measure(z_measurement_t* measurement,
                                const quantum_stabilizer_t* stabilizer,
                                const quantum_geometric_state_t* state);

// Check if Z measurement indicates error
qgt_error_t z_measurement_has_error(bool* has_error,
                                   const z_measurement_t* measurement,
                                   double threshold);

// Get Z measurement reliability
qgt_error_t z_measurement_reliability(double* reliability,
                                    const z_measurement_t* measurement);

// Compare two Z measurements
qgt_error_t z_measurement_compare(bool* equal,
                                const z_measurement_t* measurement1,
                                const z_measurement_t* measurement2,
                                double tolerance);

// Validate Z measurement result
qgt_error_t z_measurement_validate(const z_measurement_t* measurement);

// Apply Z correction based on measurement
qgt_error_t z_stabilizer_correct(quantum_geometric_state_t* state,
                                const z_measurement_t* measurement,
                                const quantum_stabilizer_t* stabilizer);

// Get Z stabilizer correlation between two stabilizers
qgt_error_t z_stabilizer_correlation(double* correlation,
                                   const quantum_stabilizer_t* stabilizer1,
                                   const quantum_stabilizer_t* stabilizer2,
                                   const quantum_geometric_state_t* state);

// Check if two Z stabilizers commute
qgt_error_t z_stabilizer_commute(bool* commute,
                                const quantum_stabilizer_t* stabilizer1,
                                const quantum_stabilizer_t* stabilizer2);

// Get Z stabilizer weight (number of non-identity terms)
qgt_error_t z_stabilizer_weight(size_t* weight,
                               const quantum_stabilizer_t* stabilizer);

// Check if Z stabilizer is valid
qgt_error_t z_stabilizer_validate(const quantum_stabilizer_t* stabilizer);

// Apply Z error mitigation sequence
bool apply_z_error_mitigation_sequence(
    ZStabilizerState* state,
    size_t x,
    size_t y
);

// Get Z stabilizer correlation between points
double get_z_stabilizer_correlation(
    const ZStabilizerState* state,
    size_t x1,
    size_t y1,
    size_t x2,
    size_t y2
);

// Apply Z measurement correction
void apply_z_measurement_correction(
    ZStabilizerState* state,
    size_t x,
    size_t y,
    double* result
);

// Measure Z stabilizers in parallel
bool measure_z_stabilizers_parallel(
    ZStabilizerState* state,
    const size_t* qubit_coords,
    size_t num_qubits,
    double* results
);

// Update Z measurement history
bool update_z_measurement_history(
    ZStabilizerState* state,
    size_t x,
    size_t y,
    double result
);

// Get Z stabilizer results
ZStabilizerResults get_z_stabilizer_results(const ZStabilizerState* state);

// Get Z error rate
double get_z_error_rate(const ZStabilizerState* state);

// Optimize Z measurement sequence
bool optimize_z_measurement_sequence(
    ZStabilizerState* state,
    const ZHardwareConfig* hardware
);

// Apply hardware-specific Z optimizations
bool apply_hardware_z_optimizations(
    ZStabilizerState* state,
    const ZHardwareConfig* hardware
);

// Initialize Z stabilizer measurement system
ZStabilizerState* init_z_stabilizer_measurement(
    const ZStabilizerConfig* config,
    const ZHardwareConfig* hardware
);

// Clean up Z stabilizer measurement system
void cleanup_z_stabilizer_measurement(ZStabilizerState* state);

#endif // Z_STABILIZER_OPERATIONS_H
