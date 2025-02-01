#ifndef SURFACE_CODE_H
#define SURFACE_CODE_H

#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/physics/syndrome_extraction.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>

#define MAX_SURFACE_SIZE 1024
#define MAX_LOGICAL_QUBITS 64
#define MAX_STABILIZERS 4096

// Surface code type
typedef enum {
    SURFACE_CODE_STANDARD,    // Standard square lattice
    SURFACE_CODE_ROTATED,     // Rotated (more efficient) lattice
    SURFACE_CODE_HEAVY_HEX,   // Heavy hexagonal lattice
    SURFACE_CODE_FLOQUET      // Time-periodic Floquet code
} surface_code_type_t;

// Stabilizer measurement result
typedef struct {
    int value;                // +1 or -1 measurement result
    double confidence;        // Measurement confidence [0,1]
    bool needs_correction;    // Whether correction is needed
} StabilizerResult;

// Stabilizer operator
typedef struct {
    stabilizer_type_t type;   // Type of stabilizer (X/Z)
    size_t* qubits;          // Array of qubit indices
    size_t num_qubits;       // Number of qubits
    StabilizerResult result; // Last measurement result
    double error_rate;       // Current error rate
    double weight;           // Stabilizer weight
    size_t time_step;        // For Floquet codes
} Stabilizer;

// Logical qubit
typedef struct {
    size_t* data_qubits;     // Physical data qubits
    size_t num_data_qubits;  // Number of data qubits
    size_t* stabilizers;     // Associated stabilizers
    size_t num_stabilizers;  // Number of stabilizers
    double logical_error_rate; // Logical error rate
} LogicalQubit;

// Surface code configuration
typedef struct {
    surface_code_type_t type;  // Lattice type
    size_t distance;          // Code distance
    size_t width;            // Lattice width
    size_t height;           // Lattice height
    size_t time_steps;       // For Floquet codes
    double threshold;        // Error threshold
    double measurement_error_rate; // Measurement error rate
    double error_weight_factor;   // Error weight factor
    double correlation_factor;    // Correlation factor
    bool use_metal_acceleration;  // Whether to use Metal acceleration
} SurfaceConfig;

// Metal acceleration types
typedef struct {
    bool enable_optimization;
    size_t num_measurements;
    double error_threshold;
    double confidence_threshold;
    bool use_phase_tracking;
    bool track_correlations;
    size_t history_capacity;
    bool use_metal_acceleration;
    size_t num_stabilizers;
    size_t parallel_group_size;
    double phase_calibration;
    double correlation_factor;
} ZStabilizerConfig;

typedef struct {
    double value;
    double confidence;
    bool needs_correction;
} ZStabilizerMeasurement;

typedef struct {
    ZStabilizerMeasurement* measurements;
    size_t num_measurements;
} ZStabilizerResults;

// Surface code state
typedef struct {
    Stabilizer* stabilizers;  // Array of stabilizers
    size_t num_stabilizers;   // Number of stabilizers
    LogicalQubit* logical_qubits; // Array of logical qubits
    size_t num_logical_qubits;   // Number of logical qubits
    double* correlations;     // Stabilizer correlations
    double total_error_rate;  // Total error rate
    SurfaceConfig config;     // Configuration
    bool initialized;         // Whether initialized
} SurfaceCode;

// Initialize surface code
SurfaceCode* init_surface_code(const SurfaceConfig* config);

// Clean up surface code
void cleanup_surface_code(SurfaceCode* state);

// Measure stabilizers
size_t measure_stabilizers(SurfaceCode* state, StabilizerResult* results);

// Apply corrections based on syndromes
size_t apply_corrections(SurfaceCode* state,
                        const SyndromeVertex* syndromes,
                        size_t num_syndromes);

// Encode logical qubit
int encode_logical_qubit(SurfaceCode* state,
                        const size_t* data_qubits,
                        size_t num_qubits);

// Measure logical qubit
bool measure_logical_qubit(SurfaceCode* state,
                          size_t logical_idx,
                          StabilizerResult* result);

// Update error rates
double update_error_rates(SurfaceCode* state,
                         const StabilizerResult* measurements,
                         size_t num_measurements);

// Get stabilizer
const Stabilizer* get_stabilizer(const SurfaceCode* state,
                                size_t stabilizer_idx);

// Get logical qubit
const LogicalQubit* get_logical_qubit(const SurfaceCode* state,
                                     size_t logical_idx);

// Helper functions
bool validate_surface_config(const SurfaceConfig* config);
void initialize_stabilizers(SurfaceCode* state);
void initialize_logical_qubits(SurfaceCode* state);
bool check_error_threshold(const SurfaceCode* state);
void update_logical_error_rates(SurfaceCode* state);
size_t get_qubit_neighbors(const SurfaceCode* state,
                          size_t qubit_idx,
                          size_t* neighbors,
                          size_t max_neighbors);
bool is_valid_stabilizer_configuration(const SurfaceCode* state,
                                     const Stabilizer* stabilizer);
void apply_stabilizer_corrections(SurfaceCode* state,
                                const Stabilizer* stabilizer);

// Metal acceleration functions
void* get_metal_context(void);
bool measure_z_stabilizers(void* metal_context,
                         void* quantum_state,
                         size_t* stabilizer_indices,
                         const ZStabilizerConfig* config,
                         ZStabilizerResults* results);

#endif // SURFACE_CODE_H
