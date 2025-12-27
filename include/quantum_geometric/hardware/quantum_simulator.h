/**
 * @file quantum_simulator.h
 * @brief Semi-classical quantum simulator for quantum hardware emulation
 */

#ifndef QUANTUM_SIMULATOR_H
#define QUANTUM_SIMULATOR_H

#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include <complex.h>

// Maximum number of qubits supported by the simulator
#ifndef MAX_QUBITS
#define MAX_QUBITS 32
#endif

// Use GateType from quantum_circuit.h

// Use types from quantum_hardware_types.h
// Note: QuantumGate is defined as a struct directly in quantum_hardware_types.h
// We use quantum_gate_t for internal gate operations and QuantumGate for API
#ifndef QUANTUM_SIMULATOR_TYPES_DEFINED
#define QUANTUM_SIMULATOR_TYPES_DEFINED
// Forward declarations to avoid redefinition
struct quantum_circuit_t;
typedef struct quantum_circuit_t SimulatorCircuit;
#endif

// Noise model types
typedef enum noise_type_t {
    NOISE_NONE,
    NOISE_DEPOLARIZING,
    NOISE_AMPLITUDE_DAMPING,
    NOISE_PHASE_DAMPING,
    NOISE_THERMAL,
    NOISE_CUSTOM
} NoiseType;

// Forward declaration for hierarchical matrix (from hierarchical_matrix.h)
struct HierarchicalMatrix;

// Noise model parameters
typedef struct noise_model_t {
    NoiseType type;
    double gate_error_rate;
    double measurement_error_rate;
    double decoherence_rate;
    double* custom_parameters;
    size_t num_custom_parameters;
    // Hierarchical noise representation for O(log n) operations
    size_t size;
    struct HierarchicalMatrix* h_matrix;
} NoiseModel;

// Use MitigationType from quantum_hardware_types.h

// Error mitigation parameters
typedef struct {
    MitigationType type;
    uint32_t num_samples;
    double scale_factors[4];
    void* custom_parameters;
} MitigationParams;

// Use SimulatorConfig from quantum_hardware_types.h

// Simulator state
typedef struct {
    double complex* amplitudes;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool* classical_bits;
    double fidelity;
    double error_rate;
    NoiseModel active_noise;
    MitigationParams active_mitigation;
    void* device_ptr;      // For GPU acceleration
    void* custom_state;    // For backend-specific data
} SimulatorState;

// Initialize simulator
SimulatorState* sim_init(uint32_t num_qubits, uint32_t num_classical_bits, const struct SimulatorConfig* config);

// Create quantum circuit for simulator
SimulatorCircuit* sim_create_circuit(uint32_t num_qubits, uint32_t num_classical_bits);

// Add gate to simulator circuit
bool sim_add_gate(SimulatorCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, double* parameters);

// Add controlled gate to simulator circuit
bool sim_add_controlled_gate(SimulatorCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, uint32_t control2, double* parameters);

// Execute circuit on simulator
bool sim_execute_circuit(SimulatorState* state, const SimulatorCircuit* circuit);

// Measure qubit
bool sim_measure_qubit(SimulatorState* state, uint32_t qubit, uint32_t classical_bit);

// Measure all qubits
bool sim_measure_all(SimulatorState* state);

// Get measurement results
bool* sim_get_measurement_results(const SimulatorState* state);

// Get measurement counts
uint64_t* sim_get_measurement_counts(const SimulatorState* state, uint32_t shots);

// Get state vector
double complex* sim_get_statevector(const SimulatorState* state);

// Get density matrix
double complex* sim_get_density_matrix(const SimulatorState* state);

// Get expectation value
double sim_get_expectation_value(const SimulatorState* state, const char* observable);

// Apply noise to state
bool sim_apply_noise(SimulatorState* state, const NoiseModel* noise);

// Apply error mitigation for simulator
bool sim_apply_error_mitigation(SimulatorState* state, const MitigationParams* params);

// Reset simulator state
void sim_reset_state(SimulatorState* state);

// Free simulator resources
void sim_cleanup(SimulatorState* state);

// Free circuit resources
void sim_cleanup_circuit(SimulatorCircuit* circuit);

// Utility functions for simulator
bool sim_validate_circuit(const SimulatorCircuit* circuit);
bool sim_optimize_circuit(SimulatorCircuit* circuit);
char* sim_circuit_to_string(const SimulatorCircuit* circuit);
double sim_get_circuit_depth(const SimulatorCircuit* circuit);
double sim_estimate_runtime(const SimulatorCircuit* circuit);
bool sim_save_circuit(const SimulatorCircuit* circuit, const char* filename);
SimulatorCircuit* sim_load_circuit(const char* filename);

#endif // QUANTUM_SIMULATOR_H
