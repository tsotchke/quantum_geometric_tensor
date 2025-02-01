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
#define MAX_QUBITS 32

// Use GateType from quantum_circuit.h

// Use types from quantum_hardware_types.h
typedef struct quantum_gate_t QuantumGate;
typedef struct quantum_circuit_t QuantumCircuit;

// Noise model types
typedef enum noise_type_t {
    NOISE_NONE,
    NOISE_DEPOLARIZING,
    NOISE_AMPLITUDE_DAMPING,
    NOISE_PHASE_DAMPING,
    NOISE_THERMAL,
    NOISE_CUSTOM
} NoiseType;

// Noise model parameters
typedef struct noise_model_t {
    NoiseType type;
    double gate_error_rate;
    double measurement_error_rate;
    double decoherence_rate;
    double* custom_parameters;
    size_t num_custom_parameters;
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
SimulatorState* init_simulator(uint32_t num_qubits, uint32_t num_classical_bits, const SimulatorConfig* config);

// Create quantum circuit
QuantumCircuit* create_circuit(uint32_t num_qubits, uint32_t num_classical_bits);

// Add gate to circuit
bool add_gate(QuantumCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, double* parameters);

// Add controlled gate
bool add_controlled_gate(QuantumCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, uint32_t control2, double* parameters);

// Execute circuit on simulator
bool execute_circuit(SimulatorState* state, const QuantumCircuit* circuit);

// Measure qubit
bool measure_qubit(SimulatorState* state, uint32_t qubit, uint32_t classical_bit);

// Measure all qubits
bool measure_all(SimulatorState* state);

// Get measurement results
bool* get_measurement_results(const SimulatorState* state);

// Get measurement counts
uint64_t* get_measurement_counts(const SimulatorState* state, uint32_t shots);

// Get state vector
double complex* get_statevector(const SimulatorState* state);

// Get density matrix
double complex* get_density_matrix(const SimulatorState* state);

// Get expectation value
double get_expectation_value(const SimulatorState* state, const char* observable);

// Apply noise to state
bool apply_noise(SimulatorState* state, const NoiseModel* noise);

// Apply error mitigation for simulator
bool apply_simulator_error_mitigation(SimulatorState* state, const MitigationParams* params);

// Reset simulator state
void reset_state(SimulatorState* state);

// Free simulator resources
void cleanup_simulator(SimulatorState* state);

// Free circuit resources
void cleanup_circuit(QuantumCircuit* circuit);

// Utility functions
bool validate_circuit(const QuantumCircuit* circuit);
bool optimize_circuit(QuantumCircuit* circuit);
char* circuit_to_string(const QuantumCircuit* circuit);
double get_circuit_depth(const QuantumCircuit* circuit);
double estimate_runtime(const QuantumCircuit* circuit);
bool save_circuit(const QuantumCircuit* circuit, const char* filename);
QuantumCircuit* load_circuit(const char* filename);

#endif // QUANTUM_SIMULATOR_H
