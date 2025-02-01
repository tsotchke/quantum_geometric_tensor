#ifndef QUANTUM_SIMULATOR_CPU_H
#define QUANTUM_SIMULATOR_CPU_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_rng.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include <stdbool.h>
#include <complex.h>

// Initialize simulator state
void init_simulator_state(double complex* state, size_t n);

// Measurement functions
int measure_qubit_cpu(double complex* state, size_t target_qubit, size_t n_qubits);
int measure_qubit_basic(double complex* state, size_t target_qubit, size_t n_qubits);

// Quantum circuit simulation functions
void simulate_circuit_cpu(double complex* state,
                        const QuantumCircuit* circuit,
                        size_t n_qubits);

// Configure circuit optimization parameters
void configure_circuit_optimization(QuantumCircuit* circuit,
                                 bool use_error_correction,
                                 bool use_tensor_networks,
                                 size_t cache_line_size);

// Circuit management functions
QuantumCircuit* init_quantum_circuit(size_t max_gates);
void add_gate_to_circuit(QuantumCircuit* circuit, const QuantumGate* gate);
void cleanup_circuit(QuantumCircuit* circuit);
void get_error_statistics(const QuantumCircuit* circuit,
                         double* avg_error_rate,
                         double* max_error_rate);

// Stabilizer operations
int cpu_measure_stabilizers(
    const StabilizerQubit* qubits,
    const uint32_t* qubit_indices,
    const StabilizerConfig* config,
    float2* results,
    uint32_t num_stabilizers);

int cpu_apply_correction(
    StabilizerQubit* qubits,
    const uint32_t* qubit_indices,
    const StabilizerConfig* config,
    const float2* syndrome,
    uint32_t num_qubits);

int cpu_compute_correlations(
    const StabilizerQubit* qubits,
    const uint32_t* qubit_indices,
    const StabilizerConfig* configs,
    float* correlations,
    uint32_t num_stabilizers);

// Hardware capability detection
bool cpu_has_avx2(void);
bool cpu_has_avx512(void);
void cpu_init_simd(void);

#endif // QUANTUM_SIMULATOR_CPU_H
