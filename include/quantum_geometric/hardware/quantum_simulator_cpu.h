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

// ============================================================================
// Circuit Types for CPU Simulator
// ============================================================================

// Use struct QuantumCircuit from quantum_hardware_abstraction.h
// This is the hardware-level circuit representation
#ifndef CPU_SIM_CIRCUIT_DEFINED
#define CPU_SIM_CIRCUIT_DEFINED
typedef struct QuantumCircuit CPUSimCircuit;
#endif

// ============================================================================
// State Initialization
// ============================================================================

// Initialize simulator state to |0...0⟩
void init_simulator_state(double complex* state, size_t n);

// ============================================================================
// Measurement Functions
// ============================================================================

int measure_qubit_cpu(double complex* state, size_t target_qubit, size_t n_qubits);
int measure_qubit_basic(double complex* state, size_t target_qubit, size_t n_qubits);

// ============================================================================
// Circuit Simulation
// ============================================================================

// Simulate a quantum circuit on CPU
void simulate_circuit_cpu(double complex* state,
                        const CPUSimCircuit* circuit,
                        size_t n_qubits);

// ============================================================================
// Circuit Configuration and Management (CPU simulator specific)
// ============================================================================

// Configure circuit optimization parameters
void configure_circuit_optimization(CPUSimCircuit* circuit,
                                 bool use_error_correction,
                                 bool use_tensor_networks,
                                 size_t cache_line_size);

// CPU simulator circuit management (prefixed to avoid conflicts)
CPUSimCircuit* cpu_sim_create_circuit(size_t max_gates);
void cpu_sim_add_gate(CPUSimCircuit* circuit, const QuantumGate* gate);
void cpu_sim_cleanup_circuit(CPUSimCircuit* circuit);
void cpu_sim_get_error_statistics(const CPUSimCircuit* circuit,
                                   double* avg_error_rate,
                                   double* max_error_rate);

// Backward compatibility aliases (use with caution - may conflict)
#define init_cpu_circuit cpu_sim_create_circuit
#define add_gate_to_circuit cpu_sim_add_gate
#define get_error_statistics cpu_sim_get_error_statistics

// ============================================================================
// CPU-compatible type definitions for stabilizer operations
// ============================================================================

// float2 type for CPU (matches Metal/CUDA float2)
#ifndef CPU_FLOAT2_DEFINED
#define CPU_FLOAT2_DEFINED
typedef struct {
    float x;
    float y;
} cpu_float2;
// Use cpu_float2 as float2 in this header
#ifndef __METAL_VERSION__
#ifndef float2
#define float2 cpu_float2
#endif
#endif
#endif

// CPU-compatible stabilizer qubit representation
#ifndef CPU_STABILIZER_QUBIT_DEFINED
#define CPU_STABILIZER_QUBIT_DEFINED
typedef struct {
    float state_real;      // Real part of qubit state
    float state_imag;      // Imaginary part of qubit state
    uint32_t index;        // Qubit index
    uint32_t flags;        // Status flags
} CPUStabilizerQubit;
// Use CPUStabilizerQubit as StabilizerQubit in this header
#ifndef StabilizerQubit
#define StabilizerQubit CPUStabilizerQubit
#endif
#endif

// ============================================================================
// Stabilizer Operations (CPU implementation)
// ============================================================================

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
