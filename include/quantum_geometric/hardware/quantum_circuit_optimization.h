#ifndef QUANTUM_CIRCUIT_OPTIMIZATION_H
#define QUANTUM_CIRCUIT_OPTIMIZATION_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Gate Types for Circuit Optimization
// ============================================================================

typedef enum {
    OPT_GATE_I,         // Identity
    OPT_GATE_X,         // Pauli-X
    OPT_GATE_Y,         // Pauli-Y
    OPT_GATE_Z,         // Pauli-Z
    OPT_GATE_H,         // Hadamard
    OPT_GATE_S,         // Phase gate
    OPT_GATE_T,         // T gate
    OPT_GATE_RX,        // Rotation around X
    OPT_GATE_RY,        // Rotation around Y
    OPT_GATE_RZ,        // Rotation around Z
    OPT_GATE_CX,        // Controlled-X (CNOT)
    OPT_GATE_CY,        // Controlled-Y
    OPT_GATE_CZ,        // Controlled-Z
    OPT_GATE_SWAP,      // SWAP gate
    OPT_GATE_ISWAP,     // iSWAP gate
    OPT_GATE_U1,        // Single parameter unitary
    OPT_GATE_U2,        // Two parameter unitary
    OPT_GATE_U3,        // Three parameter unitary
    OPT_GATE_CUSTOM     // Custom gate
} OptGateType;

// Compatibility aliases for the source file
#define GATE_I      OPT_GATE_I
#define GATE_X      OPT_GATE_X
#define GATE_Y      OPT_GATE_Y
#define GATE_Z      OPT_GATE_Z
#define GATE_H      OPT_GATE_H
#define GATE_S      OPT_GATE_S
#define GATE_T      OPT_GATE_T
#define GATE_RX     OPT_GATE_RX
#define GATE_RY     OPT_GATE_RY
#define GATE_RZ     OPT_GATE_RZ
#define GATE_CX     OPT_GATE_CX
#define GATE_CY     OPT_GATE_CY
#define GATE_CZ     OPT_GATE_CZ
#define GATE_SWAP   OPT_GATE_SWAP
#define GATE_ISWAP  OPT_GATE_ISWAP
#define GATE_U1     OPT_GATE_U1
#define GATE_U2     OPT_GATE_U2
#define GATE_U3     OPT_GATE_U3

// Optimized gate structure for circuit optimization
typedef struct {
    OptGateType type;
    size_t target;              // Target qubit
    size_t control;             // Control qubit (for 2-qubit gates)
    double parameter;           // Gate parameter (angle)
    double parameters[3];       // Additional parameters for U2, U3
    double fidelity;            // Estimated gate fidelity
} CircuitOptGate;

// Use CircuitOptGate as QuantumGate within this module only
// Note: This is a module-local type, not the global QuantumGate
#define QuantumGate CircuitOptGate

// ============================================================================
// Hardware Configuration for Circuit Optimization
// ============================================================================

// Circuit optimization specific config (avoids conflict with quantum_backend_types.h)
typedef struct CircuitOptRigettiConfig {
    size_t num_qubits;
    double* crosstalk_map;      // num_qubits x num_qubits crosstalk matrix
    double* gate_errors;        // Gate error rates
    double* readout_errors;     // Readout error rates
    double t1_time;             // T1 decoherence time (us)
    double t2_time;             // T2 decoherence time (us)
    double gate_time;           // Typical gate time (us)
    size_t* connectivity;       // Qubit connectivity map
    size_t connectivity_count;  // Number of connections
} CircuitOptRigettiConfig;

// Alias for source compatibility
#define RigettiConfig CircuitOptRigettiConfig

// ============================================================================
// Optimized Circuit
// ============================================================================

typedef struct {
    CircuitOptGate* gates;
    size_t num_gates;
    size_t capacity;
    double estimated_fidelity;
    double estimated_depth;
    size_t num_qubits;
} OptimizedCircuit;

// Alias for QuantumCircuit
typedef OptimizedCircuit QuantumCircuit;

// ============================================================================
// Helper Macros
// ============================================================================

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// API Functions
// ============================================================================

// Create and destroy optimized circuits
OptimizedCircuit* create_optimized_circuit(size_t num_qubits, size_t initial_capacity);
void destroy_optimized_circuit(OptimizedCircuit* circuit);

// Add gates to circuit
bool add_optimized_gate(OptimizedCircuit* circuit, const CircuitOptGate* gate);

// Circuit optimization functions
OptimizedCircuit* optimize_circuit_for_rigetti(
    const OptimizedCircuit* input,
    const CircuitOptRigettiConfig* config);

OptimizedCircuit* optimize_gate_sequence(
    const OptimizedCircuit* input,
    const CircuitOptRigettiConfig* config);

// Decomposition functions
OptimizedCircuit* decompose_to_native_gates(
    const OptimizedCircuit* input,
    const CircuitOptRigettiConfig* config);

// Fidelity estimation
double estimate_circuit_fidelity(
    const OptimizedCircuit* circuit,
    const CircuitOptRigettiConfig* config);

// Circuit depth calculation
size_t calculate_circuit_depth(const OptimizedCircuit* circuit);

// Gate count by type
size_t count_gates_of_type(const OptimizedCircuit* circuit, OptGateType type);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_CIRCUIT_OPTIMIZATION_H
