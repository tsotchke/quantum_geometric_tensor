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

// Compatibility aliases - only define if not already defined by quantum_base_types.h
#ifndef GATE_I
#define GATE_I      OPT_GATE_I
#endif
#ifndef GATE_X
#define GATE_X      OPT_GATE_X
#endif
#ifndef GATE_Y
#define GATE_Y      OPT_GATE_Y
#endif
#ifndef GATE_Z
#define GATE_Z      OPT_GATE_Z
#endif
#ifndef GATE_H
#define GATE_H      OPT_GATE_H
#endif
#ifndef GATE_S
#define GATE_S      OPT_GATE_S
#endif
#ifndef GATE_T
#define GATE_T      OPT_GATE_T
#endif
#ifndef GATE_RX
#define GATE_RX     OPT_GATE_RX
#endif
#ifndef GATE_RY
#define GATE_RY     OPT_GATE_RY
#endif
#ifndef GATE_RZ
#define GATE_RZ     OPT_GATE_RZ
#endif
#ifndef GATE_CX
#define GATE_CX     OPT_GATE_CX
#endif
#ifndef GATE_CY
#define GATE_CY     OPT_GATE_CY
#endif
#ifndef GATE_CZ
#define GATE_CZ     OPT_GATE_CZ
#endif
#ifndef GATE_SWAP
#define GATE_SWAP   OPT_GATE_SWAP
#endif
#ifndef GATE_ISWAP
#define GATE_ISWAP  OPT_GATE_ISWAP
#endif
#ifndef GATE_U1
#define GATE_U1     OPT_GATE_U1
#endif
#ifndef GATE_U2
#define GATE_U2     OPT_GATE_U2
#endif
#ifndef GATE_U3
#define GATE_U3     OPT_GATE_U3
#endif

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
// Optimized Circuit for this module
// ============================================================================

// Local circuit optimization result type (distinct from global OptimizedCircuit)
typedef struct CircuitOptResult {
    CircuitOptGate* gates;
    size_t num_gates;
    size_t capacity;
    double estimated_fidelity;
    double estimated_depth;
    size_t num_qubits;
} CircuitOptResult;

// Compatibility: define OptimizedCircuit as CircuitOptResult only if not already defined
#ifndef QUANTUM_HARDWARE_TYPES_H
typedef CircuitOptResult OptimizedCircuit;
#endif

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
CircuitOptResult* create_optimized_circuit(size_t num_qubits, size_t initial_capacity);
void destroy_optimized_circuit(CircuitOptResult* circuit);

// Add gates to circuit
bool add_optimized_gate(CircuitOptResult* circuit, const CircuitOptGate* gate);

// Circuit optimization functions
CircuitOptResult* optimize_circuit_for_rigetti(
    const CircuitOptResult* input,
    const CircuitOptRigettiConfig* config);

CircuitOptResult* optimize_gate_sequence(
    const CircuitOptResult* input,
    const CircuitOptRigettiConfig* config);

// Decomposition functions
CircuitOptResult* decompose_to_native_gates(
    const CircuitOptResult* input,
    const CircuitOptRigettiConfig* config);

// Fidelity estimation (local version for this module)
double circuit_opt_estimate_fidelity(
    const CircuitOptResult* circuit,
    const CircuitOptRigettiConfig* config);

// Circuit depth calculation
size_t calculate_circuit_depth(const CircuitOptResult* circuit);

// Gate count by type
size_t count_gates_of_type(const CircuitOptResult* circuit, OptGateType type);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_CIRCUIT_OPTIMIZATION_H
