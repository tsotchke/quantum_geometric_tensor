/**
 * @file quantum_base_types.h
 * @brief Base type definitions for quantum computing operations
 *
 * This header provides the fundamental type definitions used throughout
 * the quantum geometric tensor library. It is the authoritative source
 * for gate types, hardware types, and state types. Other headers should
 * include this file for these definitions.
 */

#ifndef QUANTUM_BASE_TYPES_H
#define QUANTUM_BASE_TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Gate Types - Canonical Definition
// ============================================================================

/**
 * @brief Enumeration of quantum gate types
 *
 * This is the authoritative definition of gate types used throughout
 * the library. All code should use these values.
 */
typedef enum {
    // Basic gates
    GATE_TYPE_I = 0,       // Identity gate
    GATE_TYPE_X = 1,       // Pauli-X (NOT) gate
    GATE_TYPE_Y = 2,       // Pauli-Y gate
    GATE_TYPE_Z = 3,       // Pauli-Z gate
    GATE_TYPE_H = 4,       // Hadamard gate
    GATE_TYPE_S = 5,       // S (Phase) gate (sqrt(Z))
    GATE_TYPE_T = 6,       // T gate (fourth root of Z)
    GATE_TYPE_RX = 7,      // Rotation around X axis
    GATE_TYPE_RY = 8,      // Rotation around Y axis
    GATE_TYPE_RZ = 9,      // Rotation around Z axis
    GATE_TYPE_CNOT = 10,   // Controlled-NOT gate
    GATE_TYPE_CY = 11,     // Controlled-Y gate
    GATE_TYPE_CZ = 12,     // Controlled-Z gate
    GATE_TYPE_SWAP = 13,   // SWAP gate
    GATE_TYPE_CUSTOM = 14, // Custom unitary gate
    // Extended gates
    GATE_TYPE_U1 = 15,     // U1 gate (phase rotation)
    GATE_TYPE_U2 = 16,     // U2 gate
    GATE_TYPE_U3 = 17,     // U3 gate (general single qubit)
    GATE_TYPE_CCX = 18,    // Toffoli gate (CCX)
    GATE_TYPE_PHASE = 19,  // General phase gate
    GATE_TYPE_CSWAP = 20,  // Fredkin gate (controlled SWAP)
    GATE_TYPE_ISWAP = 21,  // iSWAP gate
    GATE_TYPE_CRX = 22,    // Controlled RX
    GATE_TYPE_CRY = 23,    // Controlled RY
    GATE_TYPE_CRZ = 24,    // Controlled RZ
    GATE_TYPE_CH = 25,     // Controlled Hadamard
    GATE_TYPE_SDG = 26,    // S-dagger gate
    GATE_TYPE_TDG = 27,    // T-dagger gate
    GATE_TYPE_ECR = 28,    // Echoed Cross-Resonance gate (IBM native)
    GATE_TYPE_SX = 29,     // Square root of X gate
    GATE_TYPE_MEASURE = 30,// Measurement operation (pseudo-gate)
    GATE_TYPE_RESET = 31,  // Reset operation (pseudo-gate)
    GATE_TYPE_BARRIER = 32,// Barrier operation (pseudo-gate)
    // Two-qubit rotation gates
    GATE_TYPE_XX = 33,     // XX (Ising) rotation gate
    GATE_TYPE_YY = 34,     // YY rotation gate
    GATE_TYPE_ZZ = 35      // ZZ rotation gate
} gate_type_t;

// Convenience aliases for backward compatibility
#ifndef GATE_ALIASES_DEFINED
#define GATE_ALIASES_DEFINED
#define GATE_I      GATE_TYPE_I
#define GATE_X      GATE_TYPE_X
#define GATE_Y      GATE_TYPE_Y
#define GATE_Z      GATE_TYPE_Z
#define GATE_H      GATE_TYPE_H
#define GATE_S      GATE_TYPE_S
#define GATE_T      GATE_TYPE_T
#define GATE_RX     GATE_TYPE_RX
#define GATE_RY     GATE_TYPE_RY
#define GATE_RZ     GATE_TYPE_RZ
#define GATE_CNOT   GATE_TYPE_CNOT
#define GATE_CX     GATE_TYPE_CNOT  // Alias for CNOT (CX = Controlled-X)
#define GATE_CY     GATE_TYPE_CY
#define GATE_CZ     GATE_TYPE_CZ
#define GATE_SWAP   GATE_TYPE_SWAP
#define GATE_CUSTOM GATE_TYPE_CUSTOM
#define GATE_U1     GATE_TYPE_U1
#define GATE_U2     GATE_TYPE_U2
#define GATE_U3     GATE_TYPE_U3
#define GATE_CCX    GATE_TYPE_CCX
#define GATE_TOFFOLI GATE_TYPE_CCX  // Alias for Toffoli
#define GATE_PHASE  GATE_TYPE_PHASE
#define GATE_CSWAP  GATE_TYPE_CSWAP
#define GATE_ISWAP  GATE_TYPE_ISWAP
#define GATE_CRX    GATE_TYPE_CRX
#define GATE_CRY    GATE_TYPE_CRY
#define GATE_CRZ    GATE_TYPE_CRZ
#define GATE_CH     GATE_TYPE_CH
#define GATE_SDG    GATE_TYPE_SDG
#define GATE_TDG    GATE_TYPE_TDG
#define GATE_ECR    GATE_TYPE_ECR
#define GATE_SX     GATE_TYPE_SX
#define GATE_MEASURE GATE_TYPE_MEASURE
#define GATE_RESET  GATE_TYPE_RESET
#define GATE_BARRIER GATE_TYPE_BARRIER
#define GATE_XX     GATE_TYPE_XX
#define GATE_YY     GATE_TYPE_YY
#define GATE_ZZ     GATE_TYPE_ZZ
#endif

// ============================================================================
// Hardware Type Definitions
// ============================================================================

/**
 * @brief Hardware platform types
 *
 * Includes both compute hardware types (CPU, GPU, etc.) and
 * quantum backend providers (IBM, Rigetti, etc.)
 */
typedef enum {
    // Local compute hardware
    HARDWARE_TYPE_CPU,
    HARDWARE_TYPE_GPU,
    HARDWARE_TYPE_QPU,
    HARDWARE_TYPE_SIMULATOR,
    HARDWARE_TYPE_METAL,
    HARDWARE_TYPE_CUDA,
    HARDWARE_TYPE_FPGA,
    // Quantum backend providers
    HARDWARE_TYPE_IBM,
    HARDWARE_TYPE_RIGETTI,
    HARDWARE_TYPE_DWAVE,
    HARDWARE_TYPE_CUSTOM
} HardwareType;

// Backward compatibility aliases for HARDWARE_BACKEND_* naming
#define HARDWARE_BACKEND_CPU       HARDWARE_TYPE_CPU
#define HARDWARE_BACKEND_GPU       HARDWARE_TYPE_GPU
#define HARDWARE_BACKEND_METAL     HARDWARE_TYPE_METAL
#define HARDWARE_BACKEND_CUDA      HARDWARE_TYPE_CUDA
#define HARDWARE_BACKEND_FPGA      HARDWARE_TYPE_FPGA
#define HARDWARE_BACKEND_QPU       HARDWARE_TYPE_QPU
#define HARDWARE_BACKEND_IBM       HARDWARE_TYPE_IBM
#define HARDWARE_BACKEND_RIGETTI   HARDWARE_TYPE_RIGETTI
#define HARDWARE_BACKEND_DWAVE     HARDWARE_TYPE_DWAVE
#define HARDWARE_BACKEND_CUSTOM    HARDWARE_TYPE_CUSTOM
#define HARDWARE_BACKEND_SIMULATOR HARDWARE_TYPE_SIMULATOR

// Note: HARDWARE_IBM, HARDWARE_RIGETTI, HARDWARE_DWAVE, HARDWARE_CUSTOM
// are also defined in quantum_hardware_types.h (HardwareBackendType)
// for the hardware abstraction layer

// ============================================================================
// Quantum State Types
// ============================================================================

/**
 * @brief Types of quantum states
 */
typedef enum {
    QUANTUM_STATE_PURE,          // Pure quantum state |ψ⟩
    QUANTUM_STATE_MIXED,         // Mixed quantum state (density matrix)
    QUANTUM_STATE_THERMAL,       // Thermal equilibrium state
    QUANTUM_STATE_COHERENT,      // Coherent state
    QUANTUM_STATE_SQUEEZED,      // Squeezed state
    QUANTUM_STATE_ENTANGLED,     // Entangled state
    QUANTUM_STATE_CUSTOM         // Custom state type
} quantum_state_type_t;

// ============================================================================
// Complex Number Type
// ============================================================================

#ifndef COMPLEX_FLOAT_DEFINED
#define COMPLEX_FLOAT_DEFINED

/**
 * @brief Complex number representation (single precision)
 */
typedef struct {
    float real;
    float imag;
} ComplexFloat;

/**
 * @brief Double-precision complex number
 */
typedef struct {
    double real;
    double imag;
} ComplexDouble;

#endif // COMPLEX_FLOAT_DEFINED

// ============================================================================
// Maximum Limits
// ============================================================================

#ifndef MAX_QUBITS
#define MAX_QUBITS 64
#endif

#ifndef MAX_GATE_PARAMS
#define MAX_GATE_PARAMS 4
#endif

#ifndef MAX_GATE_QUBITS
#define MAX_GATE_QUBITS 4
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

struct quantum_gate_t;
struct quantum_circuit_t;
struct quantum_state_t;
struct quantum_system_t;

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_BASE_TYPES_H
