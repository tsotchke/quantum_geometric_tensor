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
    GATE_TYPE_CZ = 11,     // Controlled-Z gate
    GATE_TYPE_SWAP = 12,   // SWAP gate
    GATE_TYPE_CUSTOM = 13  // Custom unitary gate
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
#define GATE_CZ     GATE_TYPE_CZ
#define GATE_SWAP   GATE_TYPE_SWAP
#define GATE_CUSTOM GATE_TYPE_CUSTOM

// Extended gate types for special operations
#define GATE_U1     14     // U1 gate (phase rotation)
#define GATE_U2     15     // U2 gate
#define GATE_U3     16     // U3 gate (general single qubit)
#define GATE_CCX    17     // Toffoli gate (CCX)
#define GATE_TOFFOLI GATE_CCX  // Alias for Toffoli
#define GATE_PHASE  18     // General phase gate
#define GATE_CSWAP  19     // Fredkin gate (controlled SWAP)
#define GATE_ISWAP  20     // iSWAP gate
#define GATE_CRX    21     // Controlled RX
#define GATE_CRY    22     // Controlled RY
#define GATE_CRZ    23     // Controlled RZ
#define GATE_CH     24     // Controlled Hadamard
#define GATE_SDG    25     // S-dagger gate
#define GATE_TDG    26     // T-dagger gate
#define GATE_ECR    27     // Echoed Cross-Resonance gate (IBM native)
#define GATE_SX     28     // Square root of X gate
#define GATE_MEASURE 29    // Measurement operation (pseudo-gate)
#define GATE_RESET  30     // Reset operation (pseudo-gate)
#define GATE_BARRIER 31    // Barrier operation (pseudo-gate)
#endif

// ============================================================================
// Hardware Type Definitions
// ============================================================================

/**
 * @brief Hardware platform types
 */
typedef enum {
    HARDWARE_TYPE_CPU,
    HARDWARE_TYPE_GPU,
    HARDWARE_TYPE_QPU,
    HARDWARE_TYPE_SIMULATOR,
    HARDWARE_TYPE_METAL,
    HARDWARE_TYPE_CUDA,
    HARDWARE_TYPE_FPGA
} HardwareType;

// Backward compatibility aliases for HARDWARE_BACKEND_* naming
#define HARDWARE_BACKEND_CPU       HARDWARE_TYPE_CPU
#define HARDWARE_BACKEND_GPU       HARDWARE_TYPE_GPU
#define HARDWARE_BACKEND_METAL     HARDWARE_TYPE_METAL
#define HARDWARE_BACKEND_CUDA      HARDWARE_TYPE_CUDA
#define HARDWARE_BACKEND_FPGA      HARDWARE_TYPE_FPGA
#define HARDWARE_BACKEND_QPU       HARDWARE_TYPE_QPU
#define HARDWARE_BACKEND_IBM       100  // Distinct value for IBM backend
#define HARDWARE_BACKEND_RIGETTI   101  // Distinct value for Rigetti backend
#define HARDWARE_BACKEND_DWAVE     102  // Distinct value for D-Wave backend
#define HARDWARE_BACKEND_CUSTOM    103  // Distinct value for custom backend
#define HARDWARE_BACKEND_SIMULATOR HARDWARE_TYPE_SIMULATOR

// Note: HARDWARE_IBM, HARDWARE_RIGETTI, HARDWARE_DWAVE, HARDWARE_CUSTOM
// are defined as enum values in quantum_hardware_types.h (HardwareBackendType)
// Do not redefine them here as macros

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
