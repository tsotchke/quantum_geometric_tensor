#ifndef QUANTUM_GEOMETRIC_INTERFACE_H
#define QUANTUM_GEOMETRIC_INTERFACE_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/geometric_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interface constants
#define QG_VECTOR_SIZE 64
#define QG_MAX_IBM_QUBITS 127
#define QG_MAX_CIRCUIT_DEPTH 1000
#define QG_MAX_MEASUREMENT_SHOTS 8192
#define QG_MAX_OPTIMIZATION_ITERATIONS 1000
#define QG_INTERFACE_ERROR_THRESHOLD 1e-6

// Hardware type for interface (extends HardwareType)
#define HARDWARE_NONE 0

// Operation types for quantum geometric operations
typedef enum {
    PARALLEL_TRANSPORT,
    GEODESIC_EVOLUTION,
    CURVATURE_ESTIMATION,
    HOLONOMY_COMPUTATION,
    METRIC_TENSOR_EVAL,
    CONNECTION_EVAL
} OperationType;

// State properties for quantum states
typedef struct {
    double fidelity;
    double purity;
    double entropy;
    bool is_entangled;
    bool is_geometric;
} StateProperties;

// Note: QuantumGate is defined in quantum_hardware_types.h
// This interface uses the standard QuantumGate from that header

// Manifold is defined in geometric_attention.h

// Main quantum geometric interface structure
typedef struct {
    HardwareType hardware_type;
    bool is_available;
    size_t num_qubits;
    double error_rate;
    StateProperties state_props;
    void* backend_handle;
} QuantumGeometricInterface;

// Interface creation and destruction
QuantumGeometricInterface* init_quantum_interface(void);
void cleanup_quantum_interface(QuantumGeometricInterface* interface);

// State operations
StateProperties* create_quantum_state(QuantumGeometricInterface* interface, size_t num_qubits);
void destroy_quantum_state(StateProperties* state);

// Geometric operations
double* apply_geometric_operation(QuantumGeometricInterface* interface,
                                  Manifold* manifold,
                                  OperationType op_type);

// Measurement
double* measure_expectation_value(QuantumGeometricInterface* interface);

// Optimization
double* run_variational_optimization(QuantumGeometricInterface* interface);

// Hardware detection
HardwareType detect_quantum_hardware(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_INTERFACE_H
