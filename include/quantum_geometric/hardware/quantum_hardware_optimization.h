#ifndef QUANTUM_HARDWARE_OPTIMIZATION_H
#define QUANTUM_HARDWARE_OPTIMIZATION_H

#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Forward declarations for quantum circuit (struct tag required)
struct QuantumCircuit;

// IonQ configuration (if not defined elsewhere)
#ifndef IONQ_CONFIG_DEFINED
#define IONQ_CONFIG_DEFINED
typedef struct IonQConfig {
    size_t num_qubits;
    double gate_fidelity;
    double measurement_fidelity;
    double* coupling_map;
    bool use_native_gates;
    char api_key[256];
    char device_name[64];
} IonQConfig;
#endif

// Hardware optimization types
typedef struct {
    void (*ibm_optimize)(struct QuantumCircuit*, const IBMBackendConfig*);
    void (*rigetti_optimize)(struct QuantumCircuit*, const struct RigettiConfig*);
    void (*ionq_optimize)(struct QuantumCircuit*, const IonQConfig*);
    void (*dwave_optimize)(struct QuantumCircuit*, const struct DWaveConfig*);
} HardwareOptimizations;

// Initialize hardware-specific optimizations
HardwareOptimizations* init_hardware_optimizations(const char* backend_type);

// Clean up hardware optimizations
void cleanup_hardware_optimizations(HardwareOptimizations* opts);

#endif // QUANTUM_HARDWARE_OPTIMIZATION_H
