#ifndef QUANTUM_HARDWARE_OPTIMIZATION_H
#define QUANTUM_HARDWARE_OPTIMIZATION_H

#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Hardware optimization types
typedef struct {
    void (*ibm_optimize)(QuantumCircuit*, const IBMBackendConfig*);
    void (*rigetti_optimize)(QuantumCircuit*, const RigettiConfig*);
    void (*ionq_optimize)(QuantumCircuit*, const IonQConfig*);
    void (*dwave_optimize)(QuantumCircuit*, const DWaveConfig*);
} HardwareOptimizations;

// Initialize hardware-specific optimizations
HardwareOptimizations* init_hardware_optimizations(const char* backend_type);

// Clean up hardware optimizations
void cleanup_hardware_optimizations(HardwareOptimizations* opts);

#endif // QUANTUM_HARDWARE_OPTIMIZATION_H
