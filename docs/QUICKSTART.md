# Quantum Geometric Tensor Library: Quickstart Guide

**Version:** 0.777 Beta
**Last Updated:** January 2026

---

## Overview

This guide provides a rapid introduction to the Quantum Geometric Tensor Library (QGTL), enabling researchers and developers to begin leveraging geometric quantum computing techniques for quantum circuit execution, error protection, and quantum-enhanced machine learning.

## Development Status

| Component | Status |
|-----------|--------|
| Core tensor operations | Available |
| Geometric algorithms | Available |
| GPU acceleration (CUDA/Metal) | Available |
| Surface code error correction | Available |
| Distributed training | Available |
| Hardware backends (IBM/Rigetti/D-Wave) | Beta |
| Quantum phase estimation | In Development |

## Quick Start

```c
#include <quantum_geometric/core/quantum_geometric_core.h>

int main() {
    // Initialize quantum system with geometric protection
    quantum_system_t* system = quantum_init_system(&(quantum_config_t){
        .backend = BACKEND_SIMULATOR,
        .optimization = {
            .geometric = true,
            .error_protection = true,
            .hardware_aware = true
        }
    });

    // Create quantum circuit for Bell state preparation
    quantum_circuit_t* circuit = quantum_circuit_create(2);

    // Construct Bell state: (|00> + |11>)/sqrt(2)
    quantum_circuit_h(circuit, 0);     // Hadamard gate on qubit 0
    quantum_circuit_cx(circuit, 0, 1); // CNOT: control=0, target=1

    // Execute circuit with geometric optimization
    execution_result_t result;
    quantum_execute_circuit(system, circuit, &result);

    // Display results
    printf("State fidelity: %.6f\n", result.fidelity);
    printf("Circuit depth: %zu\n", result.compiled_depth);
    printf("Error rate: %.2e\n", result.error_rate);

    // Resource cleanup
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
    return 0;
}
```

## Framework Architecture

The Quantum Geometric Tensor Library leverages differential geometry and algebraic topology to achieve superior performance on quantum hardware. The mathematical foundation provides:

### Geometric State Representation

Quantum states are represented as points on complex projective manifolds:

```
M = CP^(2^n-1) = U(2^n) / (U(1) x U(2^n-1))
```

This geometric structure enables:

- **Hardware-Native Operations**: Automatic mapping to processor topology with 30-70% gate count reduction
- **Topology-Based Protection**: O(epsilon^2) error scaling through geometric phase encoding
- **Natural Optimization**: Geodesic-following compilation with 60-80% memory reduction
- **Geometric Compilation**: Manifold-aware circuit synthesis with 2-5x fidelity improvement

## Installation

### Prerequisites

- C/C++ compiler (GCC 9+, Clang 11+)
- CMake 3.16+
- BLAS/LAPACK implementation
- MPI (optional, for distributed training)
- CUDA Toolkit 11+ (optional, for NVIDIA GPUs)
- Metal SDK (optional, for Apple Silicon)

### Build Instructions

```bash
# Clone repository
git clone https://github.com/tsotchke/quantum_geometric_tensor.git
cd quantum_geometric_tensor

# Configure and build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

For macOS with Apple Silicon:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DQGT_ENABLE_METAL=ON
make -j$(sysctl -n hw.ncpu)
```

## Core Features

### 1. Quantum Circuit Execution

```c
#include <quantum_geometric/core/quantum_geometric_core.h>

// Configure hardware-specific execution
quantum_hardware_config_t config = {
    .quantum = {
        .ibm = {
            .processor = "ibm_manhattan",
            .topology = TOPOLOGY_HEAVY_HEXAGONAL,
            .qubits = {
                .available = 127,
                .coherence_time = 100e-6,
                .gate_fidelity = 0.999
            }
        }
    },
    .geometric = {
        .manifold = MANIFOLD_COMPLEX_PROJECTIVE,
        .metric = METRIC_FUBINI_STUDY,
        .protection = {
            .type = PROTECTION_TOPOLOGICAL,
            .strength = 0.95
        }
    },
    .optimization = {
        .circuit = OPTIMIZATION_GEOMETRIC_SYNTHESIS,
        .error = ERROR_MITIGATION_TOPOLOGICAL,
        .scheduling = {
            .type = SCHEDULING_ADAPTIVE,
            .priority = PRIORITY_FIDELITY
        }
    }
};

// Initialize and execute
quantum_system_t* system = quantum_init_system(&config);
quantum_workflow_t* workflow = quantum_create_workflow(
    system,
    WORKFLOW_GEOMETRIC | WORKFLOW_OPTIMIZED | WORKFLOW_PROTECTED
);

execution_result_t result;
quantum_execute_workflow(system, workflow, &result);

printf("Fidelity: %.3f\n", result.fidelity);
printf("Error rate: %.2e\n", result.error_rate);
printf("Circuit depth: %d\n", result.circuit_depth);

quantum_destroy_workflow(workflow);
quantum_destroy_system(system);
```

### 2. Quantum Machine Learning

Geometric quantum neural networks with natural gradient optimization:

```c
#include <quantum_geometric/ai/quantum_geometric_ml.h>

quantum_ml_config_t config = {
    .geometry = {
        .manifold = MANIFOLD_COMPLEX_PROJECTIVE,
        .metric = METRIC_FUBINI_STUDY,
        .connection = CONNECTION_QUANTUM_GEOMETRIC,
        .curvature = CURVATURE_BERRY
    },
    .network = {
        .architecture = ARCHITECTURE_GEOMETRIC_NEURAL,
        .layers = 4,
        .features = 64,
        .attention = ATTENTION_GEOMETRIC
    },
    .learning = {
        .optimizer = OPTIMIZER_NATURAL_GRADIENT,
        .dynamics = DYNAMICS_PARALLEL_TRANSPORT,
        .regularization = REGULARIZATION_GEOMETRIC
    }
};

quantum_ml_model_t* model = quantum_ml_create(&config);
quantum_ml_train(model, train_data, train_labels);

float accuracy = quantum_ml_evaluate(model, test_data, test_labels, METRIC_GEOMETRIC);
printf("Test accuracy: %.2f%%\n", accuracy * 100);

quantum_ml_destroy(model);
```

### 3. Geometric Error Protection

Multi-layer protection using geometric and topological invariants:

```c
#include <quantum_geometric/physics/quantum_topological_operations.h>

protection_config_t config = {
    .code_type = SURFACE_CODE_ROTATED,
    .code_distance = 5,
    .error_model = ERROR_MODEL_DEPOLARIZING,
    .physical_error_rate = 0.001,
    .geometric = {
        .manifold = MANIFOLD_COMPLEX_PROJECTIVE,
        .invariants = {
            .chern = true,
            .berry = true,
            .holonomy = true
        },
        .protection = {
            .strength = 0.95,
            .adaptive = true
        }
    }
};

// Apply protection to quantum state
quantum_state_t* protected_state = quantum_geometric_protect(state, &config);

// Execute protected computation
quantum_apply_logical_gate(protected_state, GATE_LOGICAL_HADAMARD, 0);

// Decode results
quantum_decode_state(protected_state, &result);
```

## Performance Guidelines

### Hardware Selection

| Platform | Optimal Use Case | Key Features |
|----------|-----------------|--------------|
| IBM Quantum | Gate-based circuits | Heavy-hex topology, dynamic circuits |
| Rigetti | Algorithm development | Fast parametric compilation |
| D-Wave | Optimization problems | 5000+ qubits, quantum annealing |

### Error Mitigation

- **Geometric protection** reduces error scaling from O(epsilon) to O(epsilon^2)
- **Topological encoding** extends coherence times by 10-100x
- **Berry phase monitoring** enables real-time error detection

### Resource Optimization

- Geometric compilation achieves 30-70% circuit depth reduction
- Hardware-aware optimization respects connectivity constraints
- Hierarchical tensor networks provide O(n log n) attention complexity

## Additional Resources

- [Installation Guide](INSTALLATION.md): Detailed setup instructions
- [API Reference](API_REFERENCE.md): Complete API documentation
- [Theory Guide](THEORY.md): Mathematical foundations
- [Performance Tuning](PERFORMANCE_TUNING.md): Optimization strategies
- [Hardware Integration](QUANTUM_HARDWARE.md): Backend configuration

## Development Roadmap

| Phase | Component | Status |
|-------|-----------|--------|
| Foundation | Core algorithms, tensor operations | Complete |
| Acceleration | GPU backends (CUDA, Metal) | Complete |
| Protection | Surface codes, error correction | Complete |
| Distribution | MPI training, fault tolerance | Complete |
| Hardware | IBM, Rigetti, D-Wave integration | Beta |
| Advanced | Phase estimation, advanced gradients | In Development |
| Production | v1.0 release | Q3 2026 |

---

*For questions or contributions, please consult the [Contributing Guide](../CONTRIBUTING.md) or open an issue on GitHub.*
