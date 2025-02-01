# Quantum Geometric Learning Tutorials

Step-by-step guides to help you get started with quantum geometric learning. Each tutorial builds on the previous ones, gradually introducing more advanced concepts.

## Basic Tutorials

### 1. First Quantum Program

```c
// tutorial1.c - Creating and measuring quantum states
#include <quantum_geometric/core/quantum_geometric_core.h>

int main() {
    // Initialize a single qubit
    quantum_state* state = quantum_state_create(1);
    if (!state) {
        fprintf(stderr, "Failed to create quantum state\n");
        return 1;
    }

    // Create superposition state (|0⟩ + |1⟩)/√2
    ComplexFloat amplitudes[2] = {
        {1.0f/sqrt(2.0f), 0.0f},  // |0⟩ amplitude
        {1.0f/sqrt(2.0f), 0.0f}   // |1⟩ amplitude
    };
    quantum_state_set_amplitudes(state, amplitudes, 2);

    // Measure state
    float result = quantum_state_measure(state);
    printf("Measurement result: %f\n", result);

    // Cleanup
    quantum_state_destroy(state);
    return 0;
}
```

To build and run:
```bash
gcc tutorial1.c -lquantum_geometric -o tutorial1
./tutorial1
```

### 2. Geometric Protection

```c
// tutorial2.c - Adding geometric error protection
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/core/quantum_geometric_operations.h>

int main() {
    // Create Bell state (|00⟩ + |11⟩)/√2
    quantum_state* state = quantum_state_create(2);
    ComplexFloat amplitudes[4] = {
        {1.0f/sqrt(2.0f), 0.0f},  // |00⟩
        {0.0f, 0.0f},             // |01⟩
        {0.0f, 0.0f},             // |10⟩
        {1.0f/sqrt(2.0f), 0.0f}   // |11⟩
    };
    quantum_state_set_amplitudes(state, amplitudes, 4);

    // Configure geometric protection
    protection_config_t config = {
        .type = PROTECTION_GEOMETRIC,
        .strength = 0.8f,
        .monitoring = true
    };

    // Apply protection
    qgt_error_t err = quantum_geometric_protect(state, &config);
    if (err != QGT_SUCCESS) {
        fprintf(stderr, "Protection failed: %s\n", 
                qgt_error_string(err));
        quantum_state_destroy(state);
        return 1;
    }

    // Verify protection
    protection_metrics_t metrics;
    err = quantum_geometric_verify_protection(state, &metrics);
    if (err == QGT_SUCCESS) {
        printf("Protection active: %.2f%% effective\n", 
               metrics.effectiveness * 100.0f);
    }

    quantum_state_destroy(state);
    return 0;
}
```

### 3. Hardware Integration

```c
// tutorial3.c - Running on real quantum hardware
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/hardware/quantum_hardware.h>

int main() {
    // Configure quantum hardware
    quantum_hardware_config_t config = {
        .backend = BACKEND_IBM,
        .device = "ibm_manhattan",
        .optimization = {
            .topology_aware = true,
            .error_mitigation = true
        }
    };

    // Initialize quantum system
    quantum_system_t* system = quantum_init_system(&config);
    if (!system) {
        fprintf(stderr, "Failed to initialize quantum system\n");
        return 1;
    }

    // Create and run quantum circuit
    quantum_circuit_t* circuit = quantum_circuit_create();
    quantum_circuit_h(circuit, 0);  // Hadamard on qubit 0
    quantum_circuit_cx(circuit, 0, 1);  // CNOT between qubits 0,1

    // Execute with geometric optimization
    execution_results_t results;
    qgt_error_t err = quantum_execute_circuit(
        system, circuit, &results
    );

    if (err == QGT_SUCCESS) {
        printf("Circuit executed successfully\n");
        printf("Fidelity: %.3f\n", results.fidelity);
    }

    // Cleanup
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
    return 0;
}
```

## Intermediate Tutorials

### 4. Quantum Machine Learning

```c
// tutorial4.c - Geometric quantum neural network
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/ai/quantum_geometric_ml.h>

int main() {
    // Configure ML model
    quantum_ml_config_t config = {
        .num_qubits = 4,
        .num_layers = 2,
        .learning = {
            .type = LEARNING_GEOMETRIC,
            .optimizer = OPTIMIZER_NATURAL_GRADIENT,
            .learning_rate = 0.01f
        }
    };

    // Create and initialize model
    quantum_ml_model_t* model = quantum_ml_create(&config);
    if (!model) {
        fprintf(stderr, "Failed to create ML model\n");
        return 1;
    }

    // Prepare training data
    training_data_t data = {
        .samples = your_training_data,
        .labels = your_training_labels,
        .num_samples = 1000
    };

    // Train with geometric optimization
    qgt_error_t err = quantum_ml_train(model, &data);
    if (err == QGT_SUCCESS) {
        printf("Training complete\n");
        printf("Accuracy: %.2f%%\n", 
               quantum_ml_evaluate(model, test_data) * 100.0f);
    }

    quantum_ml_destroy(model);
    return 0;
}
```

### 5. Distributed Computing

```c
// tutorial5.c - Distributed quantum computing
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/distributed/quantum_distributed.h>

int main() {
    // Configure distributed system
    distributed_config_t config = {
        .num_nodes = 4,
        .backend = DISTRIBUTED_MPI,
        .optimization = {
            .workload_balance = true,
            .communication_optimize = true
        }
    };

    // Initialize distributed system
    distributed_system_t* system = distributed_init(&config);
    if (!system) {
        fprintf(stderr, "Failed to initialize distributed system\n");
        return 1;
    }

    // Create distributed workload
    quantum_workload_t workload = {
        .circuit = your_quantum_circuit,
        .partitioning = PARTITION_GEOMETRIC,
        .synchronization = SYNC_ADAPTIVE
    };

    // Execute distributed computation
    qgt_error_t err = distributed_execute(system, &workload);
    if (err == QGT_SUCCESS) {
        printf("Distributed execution complete\n");
    }

    distributed_destroy(system);
    return 0;
}
```

## Advanced Tutorials

### 6. Custom Error Protection

```c
// tutorial6.c - Custom geometric error protection
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/physics/topological_protection.h>

// Define custom protection scheme
protection_scheme_t create_custom_protection() {
    protection_scheme_t scheme = {
        .manifold = MANIFOLD_COMPLEX_PROJECTIVE,
        .connection = {
            .type = CONNECTION_GEOMETRIC,
            .curvature = CURVATURE_OPTIMAL
        },
        .validation = {
            .method = VALIDATION_CONTINUOUS,
            .threshold = 1e-6f
        }
    };
    return scheme;
}

int main() {
    // Create quantum state
    quantum_state* state = quantum_state_create(3);
    
    // Apply custom protection
    protection_scheme_t scheme = create_custom_protection();
    qgt_error_t err = quantum_apply_protection(state, &scheme);
    
    if (err == QGT_SUCCESS) {
        printf("Custom protection applied\n");
        
        // Verify protection
        float fidelity = quantum_measure_fidelity(state);
        printf("State fidelity: %.6f\n", fidelity);
    }
    
    quantum_state_destroy(state);
    return 0;
}
```

### 7. Performance Optimization

```c
// tutorial7.c - Advanced performance optimization
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/core/performance_optimization.h>

int main() {
    // Configure performance monitoring
    performance_config_t config = {
        .metrics = {
            .circuit_depth = true,
            .gate_fidelity = true,
            .resource_usage = true
        },
        .optimization = {
            .type = OPTIMIZATION_GEOMETRIC,
            .target = TARGET_DEPTH_FIDELITY
        }
    };

    // Create performance monitor
    performance_monitor_t* monitor = 
        performance_monitor_create(&config);
    
    // Monitor and optimize quantum operations
    quantum_circuit_t* circuit = your_quantum_circuit();
    
    optimization_result_t result;
    qgt_error_t err = optimize_quantum_circuit(
        circuit, monitor, &result
    );
    
    if (err == QGT_SUCCESS) {
        printf("Optimization results:\n");
        printf("Depth reduction: %.1f%%\n", 
               result.depth_reduction * 100.0f);
        printf("Fidelity improvement: %.1f%%\n",
               result.fidelity_improvement * 100.0f);
    }
    
    performance_monitor_destroy(monitor);
    quantum_circuit_destroy(circuit);
    return 0;
}
```

## Best Practices

1. **Error Handling**
   - Always check return values
   - Use appropriate error handling functions
   - Clean up resources properly

```c
// Good error handling
quantum_state* state = quantum_state_create(2);
if (!state) {
    handle_error("Failed to create quantum state");
    return 1;
}

qgt_error_t err = quantum_operation(state);
if (err != QGT_SUCCESS) {
    handle_error(qgt_error_string(err));
    quantum_state_destroy(state);
    return 1;
}
```

2. **Resource Management**
   - Free resources when no longer needed
   - Use monitoring functions
   - Implement proper cleanup

```c
// Good resource management
quantum_system_t* system = quantum_init_system(&config);
quantum_circuit_t* circuit = NULL;

if (system) {
    circuit = quantum_circuit_create();
    if (circuit) {
        // Use system and circuit
        quantum_circuit_destroy(circuit);
    }
    quantum_system_destroy(system);
}
```

3. **Performance Optimization**
   - Use hardware acceleration when available
   - Monitor performance metrics
   - Optimize resource allocation

```c
// Good performance practices
performance_config_t perf_config = {
    .hardware_acceleration = true,
    .memory_optimization = true,
    .monitoring = MONITORING_CONTINUOUS
};

performance_monitor_t* monitor = 
    performance_monitor_create(&perf_config);

// Monitor performance
performance_metrics_t metrics;
collect_performance_metrics(monitor, &metrics);

// Optimize based on metrics
if (metrics.needs_optimization) {
    optimize_quantum_operations(monitor);
}
```

## Next Steps

After completing these tutorials, you can:

1. Explore advanced topics:
   - [Quantum Geometric Documentation](QUANTUM_GEOMETRIC.md)
   - [Hardware Integration Guide](QUANTUM_HARDWARE.md)
   - [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION.md)

2. Try more examples:
   - [Example Directory](../examples/)
   - [Advanced Examples](../examples/advanced/)
   - [Hardware Examples](../examples/hardware/)

3. Contribute to the project:
   - [Contributing Guidelines](../CONTRIBUTING.md)
   - [Development Setup](../docs/DEVELOPMENT.md)
   - [Testing Guide](../docs/TESTING.md)
