# Quantum Geometric Learning Examples

This document provides practical examples of using the Quantum Geometric Learning library for various applications. Each example demonstrates key features and best practices.

## Basic Examples

### 1. Quantum State Evolution

```c
// examples/beginner/quantum_evolution_example.c
#include <physicsml/tensor_network_operations.h>

int main() {
    // Create a 3-qubit system
    TreeTensorNetwork* ttn = physicsml_ttn_create(
        3,                          // 3 levels (8 qubits)
        16,                         // Bond dimension
        2,                          // Qubit dimension
        PHYSICSML_DTYPE_COMPLEX128  // Complex numbers
    );
    if (!ttn) {
        fprintf(stderr, "Failed to create tensor network\n");
        return 1;
    }

    // Define Hamiltonian (Transverse Ising model)
    PhysicsMLTensor* hamiltonian = create_transverse_ising_hamiltonian(3, 1.0);
    if (!hamiltonian) {
        physicsml_ttn_destroy(ttn);
        return 1;
    }

    // Time evolution parameters
    double dt = 0.01;
    size_t num_steps = 100;

    // Evolve state
    for (size_t i = 0; i < num_steps; i++) {
        // Single time step evolution
        PhysicsMLError err = physicsml_optimize_physical_network(
            ttn,
            NULL,      // No target state (real-time evolution)
            hamiltonian,
            NULL, 0,   // No additional observables
            dt,        // Time step as learning rate
            1          // Single iteration per step
        );

        if (err != PHYSICSML_SUCCESS) {
            fprintf(stderr, "Evolution failed at step %zu\n", i);
            break;
        }

        // Compute and print energy
        double energy = physicsml_compute_expectation_value(ttn, hamiltonian);
        printf("Step %zu: Energy = %.6f\n", i, energy);
    }

    // Cleanup
    physicsml_tensor_destroy(hamiltonian);
    physicsml_ttn_destroy(ttn);
    return 0;
}
```

### 2. Spin System Simulation

```c
// examples/spin_system_example.c
#include <physicsml/tensor_network_operations.h>

int main() {
    // Create 2D lattice system
    const size_t lattice_size = 4;  // 4x4 lattice
    TreeTensorNetwork* ttn = create_2d_lattice_ttn(
        lattice_size,
        16,                         // Bond dimension
        2,                          // Spin-1/2
        PHYSICSML_DTYPE_COMPLEX128
    );

    // Define Heisenberg Hamiltonian
    PhysicsMLTensor* hamiltonian = create_heisenberg_hamiltonian(
        lattice_size,
        1.0,  // J coupling
        0.0   // External field
    );

    // Find ground state
    PhysicsMLError err = physicsml_optimize_physical_network(
        ttn,
        NULL,      // No target (finding ground state)
        hamiltonian,
        NULL, 0,
        0.01,      // Learning rate
        1000       // Max iterations
    );

    if (err == PHYSICSML_SUCCESS) {
        // Compute observables
        double energy = physicsml_compute_expectation_value(ttn, hamiltonian);
        double magnetization = compute_magnetization(ttn);
        double correlation = compute_spin_correlation(ttn, 0, lattice_size/2);

        printf("Ground state properties:\n");
        printf("Energy: %.6f\n", energy);
        printf("Magnetization: %.6f\n", magnetization);
        printf("Correlation: %.6f\n", correlation);
    }

    // Cleanup
    physicsml_tensor_destroy(hamiltonian);
    physicsml_ttn_destroy(ttn);
    return (err == PHYSICSML_SUCCESS) ? 0 : 1;
}
```

## Intermediate Examples

### 3. Topological Phase Transition

```c
// examples/topological_phase_example.c
#include <physicsml/tensor_network_operations.h>

int main() {
    // Create Kitaev chain system
    TreeTensorNetwork* ttn = create_kitaev_chain(
        10,     // Number of sites
        32,     // Bond dimension
        2       // Local dimension
    );

    // Parameters for phase transition
    const size_t num_points = 50;
    const double mu_min = -2.0;
    const double mu_max = 2.0;
    
    // Scan through chemical potential
    for (size_t i = 0; i < num_points; i++) {
        double mu = mu_min + (mu_max - mu_min) * i / (num_points - 1);
        
        // Update Hamiltonian
        PhysicsMLTensor* hamiltonian = create_kitaev_hamiltonian(
            10,    // Sites
            1.0,   // Hopping
            0.5,   // Pairing
            mu     // Chemical potential
        );

        // Find ground state with topological constraints
        TopologicalConstraints constraints = {
            .winding_number_tolerance = 1e-6,
            .edge_mode_tolerance = 1e-6
        };

        PhysicsMLError err = physicsml_optimize_topological_network(
            ttn,
            NULL,
            hamiltonian,
            0.01,   // Learning rate
            1000    // Max iterations
        );

        if (err == PHYSICSML_SUCCESS) {
            // Compute topological invariants
            double winding = compute_winding_number(ttn);
            bool edge_modes = detect_edge_modes(ttn);
            
            printf("mu = %.3f: W = %.3f, Edge modes: %s\n",
                   mu, winding, edge_modes ? "Yes" : "No");
        }

        physicsml_tensor_destroy(hamiltonian);
    }

    physicsml_ttn_destroy(ttn);
    return 0;
}
```

### 4. Holographic Optimization

```c
// examples/holographic_example.c
#include <physicsml/tensor_network_operations.h>

int main() {
    // Create bulk tensor network
    TreeTensorNetwork* ttn = create_holographic_network(
        4,      // Bulk depth
        32,     // Bond dimension
        2       // Boundary local dimension
    );

    // Define boundary state
    PhysicsMLTensor* boundary_state = create_thermal_boundary_state(
        16,     // System size
        1.0     // Temperature
    );

    // Define bulk Hamiltonian
    PhysicsMLTensor* bulk_hamiltonian = create_bulk_hamiltonian();

    // Setup holographic constraints
    HolographicConstraints constraints = {
        .boundary_error_threshold = 1e-6,
        .bulk_energy_threshold = -0.99,
        .entanglement_entropy_threshold = 0.1,
        .area_law_tolerance = 1e-4
    };

    // Optimize bulk geometry
    PhysicsMLError err = physicsml_optimize_holographic_network(
        ttn,
        boundary_state,
        bulk_hamiltonian,
        0.01,   // Learning rate
        1000    // Max iterations
    );

    if (err == PHYSICSML_SUCCESS) {
        // Compute geometric quantities
        double curvature = physicsml_compute_bulk_curvature(ttn);
        
        // Get minimal surfaces
        MinimalSurface* surfaces = physicsml_compute_minimal_surfaces(ttn);
        
        // Verify RT formula
        double entropy = physicsml_compute_entanglement_entropy(ttn);
        double area = compute_minimal_surface_area(surfaces);
        
        printf("Bulk geometry properties:\n");
        printf("Curvature: %.6f\n", curvature);
        printf("Entropy: %.6f\n", entropy);
        printf("Minimal surface area: %.6f\n", area);
        
        physicsml_free_minimal_surfaces(surfaces);
    }

    // Cleanup
    physicsml_tensor_destroy(boundary_state);
    physicsml_tensor_destroy(bulk_hamiltonian);
    physicsml_ttn_destroy(ttn);
    return 0;
}
```

## Advanced Examples

### 5. Quantum Machine Learning

```c
// examples/quantum_ml_example.c
#include <physicsml/tensor_network_operations.h>

int main() {
    // Load training data
    const size_t num_samples = 1000;
    PhysicsMLTensor** training_states = load_quantum_dataset(
        "quantum_data.h5",
        num_samples
    );

    // Create quantum neural network
    TreeTensorNetwork* qnn = create_quantum_neural_network(
        4,      // Number of layers
        16,     // Hidden dimension
        2       // Input/output dimension
    );

    // Training loop
    const size_t epochs = 100;
    const size_t batch_size = 32;
    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        // Mini-batch training
        for (size_t i = 0; i < num_samples; i += batch_size) {
            size_t current_batch = min(batch_size, num_samples - i);
            
            // Compute batch gradient
            PhysicsMLTensor* batch_gradient = compute_batch_gradient(
                qnn,
                training_states + i,
                current_batch
            );

            // Update network parameters
            PhysicsMLError err = update_quantum_network(
                qnn,
                batch_gradient,
                0.01  // Learning rate
            );

            physicsml_tensor_destroy(batch_gradient);
            
            if (err != PHYSICSML_SUCCESS) {
                fprintf(stderr, "Training failed at epoch %zu, batch %zu\n",
                        epoch, i/batch_size);
                break;
            }

            // Compute batch loss
            double batch_loss = compute_batch_loss(
                qnn,
                training_states + i,
                current_batch
            );
            total_loss += batch_loss;
        }

        printf("Epoch %zu: Average loss = %.6f\n",
               epoch, total_loss/num_samples);
    }

    // Cleanup
    for (size_t i = 0; i < num_samples; i++) {
        physicsml_tensor_destroy(training_states[i]);
    }
    free(training_states);
    physicsml_ttn_destroy(qnn);
    return 0;
}
```

## Visualization Examples

See the `examples/visualization/` directory for interactive visualization examples:
- `quantum_visualization.html`: Quantum state visualization
- `topological_braiding.html`: Topological braiding animation
- `geometric_learning.html`: Geometric learning visualization

## Running the Examples

1. Build the examples:
```bash
cd examples
mkdir build && cd build
cmake ..
cmake --build .
```

2. Run specific example:
```bash
./quantum_evolution_example
./spin_system_example
./topological_phase_example
```

3. View visualizations:
```bash
cd ../visualization
python3 start_visualization.py
```

## Additional Resources

- Complete example source code in `examples/` directory
- Example datasets in `examples/data/`
- Visualization tools in `examples/visualization/`
- Benchmark examples in `benchmarks/`
- Test examples in `tests/`

## Best Practices

1. **Error Handling**
   - Always check return values
   - Clean up resources on error
   - Use appropriate error codes

2. **Memory Management**
   - Free all allocated resources
   - Use RAII-style wrappers when possible
   - Check for memory leaks

3. **Performance**
   - Use appropriate bond dimensions
   - Enable hardware acceleration
   - Profile and optimize bottlenecks

4. **Visualization**
   - Monitor convergence
   - Visualize intermediate results
   - Debug using visual tools

## Further Reading

- [API Documentation](API.md)
- [Theory Background](THEORY.md)
- [Performance Guide](PERFORMANCE.md)
