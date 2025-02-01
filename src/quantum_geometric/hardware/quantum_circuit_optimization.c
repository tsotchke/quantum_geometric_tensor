#include "quantum_geometric/hardware/quantum_circuit_optimization.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Rigetti-specific parameters
#define RIGETTI_RX_ERROR 0.01
#define RIGETTI_RZ_ERROR 0.005
#define RIGETTI_CZ_ERROR 0.03
#define RIGETTI_READOUT_ERROR 0.02
#define RIGETTI_T1_TIME 20.0  // microseconds
#define RIGETTI_T2_TIME 15.0  // microseconds
#define RIGETTI_GATE_TIME 0.1 // microseconds

// Optimization parameters
#define MAX_OPTIMIZATION_ROUNDS 10
#define MIN_FIDELITY 0.95
#define MAX_GATE_DEPTH 50
#define MAX_SWAP_DEPTH 5
#define CROSSTALK_THRESHOLD 0.01

// Gate commutation rules with noise awareness
static bool gates_commute(const QuantumGate* g1,
                         const QuantumGate* g2,
                         const RigettiConfig* config) {
    // Check for crosstalk
    if (has_crosstalk(g1, g2, config)) {
        return false;
    }
    
    // Same qubit operations
    if (g1->target == g2->target) {
        // RZ gates always commute
        if (g1->type == GATE_RZ && g2->type == GATE_RZ) {
            return true;
        }
        return false;  // Non-commuting
    }
    
    // Two-qubit operations
    if (g1->type == GATE_CZ) {
        if (g2->type == GATE_CZ) {
            // Check for overlapping qubits
            return g1->target != g2->target &&
                   g1->target != g2->control &&
                   g1->control != g2->target &&
                   g1->control != g2->control;
        }
        // Single-qubit gates commute if not on involved qubits
        return g2->target != g1->target &&
               g2->target != g1->control;
    }
    
    if (g2->type == GATE_CZ) {
        return g1->target != g2->target &&
               g1->target != g2->control;
    }
    
    // Single-qubit operations on different qubits
    return true;
}

// Check for crosstalk between gates
static bool has_crosstalk(const QuantumGate* g1,
                         const QuantumGate* g2,
                         const RigettiConfig* config) {
    // Get physical qubits
    size_t q1 = g1->target;
    size_t q2 = g2->target;
    
    // Check crosstalk map
    return config->crosstalk_map[q1 * config->num_qubits + q2] >
           CROSSTALK_THRESHOLD;
}

// Gate fusion rules for Rigetti native gates
static bool can_fuse_gates(const QuantumGate* g1,
                          const QuantumGate* g2,
                          QuantumGate* result,
                          const RigettiConfig* config) {
    // Must be same type and target
    if (g1->type != g2->type || g1->target != g2->target) {
        return false;
    }
    
    // Handle native Rigetti gates
    switch (g1->type) {
        case GATE_RX:
            result->type = GATE_RX;
            result->target = g1->target;
            result->parameter = fmod(g1->parameter + g2->parameter,
                                  2.0 * M_PI);
            return true;
            
        case GATE_RZ:
            result->type = GATE_RZ;
            result->target = g1->target;
            result->parameter = fmod(g1->parameter + g2->parameter,
                                  2.0 * M_PI);
            return true;
            
        default:
            return false;
    }
}

// Optimize sequence with noise awareness
static void optimize_sequence(QuantumCircuit* circuit,
                            size_t start,
                            size_t end,
                            const RigettiConfig* config) {
    if (end - start < 2) return;
    
    // Try to commute gates to enable fusion
    for (size_t i = start; i < end - 1; i++) {
        for (size_t j = i + 1; j < end; j++) {
            QuantumGate* g1 = &circuit->gates[i];
            QuantumGate* g2 = &circuit->gates[j];
            
            // Check if gates can be brought together
            bool can_commute = true;
            for (size_t k = i + 1; k < j; k++) {
                if (!gates_commute(g1, &circuit->gates[k], config)) {
                    can_commute = false;
                    break;
                }
            }
            
            if (can_commute) {
                // Try fusion
                QuantumGate fused;
                if (can_fuse_gates(g1, g2, &fused, config)) {
                    // Check if fusion improves fidelity
                    double original_fidelity = compute_sequence_fidelity(
                        &circuit->gates[i], 2, config);
                    double fused_fidelity = compute_gate_fidelity(
                        &fused, config);
                    
                    if (fused_fidelity > original_fidelity) {
                        // Replace with fused gate
                        circuit->gates[i] = fused;
                        memmove(&circuit->gates[j],
                               &circuit->gates[j + 1],
                               (circuit->num_gates - j - 1) *
                               sizeof(QuantumGate));
                        circuit->num_gates--;
                        end--;
                        break;
                    }
                }
            }
        }
    }
}

// Compute fidelity of gate sequence
static double compute_sequence_fidelity(const QuantumGate* gates,
                                      size_t num_gates,
                                      const RigettiConfig* config) {
    double fidelity = 1.0;
    double total_time = 0.0;
    
    for (size_t i = 0; i < num_gates; i++) {
        // Get base error rate
        double error_rate;
        switch (gates[i].type) {
            case GATE_RX:
                error_rate = RIGETTI_RX_ERROR;
                total_time += RIGETTI_GATE_TIME;
                break;
            case GATE_RZ:
                error_rate = RIGETTI_RZ_ERROR;
                total_time += RIGETTI_GATE_TIME;
                break;
            case GATE_CZ:
                error_rate = RIGETTI_CZ_ERROR;
                total_time += 2 * RIGETTI_GATE_TIME;
                break;
            default:
                error_rate = 0.02;  // Conservative estimate
                total_time += RIGETTI_GATE_TIME;
                break;
        }
        
        // Account for decoherence
        double t1_error = 1.0 - exp(-total_time / RIGETTI_T1_TIME);
        double t2_error = 1.0 - exp(-total_time / RIGETTI_T2_TIME);
        
        fidelity *= (1.0 - error_rate) * (1.0 - t1_error) *
                   (1.0 - t2_error);
    }
    
    return fidelity;
}

// Compute fidelity of single gate
static double compute_gate_fidelity(const QuantumGate* gate,
                                  const RigettiConfig* config) {
    switch (gate->type) {
        case GATE_RX:
            return 1.0 - RIGETTI_RX_ERROR;
        case GATE_RZ:
            return 1.0 - RIGETTI_RZ_ERROR;
        case GATE_CZ:
            return 1.0 - RIGETTI_CZ_ERROR;
        default:
            return 0.98;  // Conservative estimate
    }
}

// Compute circuit depth with hardware constraints
static size_t compute_depth(const QuantumCircuit* circuit,
                          const RigettiConfig* config) {
    if (!circuit || circuit->num_gates == 0) return 0;
    
    // Track last gate layer for each qubit
    size_t* qubit_layers = calloc(config->num_qubits,
                                 sizeof(size_t));
    if (!qubit_layers) return 0;
    
    size_t max_depth = 0;
    
    for (size_t i = 0; i < circuit->num_gates; i++) {
        const QuantumGate* gate = &circuit->gates[i];
        size_t layer = qubit_layers[gate->target];
        
        // For CZ gates, consider both qubits
        if (gate->type == GATE_CZ) {
            layer = max(layer, qubit_layers[gate->control]);
            
            // Check if qubits are connected
            if (!config->connectivity[gate->control *
                                   config->num_qubits +
                                   gate->target]) {
                // Add SWAP overhead
                layer += 2 * MAX_SWAP_DEPTH;
            }
        }
        
        // Place gate in next layer
        layer++;
        qubit_layers[gate->target] = layer;
        if (gate->type == GATE_CZ) {
            qubit_layers[gate->control] = layer;
        }
        
        max_depth = max(max_depth, layer);
    }
    
    free(qubit_layers);
    return max_depth;
}

// Optimize circuit for Rigetti hardware
void optimize_quantum_circuit(QuantumCircuit* circuit,
                           const RigettiConfig* config) {
    if (!circuit || !config) return;
    
    // Initial metrics
    size_t initial_depth = compute_depth(circuit, config);
    double initial_fidelity = compute_sequence_fidelity(
        circuit->gates,
        circuit->num_gates,
        config
    );
    
    printf("Initial circuit: depth=%zu, fidelity=%.6f\n",
           initial_depth, initial_fidelity);
    
    // Optimize in rounds
    for (int round = 0; round < MAX_OPTIMIZATION_ROUNDS; round++) {
        size_t old_num_gates = circuit->num_gates;
        
        // Optimize subsequences
        for (size_t i = 0; i < circuit->num_gates;
             i += MAX_GATE_DEPTH) {
            size_t end = min(i + MAX_GATE_DEPTH,
                           circuit->num_gates);
            optimize_sequence(circuit, i, end, config);
        }
        
        // Check if optimization helped
        if (circuit->num_gates == old_num_gates) {
            break;  // No more optimizations possible
        }
    }
    
    // Map to hardware topology
    optimize_qubit_mapping(circuit, config);
    
    // Final metrics
    size_t final_depth = compute_depth(circuit, config);
    double final_fidelity = compute_sequence_fidelity(
        circuit->gates,
        circuit->num_gates,
        config
    );
    
    printf("Optimized circuit: depth=%zu, fidelity=%.6f\n",
           final_depth, final_fidelity);
    printf("Gate reduction: %zu -> %zu\n",
           initial_depth, final_depth);
}

// Map circuit to hardware topology
static void optimize_qubit_mapping(QuantumCircuit* circuit,
                                 const RigettiConfig* config) {
    // Initialize qubit mapping
    size_t* mapping = malloc(config->num_qubits * sizeof(size_t));
    size_t* inverse = malloc(config->num_qubits * sizeof(size_t));
    if (!mapping || !inverse) {
        free(mapping);
        free(inverse);
        return;
    }
    
    // Start with identity mapping
    for (size_t i = 0; i < config->num_qubits; i++) {
        mapping[i] = i;
        inverse[i] = i;
    }
    
    // Process each CZ gate
    for (size_t i = 0; i < circuit->num_gates; i++) {
        QuantumGate* gate = &circuit->gates[i];
        if (gate->type != GATE_CZ) continue;
        
        // Get physical qubits
        size_t control = mapping[gate->control];
        size_t target = mapping[gate->target];
        
        // Check if directly connected
        if (config->connectivity[control * config->num_qubits +
                               target]) {
            continue;
        }
        
        // Find shortest path
        size_t path[MAX_QUBITS];
        size_t path_length = find_shortest_path(
            config,
            control,
            target,
            path
        );
        
        if (path_length == 0) continue;
        
        // Insert SWAP gates
        for (size_t j = 0; j < path_length - 1; j++) {
            // Create SWAP using CZ and RX gates
            insert_swap(circuit, i, path[j], path[j + 1]);
            
            // Update mappings
            size_t temp = mapping[inverse[path[j]]];
            mapping[inverse[path[j]]] = mapping[inverse[path[j + 1]]];
            mapping[inverse[path[j + 1]]] = temp;
            
            temp = inverse[path[j]];
            inverse[path[j]] = inverse[path[j + 1]];
            inverse[path[j + 1]] = temp;
        }
    }
    
    free(mapping);
    free(inverse);
}

// Find shortest path between qubits
static size_t find_shortest_path(const RigettiConfig* config,
                               size_t start,
                               size_t end,
                               size_t* path) {
    // Use Dijkstra's algorithm
    bool* visited = calloc(config->num_qubits, sizeof(bool));
    double* distances = malloc(config->num_qubits * sizeof(double));
    size_t* previous = malloc(config->num_qubits * sizeof(size_t));
    
    if (!visited || !distances || !previous) {
        free(visited);
        free(distances);
        free(previous);
        return 0;
    }
    
    // Initialize distances
    for (size_t i = 0; i < config->num_qubits; i++) {
        distances[i] = INFINITY;
        previous[i] = -1;
    }
    distances[start] = 0;
    
    // Main loop
    for (size_t count = 0; count < config->num_qubits - 1; count++) {
        // Find minimum distance vertex
        double min_dist = INFINITY;
        size_t min_vertex = -1;
        
        for (size_t v = 0; v < config->num_qubits; v++) {
            if (!visited[v] && distances[v] < min_dist) {
                min_dist = distances[v];
                min_vertex = v;
            }
        }
        
        if (min_vertex == -1) break;
        visited[min_vertex] = true;
        
        // Update distances
        for (size_t v = 0; v < config->num_qubits; v++) {
            if (!visited[v] &&
                config->connectivity[min_vertex * config->num_qubits + v]) {
                double dist = distances[min_vertex] + 1.0;
                if (dist < distances[v]) {
                    distances[v] = dist;
                    previous[v] = min_vertex;
                }
            }
        }
    }
    
    // Reconstruct path
    size_t length = 0;
    size_t current = end;
    
    while (current != (size_t)-1 && current != start) {
        path[length++] = current;
        current = previous[current];
    }
    
    if (current == start) {
        path[length++] = start;
        // Reverse path
        for (size_t i = 0; i < length / 2; i++) {
            size_t temp = path[i];
            path[i] = path[length - 1 - i];
            path[length - 1 - i] = temp;
        }
    } else {
        length = 0;  // No path found
    }
    
    free(visited);
    free(distances);
    free(previous);
    
    return length;
}

// Insert SWAP operation using native gates
static void insert_swap(QuantumCircuit* circuit,
                       size_t index,
                       size_t q1,
                       size_t q2) {
    // SWAP = CZ - RX(π/2) - CZ - RX(-π/2) - CZ - RX(π/2)
    QuantumGate gates[] = {
        {.type = GATE_CZ, .control = q1, .target = q2},
        {.type = GATE_RX, .target = q2, .parameter = M_PI_2},
        {.type = GATE_CZ, .control = q1, .target = q2},
        {.type = GATE_RX, .target = q2, .parameter = -M_PI_2},
        {.type = GATE_CZ, .control = q1, .target = q2},
        {.type = GATE_RX, .target = q2, .parameter = M_PI_2}
    };
    
    // Insert gates
    for (int i = 5; i >= 0; i--) {
        memmove(&circuit->gates[index + 1],
                &circuit->gates[index],
                (circuit->num_gates - index) *
                sizeof(QuantumGate));
        circuit->gates[index] = gates[i];
        circuit->num_gates++;
    }
}
