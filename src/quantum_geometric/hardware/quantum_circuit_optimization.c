/**
 * @file quantum_circuit_optimization.c
 * @brief Rigetti-specific quantum circuit optimization with noise awareness
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// Constants
// ============================================================================

// Rigetti-specific parameters
#define RIGETTI_RX_ERROR 0.01
#define RIGETTI_RZ_ERROR 0.005
#define RIGETTI_CZ_ERROR 0.03
#define RIGETTI_READOUT_ERROR 0.02
#define RIGETTI_T1_TIME 20.0   // microseconds
#define RIGETTI_T2_TIME 15.0   // microseconds
#define RIGETTI_GATE_TIME 0.1  // microseconds

// Optimization parameters
#define MAX_OPTIMIZATION_ROUNDS 10
#define MIN_FIDELITY 0.95
#define MAX_GATE_DEPTH 50
#define MAX_SWAP_DEPTH 5
#define CROSSTALK_THRESHOLD 0.01
#define MAX_QUBITS 128

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// ============================================================================
// Type Definitions (self-contained for this module)
// ============================================================================

// Gate type enumeration
typedef enum {
    OPT_GATE_I = 0,
    OPT_GATE_X,
    OPT_GATE_Y,
    OPT_GATE_Z,
    OPT_GATE_H,
    OPT_GATE_S,
    OPT_GATE_T,
    OPT_GATE_RX,
    OPT_GATE_RY,
    OPT_GATE_RZ,
    OPT_GATE_CX,
    OPT_GATE_CY,
    OPT_GATE_CZ,
    OPT_GATE_SWAP,
    OPT_GATE_ISWAP
} OptGateType;

// Gate structure for optimization
typedef struct {
    OptGateType type;
    size_t target;
    size_t control;
    double parameter;
} OptGate;

// Circuit structure
typedef struct {
    OptGate* gates;
    size_t num_gates;
    size_t capacity;
    size_t num_qubits;
} OptCircuit;

// Rigetti hardware configuration
typedef struct {
    size_t num_qubits;
    double* crosstalk_map;       // num_qubits x num_qubits matrix
    size_t* connectivity;        // Adjacency matrix for qubit connectivity
    double* gate_errors;
    double* readout_errors;
    double t1_time;
    double t2_time;
    double gate_time;
} OptRigettiConfig;

// ============================================================================
// Helper macros
// ============================================================================

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// Forward Declarations
// ============================================================================

static bool has_crosstalk(const OptGate* g1, const OptGate* g2, const OptRigettiConfig* config);
static double compute_sequence_fidelity(const OptGate* gates, size_t num_gates, const OptRigettiConfig* config);
static double compute_gate_fidelity(const OptGate* gate, const OptRigettiConfig* config);
static size_t find_shortest_path(const OptRigettiConfig* config, size_t start, size_t end, size_t* path);
static void insert_swap(OptCircuit* circuit, size_t index, size_t q1, size_t q2);
static void optimize_qubit_mapping(OptCircuit* circuit, const OptRigettiConfig* config);

// ============================================================================
// Gate Commutation and Fusion
// ============================================================================

// Check if gates commute (considering crosstalk)
static bool gates_commute(const OptGate* g1, const OptGate* g2, const OptRigettiConfig* config) {
    // Check for crosstalk
    if (has_crosstalk(g1, g2, config)) {
        return false;
    }

    // Same qubit operations
    if (g1->target == g2->target) {
        // RZ gates always commute
        if (g1->type == OPT_GATE_RZ && g2->type == OPT_GATE_RZ) {
            return true;
        }
        return false;  // Non-commuting
    }

    // Two-qubit operations
    if (g1->type == OPT_GATE_CZ) {
        if (g2->type == OPT_GATE_CZ) {
            return g1->target != g2->target &&
                   g1->target != g2->control &&
                   g1->control != g2->target &&
                   g1->control != g2->control;
        }
        return g2->target != g1->target && g2->target != g1->control;
    }

    if (g2->type == OPT_GATE_CZ) {
        return g1->target != g2->target && g1->target != g2->control;
    }

    // Single-qubit operations on different qubits commute
    return true;
}

// Check for crosstalk between gates
static bool has_crosstalk(const OptGate* g1, const OptGate* g2, const OptRigettiConfig* config) {
    if (!config || !config->crosstalk_map) return false;

    size_t q1 = g1->target;
    size_t q2 = g2->target;

    if (q1 >= config->num_qubits || q2 >= config->num_qubits) return false;

    return config->crosstalk_map[q1 * config->num_qubits + q2] > CROSSTALK_THRESHOLD;
}

// Try to fuse two gates
static bool can_fuse_gates(const OptGate* g1, const OptGate* g2, OptGate* result,
                           const OptRigettiConfig* config) {
    (void)config;

    if (g1->type != g2->type || g1->target != g2->target) {
        return false;
    }

    switch (g1->type) {
        case OPT_GATE_RX:
            result->type = OPT_GATE_RX;
            result->target = g1->target;
            result->control = 0;
            result->parameter = fmod(g1->parameter + g2->parameter, 2.0 * M_PI);
            return true;

        case OPT_GATE_RZ:
            result->type = OPT_GATE_RZ;
            result->target = g1->target;
            result->control = 0;
            result->parameter = fmod(g1->parameter + g2->parameter, 2.0 * M_PI);
            return true;

        default:
            return false;
    }
}

// ============================================================================
// Sequence Optimization
// ============================================================================

static void optimize_sequence(OptCircuit* circuit, size_t start, size_t end,
                              const OptRigettiConfig* config) {
    if (end - start < 2) return;

    for (size_t i = start; i < end - 1; i++) {
        for (size_t j = i + 1; j < end; j++) {
            OptGate* g1 = &circuit->gates[i];
            OptGate* g2 = &circuit->gates[j];

            // Check if gates can be brought together
            bool can_commute = true;
            for (size_t k = i + 1; k < j; k++) {
                if (!gates_commute(g1, &circuit->gates[k], config)) {
                    can_commute = false;
                    break;
                }
            }

            if (can_commute) {
                OptGate fused;
                if (can_fuse_gates(g1, g2, &fused, config)) {
                    double original_fidelity = compute_sequence_fidelity(&circuit->gates[i], 2, config);
                    double fused_fidelity = compute_gate_fidelity(&fused, config);

                    if (fused_fidelity > original_fidelity) {
                        circuit->gates[i] = fused;
                        memmove(&circuit->gates[j], &circuit->gates[j + 1],
                               (circuit->num_gates - j - 1) * sizeof(OptGate));
                        circuit->num_gates--;
                        end--;
                        break;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Fidelity Calculations
// ============================================================================

static double compute_sequence_fidelity(const OptGate* gates, size_t num_gates,
                                        const OptRigettiConfig* config) {
    (void)config;

    double fidelity = 1.0;
    double total_time = 0.0;

    for (size_t i = 0; i < num_gates; i++) {
        double error_rate;
        switch (gates[i].type) {
            case OPT_GATE_RX:
                error_rate = RIGETTI_RX_ERROR;
                total_time += RIGETTI_GATE_TIME;
                break;
            case OPT_GATE_RZ:
                error_rate = RIGETTI_RZ_ERROR;
                total_time += RIGETTI_GATE_TIME;
                break;
            case OPT_GATE_CZ:
                error_rate = RIGETTI_CZ_ERROR;
                total_time += 2 * RIGETTI_GATE_TIME;
                break;
            default:
                error_rate = 0.02;
                total_time += RIGETTI_GATE_TIME;
                break;
        }

        double t1_error = 1.0 - exp(-total_time / RIGETTI_T1_TIME);
        double t2_error = 1.0 - exp(-total_time / RIGETTI_T2_TIME);

        fidelity *= (1.0 - error_rate) * (1.0 - t1_error) * (1.0 - t2_error);
    }

    return fidelity;
}

static double compute_gate_fidelity(const OptGate* gate, const OptRigettiConfig* config) {
    (void)config;

    switch (gate->type) {
        case OPT_GATE_RX: return 1.0 - RIGETTI_RX_ERROR;
        case OPT_GATE_RZ: return 1.0 - RIGETTI_RZ_ERROR;
        case OPT_GATE_CZ: return 1.0 - RIGETTI_CZ_ERROR;
        default: return 0.98;
    }
}

// ============================================================================
// Circuit Depth Calculation
// ============================================================================

static size_t compute_depth(const OptCircuit* circuit, const OptRigettiConfig* config) {
    if (!circuit || circuit->num_gates == 0) return 0;

    size_t* qubit_layers = calloc(config->num_qubits, sizeof(size_t));
    if (!qubit_layers) return 0;

    size_t max_depth = 0;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        const OptGate* gate = &circuit->gates[i];
        size_t layer = qubit_layers[gate->target];

        if (gate->type == OPT_GATE_CZ) {
            layer = max(layer, qubit_layers[gate->control]);

            if (config->connectivity &&
                !config->connectivity[gate->control * config->num_qubits + gate->target]) {
                layer += 2 * MAX_SWAP_DEPTH;
            }
        }

        layer++;
        qubit_layers[gate->target] = layer;
        if (gate->type == OPT_GATE_CZ) {
            qubit_layers[gate->control] = layer;
        }

        max_depth = max(max_depth, layer);
    }

    free(qubit_layers);
    return max_depth;
}

// ============================================================================
// Qubit Mapping Optimization
// ============================================================================

static size_t find_shortest_path(const OptRigettiConfig* config,
                                 size_t start, size_t end, size_t* path) {
    if (!config->connectivity) {
        path[0] = start;
        path[1] = end;
        return 2;
    }

    bool* visited = calloc(config->num_qubits, sizeof(bool));
    double* distances = malloc(config->num_qubits * sizeof(double));
    size_t* previous = malloc(config->num_qubits * sizeof(size_t));

    if (!visited || !distances || !previous) {
        free(visited);
        free(distances);
        free(previous);
        return 0;
    }

    for (size_t i = 0; i < config->num_qubits; i++) {
        distances[i] = INFINITY;
        previous[i] = (size_t)-1;
    }
    distances[start] = 0;

    for (size_t count = 0; count < config->num_qubits - 1; count++) {
        double min_dist = INFINITY;
        size_t min_vertex = (size_t)-1;

        for (size_t v = 0; v < config->num_qubits; v++) {
            if (!visited[v] && distances[v] < min_dist) {
                min_dist = distances[v];
                min_vertex = v;
            }
        }

        if (min_vertex == (size_t)-1) break;
        visited[min_vertex] = true;

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

    size_t length = 0;
    size_t current = end;

    while (current != (size_t)-1 && current != start) {
        path[length++] = current;
        current = previous[current];
    }

    if (current == start) {
        path[length++] = start;
        for (size_t i = 0; i < length / 2; i++) {
            size_t temp = path[i];
            path[i] = path[length - 1 - i];
            path[length - 1 - i] = temp;
        }
    } else {
        length = 0;
    }

    free(visited);
    free(distances);
    free(previous);

    return length;
}

static void insert_swap(OptCircuit* circuit, size_t index, size_t q1, size_t q2) {
    // SWAP = CZ - RX(π/2) - CZ - RX(-π/2) - CZ - RX(π/2)
    OptGate gates[6] = {
        {.type = OPT_GATE_CZ, .control = q1, .target = q2, .parameter = 0},
        {.type = OPT_GATE_RX, .target = q2, .control = 0, .parameter = M_PI_2},
        {.type = OPT_GATE_CZ, .control = q1, .target = q2, .parameter = 0},
        {.type = OPT_GATE_RX, .target = q2, .control = 0, .parameter = -M_PI_2},
        {.type = OPT_GATE_CZ, .control = q1, .target = q2, .parameter = 0},
        {.type = OPT_GATE_RX, .target = q2, .control = 0, .parameter = M_PI_2}
    };

    // Ensure capacity
    if (circuit->num_gates + 6 > circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        OptGate* new_gates = realloc(circuit->gates, new_capacity * sizeof(OptGate));
        if (!new_gates) return;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Insert gates
    for (int i = 5; i >= 0; i--) {
        memmove(&circuit->gates[index + 1], &circuit->gates[index],
                (circuit->num_gates - index) * sizeof(OptGate));
        circuit->gates[index] = gates[i];
        circuit->num_gates++;
    }
}

static void optimize_qubit_mapping(OptCircuit* circuit, const OptRigettiConfig* config) {
    if (!config->connectivity) return;

    size_t* mapping = malloc(config->num_qubits * sizeof(size_t));
    size_t* inverse = malloc(config->num_qubits * sizeof(size_t));

    if (!mapping || !inverse) {
        free(mapping);
        free(inverse);
        return;
    }

    for (size_t i = 0; i < config->num_qubits; i++) {
        mapping[i] = i;
        inverse[i] = i;
    }

    for (size_t i = 0; i < circuit->num_gates; i++) {
        OptGate* gate = &circuit->gates[i];
        if (gate->type != OPT_GATE_CZ) continue;

        size_t control = mapping[gate->control];
        size_t target = mapping[gate->target];

        if (config->connectivity[control * config->num_qubits + target]) {
            continue;
        }

        size_t path[MAX_QUBITS];
        size_t path_length = find_shortest_path(config, control, target, path);

        if (path_length == 0) continue;

        for (size_t j = 0; j + 1 < path_length; j++) {
            insert_swap(circuit, i, path[j], path[j + 1]);

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

// ============================================================================
// Public Interface
// ============================================================================

// Create optimized circuit
OptCircuit* create_opt_circuit(size_t num_qubits, size_t initial_capacity) {
    OptCircuit* circuit = calloc(1, sizeof(OptCircuit));
    if (!circuit) return NULL;

    circuit->gates = calloc(initial_capacity, sizeof(OptGate));
    if (!circuit->gates) {
        free(circuit);
        return NULL;
    }

    circuit->capacity = initial_capacity;
    circuit->num_gates = 0;
    circuit->num_qubits = num_qubits;

    return circuit;
}

// Destroy optimized circuit
void destroy_opt_circuit(OptCircuit* circuit) {
    if (!circuit) return;
    free(circuit->gates);
    free(circuit);
}

// Add gate to circuit
bool add_opt_gate(OptCircuit* circuit, const OptGate* gate) {
    if (!circuit || !gate) return false;

    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        OptGate* new_gates = realloc(circuit->gates, new_capacity * sizeof(OptGate));
        if (!new_gates) return false;
        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    circuit->gates[circuit->num_gates++] = *gate;
    return true;
}

// Optimize circuit for Rigetti hardware
void optimize_quantum_circuit(OptCircuit* circuit, const OptRigettiConfig* config) {
    if (!circuit || !config) return;

    size_t initial_depth = compute_depth(circuit, config);
    double initial_fidelity = compute_sequence_fidelity(circuit->gates, circuit->num_gates, config);

    printf("Initial circuit: depth=%zu, fidelity=%.6f\n", initial_depth, initial_fidelity);

    // Optimize in rounds
    for (int round = 0; round < MAX_OPTIMIZATION_ROUNDS; round++) {
        size_t old_num_gates = circuit->num_gates;

        for (size_t i = 0; i < circuit->num_gates; i += MAX_GATE_DEPTH) {
            size_t end = min(i + MAX_GATE_DEPTH, circuit->num_gates);
            optimize_sequence(circuit, i, end, config);
        }

        if (circuit->num_gates == old_num_gates) {
            break;
        }
    }

    // Map to hardware topology
    optimize_qubit_mapping(circuit, config);

    size_t final_depth = compute_depth(circuit, config);
    double final_fidelity = compute_sequence_fidelity(circuit->gates, circuit->num_gates, config);

    printf("Optimized circuit: depth=%zu, fidelity=%.6f\n", final_depth, final_fidelity);
    printf("Gate reduction: %zu -> %zu\n", initial_depth, final_depth);
}

// Create Rigetti configuration
OptRigettiConfig* create_rigetti_config(size_t num_qubits) {
    OptRigettiConfig* config = calloc(1, sizeof(OptRigettiConfig));
    if (!config) return NULL;

    config->num_qubits = num_qubits;
    config->t1_time = RIGETTI_T1_TIME;
    config->t2_time = RIGETTI_T2_TIME;
    config->gate_time = RIGETTI_GATE_TIME;

    // Allocate crosstalk map (initialized to zero - no crosstalk)
    config->crosstalk_map = calloc(num_qubits * num_qubits, sizeof(double));

    // Allocate connectivity (fully connected by default)
    config->connectivity = calloc(num_qubits * num_qubits, sizeof(size_t));
    if (config->connectivity) {
        for (size_t i = 0; i < num_qubits; i++) {
            for (size_t j = 0; j < num_qubits; j++) {
                if (i != j) {
                    config->connectivity[i * num_qubits + j] = 1;
                }
            }
        }
    }

    return config;
}

// Destroy Rigetti configuration
void destroy_rigetti_config(OptRigettiConfig* config) {
    if (!config) return;
    free(config->crosstalk_map);
    free(config->connectivity);
    free(config->gate_errors);
    free(config->readout_errors);
    free(config);
}
