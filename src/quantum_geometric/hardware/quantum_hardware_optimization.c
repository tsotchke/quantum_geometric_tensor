/**
 * @file quantum_hardware_optimization.c
 * @brief Hardware-specific quantum circuit optimizations
 *
 * Provides optimization routines for different quantum hardware backends
 * including IBM, Rigetti, IonQ, and D-Wave systems.
 */

#include "quantum_geometric/hardware/quantum_error_constants.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// GATE_CX is same as GATE_CNOT - don't redefine, just use GATE_CNOT

// Additional gate type aliases for multi-qubit gates
#ifndef GATE_XX
#define GATE_XX 50
#endif
#ifndef GATE_YY
#define GATE_YY 51
#endif
#ifndef GATE_ZZ
#define GATE_ZZ 52
#endif
#ifndef GATE_CY
#define GATE_CY 53
#endif

// Forward declarations for helper functions
static bool can_fuse_gates(const HardwareGate* g1, const HardwareGate* g2);
static void fuse_gates(HardwareGate* g1, const HardwareGate* g2);
static double get_gate_time(const HardwareGate* gate);
static void split_long_gate(HardwareGate* gate, double max_time);
static void optimize_pulses(QuantumCircuit* circuit, double t1_time, double t2_time);
static double get_circuit_duration(const QuantumCircuit* circuit);
static void optimize_annealing_schedule(QuantumCircuit* circuit, size_t annealing_time);
static void add_chain_strength(QuantumCircuit* circuit, double strength);

// ============================================================================
// IBM Quantum Hardware Optimization
// ============================================================================

static void optimize_ibm_circuit_impl(QuantumCircuit* circuit, const IBMBackendConfig* config) {
    if (!circuit || !config) return;

    // Apply dynamic decoupling if enabled
    if (config->dynamic_decoupling) {
        for (size_t i = 0; i < circuit->num_gates; i++) {
            // Add X gates for dynamical decoupling around CNOT gates
            if (circuit->gates[i].type == GATE_CNOT) {
                HardwareGate x_gate = {
                    .type = GATE_X,
                    .target = circuit->gates[i].target,
                    .control = 0,
                    .parameter = 0.0
                };
                hw_insert_gate(circuit, i, x_gate);
                hw_insert_gate(circuit, i + 2, x_gate);
                i += 2; // Skip inserted gates
            }
        }
    }

    // Optimize circuit depth using gate fusion
    if (circuit->depth > (size_t)circuit->max_circuit_depth) {
        for (size_t i = 0; i + 1 < circuit->num_gates; i++) {
            if (can_fuse_gates(&circuit->gates[i], &circuit->gates[i + 1])) {
                fuse_gates(&circuit->gates[i], &circuit->gates[i + 1]);
                hw_remove_gate(circuit, i + 1);
                if (i > 0) i--; // Check new neighbor
            }
        }
    }
}

// ============================================================================
// Rigetti Quantum Hardware Optimization
// ============================================================================

static void optimize_rigetti_circuit_impl(QuantumCircuit* circuit, const struct RigettiConfig* config) {
    if (!circuit || !config) return;

    // Use native gates if possible - decompose to Rigetti native gate set
    for (size_t i = 0; i < circuit->num_gates; i++) {
        switch (circuit->gates[i].type) {
            case GATE_H: {
                // Replace H with Rx(π/2)Rz(π)Rx(π/2)
                HardwareGate rx1 = {
                    .type = GATE_RX,
                    .target = circuit->gates[i].target,
                    .parameter = M_PI_2
                };
                HardwareGate rz = {
                    .type = GATE_RZ,
                    .target = circuit->gates[i].target,
                    .parameter = M_PI
                };
                HardwareGate rx2 = rx1;

                hw_replace_gate(circuit, i, rx1);
                hw_insert_gate(circuit, i + 1, rz);
                hw_insert_gate(circuit, i + 2, rx2);
                i += 2; // Skip inserted gates
                break;
            }

            case GATE_CNOT: {
                // Replace CNOT with CZ and Rx gates (Rigetti native)
                HardwareGate rx_pre = {
                    .type = GATE_RX,
                    .target = circuit->gates[i].target,
                    .parameter = M_PI_2
                };
                HardwareGate cz = {
                    .type = GATE_CZ,
                    .control = circuit->gates[i].control,
                    .target = circuit->gates[i].target
                };
                HardwareGate rx_post = rx_pre;

                hw_replace_gate(circuit, i, rx_pre);
                hw_insert_gate(circuit, i + 1, cz);
                hw_insert_gate(circuit, i + 2, rx_post);
                i += 2; // Skip inserted gates
                break;
            }

            default:
                break;
        }
    }

    // Note: Pulse-level optimization requires extended config with t1/t2 times
    // This would be added via config->custom_data if needed
    (void)config; // Suppress unused warning
}

// ============================================================================
// IonQ Quantum Hardware Optimization
// ============================================================================

static void optimize_ionq_circuit_impl(QuantumCircuit* circuit, const IonQConfig* config) {
    if (!circuit || !config) return;

    // Use native XX gates when possible (IonQ's native 2-qubit gate)
    if (config->use_native_gates) {
        for (size_t i = 0; i + 1 < circuit->num_gates; i++) {
            // Convert adjacent RX gates on different qubits to XX gate
            if (circuit->gates[i].type == GATE_RX &&
                circuit->gates[i + 1].type == GATE_RX &&
                circuit->gates[i].target != circuit->gates[i + 1].target) {
                HardwareGate xx = {
                    .type = GATE_XX,
                    .target1 = circuit->gates[i].target,
                    .target2 = circuit->gates[i + 1].target,
                    .parameter = (circuit->gates[i].parameter +
                                circuit->gates[i + 1].parameter) / 2
                };
                hw_replace_gate(circuit, i, xx);
                hw_remove_gate(circuit, i + 1);
            }
        }
    }

    // Optimize gate timing - split long gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (get_gate_time(&circuit->gates[i]) > config->max_gate_time) {
            split_long_gate(&circuit->gates[i], config->max_gate_time);
        }
    }
}

// ============================================================================
// D-Wave Quantum Annealing Optimization
// ============================================================================

static void optimize_dwave_circuit_impl(QuantumCircuit* circuit, const struct DWaveConfig* config) {
    if (!circuit || !config) return;

    // D-Wave optimization is handled by the optimized backend directly
    // This function provides circuit-level optimizations for annealing problems

    // Basic optimization: ensure all gates are compatible with annealing
    // D-Wave primarily uses ZZ interactions for QUBO/Ising problems
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].type == GATE_ZZ) {
            // ZZ gates map directly to Ising couplings - no optimization needed
            continue;
        }
        // Other gates would need to be decomposed for annealing
    }

    (void)config; // Config parameters handled by backend
}

// ============================================================================
// Public API
// ============================================================================

HardwareOptimizations* init_hardware_optimizations(const char* backend_type) {
    HardwareOptimizations* opts = malloc(sizeof(HardwareOptimizations));
    if (!opts) return NULL;

    // Initialize all pointers to NULL
    memset(opts, 0, sizeof(HardwareOptimizations));

    // Set optimization functions based on backend type
    if (strcmp(backend_type, "ibm") == 0) {
        opts->ibm_optimize = optimize_ibm_circuit_impl;
    } else if (strcmp(backend_type, "rigetti") == 0) {
        opts->rigetti_optimize = optimize_rigetti_circuit_impl;
    } else if (strcmp(backend_type, "ionq") == 0) {
        opts->ionq_optimize = optimize_ionq_circuit_impl;
    } else if (strcmp(backend_type, "dwave") == 0) {
        opts->dwave_optimize = optimize_dwave_circuit_impl;
    } else {
        free(opts);
        return NULL;
    }

    return opts;
}

void cleanup_hardware_optimizations(HardwareOptimizations* opts) {
    free(opts);
}

// ============================================================================
// Helper Functions
// ============================================================================

static bool can_fuse_gates(const HardwareGate* g1, const HardwareGate* g2) {
    if (!g1 || !g2) return false;

    // Gates can be fused if they:
    // 1. Act on same qubit(s)
    // 2. Are of same type
    // 3. Have compatible parameters

    if (g1->type != g2->type) return false;

    switch (g1->type) {
        case GATE_RX:
        case GATE_RY:
        case GATE_RZ:
            return g1->target == g2->target;

        case GATE_XX:
        case GATE_YY:
        case GATE_ZZ:
            return (g1->target1 == g2->target1 && g1->target2 == g2->target2) ||
                   (g1->target1 == g2->target2 && g1->target2 == g2->target1);

        default:
            return false;
    }
}

static void fuse_gates(HardwareGate* g1, const HardwareGate* g2) {
    if (!g1 || !g2) return;

    // Add parameters for rotation gates
    switch (g1->type) {
        case GATE_RX:
        case GATE_RY:
        case GATE_RZ:
        case GATE_XX:
        case GATE_YY:
        case GATE_ZZ:
            g1->parameter += g2->parameter;
            break;

        default:
            break;
    }
}

static double get_gate_time(const HardwareGate* gate) {
    if (!gate) return 0.0;

    // Approximate gate times in microseconds
    switch (gate->type) {
        case GATE_X:
        case GATE_Y:
        case GATE_Z:
            return 0.1;

        case GATE_RX:
        case GATE_RY:
        case GATE_RZ:
            return 0.1 * fabs(gate->parameter) / M_PI;

        case GATE_CNOT:  // GATE_CX is same as GATE_CNOT
        case GATE_CY:
        case GATE_CZ:
            return 0.3;

        case GATE_XX:
        case GATE_YY:
        case GATE_ZZ:
            return 0.2 * fabs(gate->parameter) / M_PI;

        default:
            return 0.0;
    }
}

static void split_long_gate(HardwareGate* gate, double max_time) {
    if (!gate) return;

    double time = get_gate_time(gate);
    if (time <= max_time) return;

    // Split rotation into smaller rotations
    int num_splits = (int)ceil(time / max_time);
    if (num_splits > 0) {
        gate->parameter /= num_splits;
    }
}

static void optimize_pulses(QuantumCircuit* circuit, double t1_time, double t2_time) {
    if (!circuit || t2_time <= 0) return;

    // Scale pulse amplitudes based on coherence times
    double duration = get_circuit_duration(circuit);
    double scale = exp(-duration / t2_time);

    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].type == GATE_RX ||
            circuit->gates[i].type == GATE_RY ||
            circuit->gates[i].type == GATE_RZ) {
            circuit->gates[i].parameter *= scale;
        }
    }
    (void)t1_time; // Reserved for future T1-based optimization
}

static double get_circuit_duration(const QuantumCircuit* circuit) {
    if (!circuit) return 0.0;

    double duration = 0.0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        duration += get_gate_time(&circuit->gates[i]);
    }
    return duration;
}

static void optimize_annealing_schedule(QuantumCircuit* circuit, size_t annealing_time) {
    if (!circuit || annealing_time == 0) return;

    // Adjust ZZ coupling strengths based on annealing time
    double scale = sqrt((double)annealing_time / 1000.0); // Base scale on microseconds

    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].type == GATE_ZZ) {
            circuit->gates[i].parameter *= scale;
        }
    }
}

static void add_chain_strength(QuantumCircuit* circuit, double strength) {
    if (!circuit || strength <= 0) return;

    // Add ZZ interactions within chains for error mitigation
    for (size_t i = 0; i + 1 < circuit->num_qubits; i++) {
        HardwareGate zz = {
            .type = GATE_ZZ,
            .target1 = (uint32_t)i,
            .target2 = (uint32_t)(i + 1),
            .parameter = -strength // Ferromagnetic coupling
        };
        hw_add_gate(circuit, zz);
    }
}
