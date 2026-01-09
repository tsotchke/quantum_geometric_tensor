#include "quantum_geometric/hybrid/quantum_classical_algorithms.h"
#include "quantum_geometric/hybrid/quantum_classical_orchestrator.h"
#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/hardware/quantum_circuit_optimization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

// Algorithm parameters
#define MAX_VQE_ITERATIONS 100
#define MAX_QAOA_ITERATIONS 50
#define CONVERGENCE_THRESHOLD 1e-6

// Hamiltonian operator structure (internal implementation)
struct HamiltonianOperator {
    size_t num_qubits;
    size_t num_terms;
    double* coefficients;
    char** pauli_strings;  // e.g., "XZZI", "IIZY"
    void* auxiliary_data;
};

// Function declarations
static double vqe_objective(const double* parameters,
                          double* gradients,
                          void* data);
static double qaoa_objective(const double* parameters,
                           double* gradients,
                           void* data);
static quantum_circuit_t* create_mixer_circuit(size_t num_qubits);
static quantum_circuit_t* build_qaoa_circuit(const quantum_circuit_t* problem,
                                            const quantum_circuit_t* mixer,
                                            const double* parameters,
                                            size_t depth);

// ============================================================================
// Hamiltonian Operations
// ============================================================================

// Copy a Hamiltonian operator
static HamiltonianOperator* copy_hamiltonian(const HamiltonianOperator* src) {
    if (!src) return NULL;

    HamiltonianOperator* dst = malloc(sizeof(HamiltonianOperator));
    if (!dst) return NULL;

    dst->num_qubits = src->num_qubits;
    dst->num_terms = src->num_terms;
    dst->auxiliary_data = NULL;

    // Copy coefficients
    dst->coefficients = malloc(src->num_terms * sizeof(double));
    if (!dst->coefficients) {
        free(dst);
        return NULL;
    }
    memcpy(dst->coefficients, src->coefficients, src->num_terms * sizeof(double));

    // Copy Pauli strings
    dst->pauli_strings = malloc(src->num_terms * sizeof(char*));
    if (!dst->pauli_strings) {
        free(dst->coefficients);
        free(dst);
        return NULL;
    }

    for (size_t i = 0; i < src->num_terms; i++) {
        if (src->pauli_strings[i]) {
            dst->pauli_strings[i] = strdup(src->pauli_strings[i]);
            if (!dst->pauli_strings[i]) {
                // Cleanup on failure
                for (size_t j = 0; j < i; j++) {
                    free(dst->pauli_strings[j]);
                }
                free(dst->pauli_strings);
                free(dst->coefficients);
                free(dst);
                return NULL;
            }
        } else {
            dst->pauli_strings[i] = NULL;
        }
    }

    return dst;
}

// Cleanup Hamiltonian operator
static void cleanup_hamiltonian(HamiltonianOperator* hamiltonian) {
    if (!hamiltonian) return;

    if (hamiltonian->pauli_strings) {
        for (size_t i = 0; i < hamiltonian->num_terms; i++) {
            free(hamiltonian->pauli_strings[i]);
        }
        free(hamiltonian->pauli_strings);
    }

    free(hamiltonian->coefficients);
    free(hamiltonian);
}

// ============================================================================
// Circuit Parameter Operations
// ============================================================================

// Forward declarations
static void update_circuit_parameters(quantum_circuit_t* circuit, const double* parameters);

// Count parameters in a circuit (parameterized gates)
static size_t count_parameters(const quantum_circuit_t* circuit) {
    if (!circuit) return 0;

    size_t count = 0;

    // Count rotation gates which have parameters
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            gate_type_t type = circuit->gates[i]->type;
            // Rotation gates have parameters
            if (type == GATE_RX || type == GATE_RY || type == GATE_RZ ||
                type == GATE_U1 || type == GATE_U2 || type == GATE_U3 ||
                type == GATE_CRX || type == GATE_CRY || type == GATE_CRZ) {
                count++;
            }
        }
    }

    return count > 0 ? count : 1;  // At least 1 parameter for circuits without explicit params
}

// Extract current parameters from circuit into array
static void extract_circuit_parameters(const quantum_circuit_t* circuit, double* parameters) {
    if (!circuit || !parameters) return;

    size_t param_idx = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            gate_type_t type = circuit->gates[i]->type;
            if (type == GATE_RX || type == GATE_RY || type == GATE_RZ ||
                type == GATE_U1 || type == GATE_U2 || type == GATE_U3 ||
                type == GATE_CRX || type == GATE_CRY || type == GATE_CRZ) {
                if (circuit->gates[i]->parameters) {
                    parameters[param_idx] = circuit->gates[i]->parameters[0];
                } else {
                    parameters[param_idx] = 0.0;
                }
                param_idx++;
            }
        }
    }
}

// ============================================================================
// Statevector Simulation Engine
// ============================================================================

// Apply single-qubit gate to statevector
// gate_matrix is 2x2 in row-major order
static void apply_single_qubit_gate(double complex* statevector, size_t num_qubits,
                                   size_t target, const double complex gate_matrix[4]) {
    size_t dim = 1ULL << num_qubits;
    size_t target_mask = 1ULL << target;

    // Process pairs of amplitudes that differ only in the target qubit
    for (size_t i = 0; i < dim; i++) {
        if ((i & target_mask) == 0) {
            size_t j = i | target_mask;  // Partner state with target qubit flipped

            double complex a0 = statevector[i];
            double complex a1 = statevector[j];

            // Apply 2x2 gate: [a0', a1']^T = G * [a0, a1]^T
            statevector[i] = gate_matrix[0] * a0 + gate_matrix[1] * a1;
            statevector[j] = gate_matrix[2] * a0 + gate_matrix[3] * a1;
        }
    }
}

// Apply controlled-NOT gate to statevector
static void apply_cnot_gate(double complex* statevector, size_t num_qubits,
                           size_t control, size_t target) {
    size_t dim = 1ULL << num_qubits;
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    for (size_t i = 0; i < dim; i++) {
        // Only apply X to target when control is |1⟩
        if ((i & control_mask) && !(i & target_mask)) {
            size_t j = i | target_mask;  // Target qubit flipped
            double complex temp = statevector[i];
            statevector[i] = statevector[j];
            statevector[j] = temp;
        }
    }
}

// Apply controlled-Z gate to statevector
static void apply_cz_gate(double complex* statevector, size_t num_qubits,
                         size_t control, size_t target) {
    size_t dim = 1ULL << num_qubits;
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    for (size_t i = 0; i < dim; i++) {
        // Apply phase flip when both qubits are |1⟩
        if ((i & control_mask) && (i & target_mask)) {
            statevector[i] *= -1.0;
        }
    }
}

// Standard quantum gates as 2x2 matrices
static void get_gate_matrix(gate_type_t type, double param, double complex gate_matrix[4]) {
    switch (type) {
        case GATE_X:  // Pauli X
            gate_matrix[0] = 0.0; gate_matrix[1] = 1.0;
            gate_matrix[2] = 1.0; gate_matrix[3] = 0.0;
            break;
        case GATE_Y:  // Pauli Y
            gate_matrix[0] = 0.0;       gate_matrix[1] = -I;
            gate_matrix[2] = I;         gate_matrix[3] = 0.0;
            break;
        case GATE_Z:  // Pauli Z
            gate_matrix[0] = 1.0; gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0; gate_matrix[3] = -1.0;
            break;
        case GATE_H:  // Hadamard
            gate_matrix[0] = M_SQRT1_2; gate_matrix[1] = M_SQRT1_2;
            gate_matrix[2] = M_SQRT1_2; gate_matrix[3] = -M_SQRT1_2;
            break;
        case GATE_S:  // S gate (sqrt(Z))
            gate_matrix[0] = 1.0; gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0; gate_matrix[3] = I;
            break;
        case GATE_T:  // T gate
            gate_matrix[0] = 1.0; gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0; gate_matrix[3] = cexp(I * M_PI / 4.0);
            break;
        case GATE_RX:  // Rx(θ) = exp(-iθX/2)
            gate_matrix[0] = cos(param / 2.0);         gate_matrix[1] = -I * sin(param / 2.0);
            gate_matrix[2] = -I * sin(param / 2.0);    gate_matrix[3] = cos(param / 2.0);
            break;
        case GATE_RY:  // Ry(θ) = exp(-iθY/2)
            gate_matrix[0] = cos(param / 2.0);  gate_matrix[1] = -sin(param / 2.0);
            gate_matrix[2] = sin(param / 2.0);  gate_matrix[3] = cos(param / 2.0);
            break;
        case GATE_RZ:  // Rz(θ) = exp(-iθZ/2)
            gate_matrix[0] = cexp(-I * param / 2.0); gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0;                    gate_matrix[3] = cexp(I * param / 2.0);
            break;
        case GATE_U1:  // U1(λ) = Rz(λ) up to global phase
            gate_matrix[0] = 1.0;              gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0;              gate_matrix[3] = cexp(I * param);
            break;
        default:  // Identity
            gate_matrix[0] = 1.0; gate_matrix[1] = 0.0;
            gate_matrix[2] = 0.0; gate_matrix[3] = 1.0;
            break;
    }
}

// Execute quantum circuit on statevector
static bool execute_circuit_on_statevector(const quantum_circuit_t* circuit,
                                          double complex* statevector) {
    if (!circuit || !statevector) return false;

    size_t num_qubits = circuit->num_qubits;

    for (size_t g = 0; g < circuit->num_gates; g++) {
        quantum_gate_t* gate = circuit->gates[g];
        if (!gate) continue;

        double param = (gate->parameters && gate->num_parameters > 0) ?
                       gate->parameters[0] : 0.0;

        // Two-qubit gates
        if (gate->type == GATE_CNOT || gate->type == GATE_CX) {
            if (gate->num_controls > 0 && gate->num_qubits > 0) {
                apply_cnot_gate(statevector, num_qubits,
                               gate->control_qubits[0], gate->target_qubits[0]);
            }
        } else if (gate->type == GATE_CZ) {
            if (gate->num_controls > 0 && gate->num_qubits > 0) {
                apply_cz_gate(statevector, num_qubits,
                             gate->control_qubits[0], gate->target_qubits[0]);
            }
        } else {
            // Single-qubit gates
            double complex gate_matrix[4];
            get_gate_matrix(gate->type, param, gate_matrix);

            for (size_t q = 0; q < gate->num_qubits; q++) {
                apply_single_qubit_gate(statevector, num_qubits,
                                       gate->target_qubits[q], gate_matrix);
            }
        }
    }

    return true;
}

// Compute expectation value of a Pauli string observable
// pauli_string is a string like "XZIY" where each character specifies the Pauli
// operator on that qubit (I=identity, X, Y, Z)
static double compute_pauli_expectation(const double complex* statevector,
                                       size_t num_qubits,
                                       const char* pauli_string) {
    if (!statevector || !pauli_string) return 0.0;

    size_t dim = 1ULL << num_qubits;
    size_t len = strlen(pauli_string);
    if (len != num_qubits) return 0.0;  // Pauli string must match qubit count

    // For Pauli strings, we compute ⟨ψ|P|ψ⟩
    // We create P|ψ⟩ and compute inner product with |ψ⟩
    double complex* temp_state = (double complex*)malloc(dim * sizeof(double complex));
    if (!temp_state) return 0.0;

    // Copy statevector
    memcpy(temp_state, statevector, dim * sizeof(double complex));

    // Apply Pauli operators from right to left (tensor product order)
    for (size_t q = 0; q < num_qubits; q++) {
        char pauli = pauli_string[num_qubits - 1 - q];  // Reverse order for tensor product
        double complex gate_matrix[4];

        switch (pauli) {
            case 'I':
            case 'i':
                // Identity - no operation needed
                continue;
            case 'X':
            case 'x':
                gate_matrix[0] = 0.0; gate_matrix[1] = 1.0;
                gate_matrix[2] = 1.0; gate_matrix[3] = 0.0;
                break;
            case 'Y':
            case 'y':
                gate_matrix[0] = 0.0;  gate_matrix[1] = -I;
                gate_matrix[2] = I;    gate_matrix[3] = 0.0;
                break;
            case 'Z':
            case 'z':
                gate_matrix[0] = 1.0;  gate_matrix[1] = 0.0;
                gate_matrix[2] = 0.0;  gate_matrix[3] = -1.0;
                break;
            default:
                // Unknown Pauli, treat as identity
                continue;
        }

        apply_single_qubit_gate(temp_state, num_qubits, q, gate_matrix);
    }

    // Compute inner product ⟨ψ|P|ψ⟩ = Σ_i conj(ψ_i) * (Pψ)_i
    double complex expectation = 0.0;
    for (size_t i = 0; i < dim; i++) {
        expectation += conj(statevector[i]) * temp_state[i];
    }

    free(temp_state);

    // Expectation value must be real for Hermitian operators
    return creal(expectation);
}

// Compute expectation value for a specific parameter configuration
// This is the core quantum computation - full statevector simulation
static double evaluate_circuit_expectation(quantum_circuit_t* circuit,
                                          const HamiltonianOperator* hamiltonian,
                                          const double* parameters) {
    if (!circuit || !hamiltonian || !parameters) return INFINITY;

    size_t num_qubits = circuit->num_qubits;
    if (num_qubits == 0 || num_qubits > 24) return INFINITY;  // Limit for classical simulation

    size_t dim = 1ULL << num_qubits;

    // Update circuit with given parameters
    update_circuit_parameters(circuit, parameters);

    // Allocate statevector initialized to |0...0⟩
    double complex* statevector = (double complex*)calloc(dim, sizeof(double complex));
    if (!statevector) return INFINITY;
    statevector[0] = 1.0;  // Initial state |0...0⟩

    // Execute circuit to prepare the variational state |ψ(θ)⟩
    if (!execute_circuit_on_statevector(circuit, statevector)) {
        free(statevector);
        return INFINITY;
    }

    // Compute expectation value ⟨ψ(θ)|H|ψ(θ)⟩
    // H = Σ_i c_i P_i where P_i are Pauli strings
    double energy = 0.0;

    for (size_t i = 0; i < hamiltonian->num_terms; i++) {
        double coeff = hamiltonian->coefficients[i];
        const char* pauli_string = hamiltonian->pauli_strings[i];

        if (pauli_string) {
            double term_expectation = compute_pauli_expectation(statevector, num_qubits,
                                                                pauli_string);
            energy += coeff * term_expectation;
        }
    }

    free(statevector);
    return energy;
}

// Update circuit parameters (for variational algorithms)
static void update_circuit_parameters(quantum_circuit_t* circuit,
                                     const double* parameters) {
    if (!circuit || !parameters) return;

    size_t param_idx = 0;

    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i]) {
            gate_type_t type = circuit->gates[i]->type;
            // Update rotation gate parameters
            if (type == GATE_RX || type == GATE_RY || type == GATE_RZ ||
                type == GATE_U1 || type == GATE_U2 || type == GATE_U3 ||
                type == GATE_CRX || type == GATE_CRY || type == GATE_CRZ) {
                if (circuit->gates[i]->parameters) {
                    circuit->gates[i]->parameters[0] = parameters[param_idx];
                }
                param_idx++;
            }
        }
    }
}

// Append parameterized circuit with scaling factor
static qgt_error_t append_parameterized_circuit(quantum_circuit_t* target,
                                               const quantum_circuit_t* source,
                                               double scale) {
    if (!target || !source) return QGT_ERROR_INVALID_PARAMETER;

    // Copy gates from source to target with scaled parameters
    for (size_t i = 0; i < source->num_gates; i++) {
        if (source->gates[i]) {
            quantum_gate_t* gate = source->gates[i];

            // For rotation gates, scale the parameter
            if (gate->type == GATE_RX || gate->type == GATE_RY || gate->type == GATE_RZ) {
                double angle = gate->parameters ? gate->parameters[0] * scale : scale;
                // Add rotation to target with scaled angle
                for (size_t q = 0; q < gate->num_qubits; q++) {
                    size_t qubit = gate->target_qubits[q];
                    qgt_error_t err = quantum_circuit_rotation(target, qubit, angle,
                        gate->type == GATE_RX ? PAULI_X :
                        gate->type == GATE_RY ? PAULI_Y : PAULI_Z);
                    if (err != QGT_SUCCESS) return err;
                }
            } else {
                // Clone non-parameterized gates based on their type
                qgt_error_t err = QGT_SUCCESS;
                size_t target_qubit = gate->target_qubits ? gate->target_qubits[0] : 0;
                size_t control_qubit = (gate->control_qubits && gate->num_controls > 0) ?
                                        gate->control_qubits[0] : 0;

                switch (gate->type) {
                    case GATE_X:
                        err = quantum_circuit_pauli_x(target, target_qubit);
                        break;
                    case GATE_Y:
                        err = quantum_circuit_pauli_y(target, target_qubit);
                        break;
                    case GATE_Z:
                        err = quantum_circuit_pauli_z(target, target_qubit);
                        break;
                    case GATE_H:
                        err = quantum_circuit_hadamard(target, target_qubit);
                        break;
                    case GATE_S:
                        // S gate = phase by π/2
                        err = quantum_circuit_phase(target, target_qubit, M_PI / 2.0);
                        break;
                    case GATE_T:
                        // T gate = phase by π/4
                        err = quantum_circuit_phase(target, target_qubit, M_PI / 4.0);
                        break;
                    case GATE_CNOT:
                        // GATE_CX is an alias, handled by same case
                        if (gate->num_controls > 0) {
                            err = quantum_circuit_cnot(target, control_qubit, target_qubit);
                        }
                        break;
                    case GATE_CZ:
                        if (gate->num_controls > 0) {
                            err = quantum_circuit_cz(target, control_qubit, target_qubit);
                        }
                        break;
                    case GATE_SWAP:
                        if (gate->num_qubits >= 2 && gate->target_qubits) {
                            err = quantum_circuit_swap(target, gate->target_qubits[0],
                                                       gate->target_qubits[1]);
                        }
                        break;
                    case GATE_TOFFOLI:
                        // GATE_CCX is an alias, handled by same case
                        // Toffoli decomposition using T, CNOT sequence
                        if (gate->num_controls >= 2 && gate->control_qubits) {
                            size_t c1 = gate->control_qubits[0];
                            size_t c2 = gate->control_qubits[1];
                            quantum_circuit_phase(target, target_qubit, M_PI / 4.0);
                            quantum_circuit_phase(target, c1, M_PI / 4.0);
                            quantum_circuit_phase(target, c2, M_PI / 4.0);
                            quantum_circuit_cnot(target, c1, target_qubit);
                            quantum_circuit_phase(target, target_qubit, -M_PI / 4.0);
                            quantum_circuit_cnot(target, c2, target_qubit);
                            quantum_circuit_phase(target, target_qubit, M_PI / 4.0);
                            quantum_circuit_cnot(target, c1, target_qubit);
                            quantum_circuit_phase(target, target_qubit, -M_PI / 4.0);
                            quantum_circuit_cnot(target, c2, c1);
                            quantum_circuit_phase(target, c1, -M_PI / 4.0);
                            err = quantum_circuit_cnot(target, c2, c1);
                        }
                        break;
                    case GATE_U1:
                    case GATE_U2:
                    case GATE_U3:
                        // U gates decomposed to rotations: U3(θ,φ,λ) = Rz(φ)Ry(θ)Rz(λ)
                        if (gate->parameters && gate->num_parameters > 0) {
                            double theta = gate->parameters[0];
                            double phi = gate->num_parameters > 1 ? gate->parameters[1] : 0.0;
                            double lambda = gate->num_parameters > 2 ? gate->parameters[2] : 0.0;
                            quantum_circuit_rotation(target, target_qubit, lambda, PAULI_Z);
                            quantum_circuit_rotation(target, target_qubit, theta, PAULI_Y);
                            err = quantum_circuit_rotation(target, target_qubit, phi, PAULI_Z);
                        }
                        break;
                    case GATE_CRX:
                    case GATE_CRY:
                    case GATE_CRZ:
                        // Controlled rotation: CR(θ) = R(θ/2) CNOT R(-θ/2) CNOT
                        if (gate->parameters && gate->num_controls > 0) {
                            double angle = gate->parameters[0] * scale;
                            pauli_type axis = gate->type == GATE_CRX ? PAULI_X :
                                             gate->type == GATE_CRY ? PAULI_Y : PAULI_Z;
                            quantum_circuit_rotation(target, target_qubit, angle / 2.0, axis);
                            quantum_circuit_cnot(target, control_qubit, target_qubit);
                            quantum_circuit_rotation(target, target_qubit, -angle / 2.0, axis);
                            err = quantum_circuit_cnot(target, control_qubit, target_qubit);
                        }
                        break;
                    default:
                        // For other gates, skip (identity operation)
                        break;
                }
                if (err != QGT_SUCCESS) return err;
            }
        }
    }

    return QGT_SUCCESS;
}

// ============================================================================
// Expectation Value Computation
// ============================================================================

// Compute expectation value of Hamiltonian for VQE with parameter shift gradients
static double compute_expectation_value(quantum_circuit_t* circuit,
                                       const HamiltonianOperator* hamiltonian,
                                       double* gradients) {
    if (!circuit || !hamiltonian) return INFINITY;

    // Evaluate the circuit at current parameters
    double energy = evaluate_circuit_expectation(circuit, hamiltonian, NULL);

    // Compute gradients using parameter shift rule:
    // ∂⟨H⟩/∂θ = (⟨H⟩_{θ+π/2} - ⟨H⟩_{θ-π/2}) / 2
    if (gradients) {
        size_t num_params = count_parameters(circuit);
        double shift = M_PI / 2.0;

        // Extract current parameters
        double* params = malloc(num_params * sizeof(double));
        if (!params) return energy;

        size_t param_idx = 0;
        for (size_t i = 0; i < circuit->num_gates && param_idx < num_params; i++) {
            if (circuit->gates[i]) {
                gate_type_t type = circuit->gates[i]->type;
                if (type == GATE_RX || type == GATE_RY || type == GATE_RZ ||
                    type == GATE_U1 || type == GATE_U2 || type == GATE_U3 ||
                    type == GATE_CRX || type == GATE_CRY || type == GATE_CRZ) {
                    params[param_idx] = circuit->gates[i]->parameters ?
                                        circuit->gates[i]->parameters[0] : 0.0;
                    param_idx++;
                }
            }
        }

        // Compute gradient for each parameter using parameter shift rule
        for (size_t p = 0; p < num_params; p++) {
            double original = params[p];

            // Shift parameter by +π/2
            params[p] = original + shift;
            update_circuit_parameters(circuit, params);
            double e_plus = evaluate_circuit_expectation(circuit, hamiltonian, NULL);

            // Shift parameter by -π/2
            params[p] = original - shift;
            update_circuit_parameters(circuit, params);
            double e_minus = evaluate_circuit_expectation(circuit, hamiltonian, NULL);

            // Parameter shift gradient: (E+ - E-) / 2
            gradients[p] = (e_plus - e_minus) / 2.0;

            // Restore original parameter
            params[p] = original;
        }

        // Restore original circuit parameters
        update_circuit_parameters(circuit, params);
        free(params);
    }

    return energy;
}

// Compute QAOA cost function using statevector simulation
// QAOA objective: ⟨ψ(γ,β)|C|ψ(γ,β)⟩ where C is the cost Hamiltonian
static double compute_qaoa_cost(quantum_circuit_t* circuit,
                               double* gradients) {
    if (!circuit) return INFINITY;

    size_t num_qubits = circuit->num_qubits;
    size_t dim = 1ULL << num_qubits;

    // Allocate statevector
    double complex* statevector = calloc(dim, sizeof(double complex));
    if (!statevector) return INFINITY;
    statevector[0] = 1.0;  // |0...0⟩

    // Execute QAOA circuit
    if (!execute_circuit_on_statevector(circuit, statevector)) {
        free(statevector);
        return INFINITY;
    }

    // Compute cost expectation: ⟨ψ|C|ψ⟩
    // For MaxCut: C = Σ_{(i,j)∈E} (1 - Z_i Z_j)/2
    // Here we compute a generic diagonal cost
    double cost = 0.0;
    for (size_t i = 0; i < dim; i++) {
        double prob = cabs(statevector[i]) * cabs(statevector[i]);
        // Cost function based on bitstring: count number of 1s (simple MaxCut-like)
        int ones = 0;
        for (size_t b = 0; b < num_qubits; b++) {
            if (i & (1ULL << b)) ones++;
        }
        // Cost contribution: number of pairs with different bits
        int zeros = (int)num_qubits - ones;
        cost += prob * (double)(ones * zeros);
    }

    free(statevector);

    // Compute gradients using parameter shift rule
    if (gradients) {
        size_t num_params = count_parameters(circuit);
        double shift = M_PI / 2.0;

        double* params = malloc(num_params * sizeof(double));
        if (!params) return cost;

        // Extract current parameters
        size_t param_idx = 0;
        for (size_t i = 0; i < circuit->num_gates && param_idx < num_params; i++) {
            if (circuit->gates[i]) {
                gate_type_t type = circuit->gates[i]->type;
                if (type == GATE_RX || type == GATE_RY || type == GATE_RZ ||
                    type == GATE_U1 || type == GATE_U2 || type == GATE_U3) {
                    params[param_idx] = circuit->gates[i]->parameters ?
                                        circuit->gates[i]->parameters[0] : 0.0;
                    param_idx++;
                }
            }
        }

        // Parameter shift for each parameter
        for (size_t p = 0; p < num_params; p++) {
            double original = params[p];

            // +shift
            params[p] = original + shift;
            update_circuit_parameters(circuit, params);
            double c_plus = compute_qaoa_cost(circuit, NULL);  // Recursive but without gradients

            // -shift
            params[p] = original - shift;
            update_circuit_parameters(circuit, params);
            double c_minus = compute_qaoa_cost(circuit, NULL);

            gradients[p] = (c_plus - c_minus) / 2.0;
            params[p] = original;
        }

        update_circuit_parameters(circuit, params);
        free(params);
    }

    return cost;
}

// ============================================================================
// VQE Implementation
// ============================================================================

// Initialize VQE
VQEContext* init_vqe(const quantum_circuit_t* ansatz,
                    const HamiltonianOperator* hamiltonian) {
    VQEContext* ctx = malloc(sizeof(VQEContext));
    if (!ctx) return NULL;

    // Copy ansatz circuit
    ctx->ansatz = quantum_circuit_create(ansatz->num_qubits);
    if (!ctx->ansatz) {
        free(ctx);
        return NULL;
    }

    // Copy Hamiltonian
    ctx->hamiltonian = copy_hamiltonian(hamiltonian);
    if (!ctx->hamiltonian) {
        quantum_circuit_destroy(ctx->ansatz);
        free(ctx);
        return NULL;
    }

    // Count parameters in ansatz
    ctx->num_parameters = count_parameters(ansatz);

    // Initialize parameters
    ctx->parameters = aligned_alloc(64,
        ctx->num_parameters * sizeof(double));
    if (!ctx->parameters) {
        cleanup_hamiltonian(ctx->hamiltonian);
        quantum_circuit_destroy(ctx->ansatz);
        free(ctx);
        return NULL;
    }

    // Initialize optimizer
    ctx->optimizer = init_classical_optimizer(
        OPTIMIZER_ADAM,
        ctx->num_parameters,
        true  // Use GPU
    );

    if (!ctx->optimizer) {
        free(ctx->parameters);
        cleanup_hamiltonian(ctx->hamiltonian);
        quantum_circuit_destroy(ctx->ansatz);
        free(ctx);
        return NULL;
    }

    // Initialize parameters randomly
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ctx->parameters[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }

    ctx->current_energy = INFINITY;

    return ctx;
}

// ============================================================================
// QAOA Implementation
// ============================================================================

// Initialize QAOA
QAOAContext* init_qaoa(const quantum_circuit_t* problem,
                      size_t depth) {
    QAOAContext* ctx = malloc(sizeof(QAOAContext));
    if (!ctx) return NULL;

    // Copy problem circuit
    ctx->problem = quantum_circuit_create(problem->num_qubits);
    if (!ctx->problem) {
        free(ctx);
        return NULL;
    }

    // Create mixer circuit
    ctx->mixer = create_mixer_circuit(problem->num_qubits);
    if (!ctx->mixer) {
        quantum_circuit_destroy(ctx->problem);
        free(ctx);
        return NULL;
    }

    ctx->depth = depth;
    ctx->num_parameters = 2 * depth;  // gamma and beta for each layer

    // Initialize parameters
    ctx->parameters = aligned_alloc(64,
        ctx->num_parameters * sizeof(double));
    if (!ctx->parameters) {
        quantum_circuit_destroy(ctx->mixer);
        quantum_circuit_destroy(ctx->problem);
        free(ctx);
        return NULL;
    }

    // Initialize optimizer
    ctx->optimizer = init_classical_optimizer(
        OPTIMIZER_ADAM,
        ctx->num_parameters,
        true  // Use GPU
    );

    if (!ctx->optimizer) {
        free(ctx->parameters);
        quantum_circuit_destroy(ctx->mixer);
        quantum_circuit_destroy(ctx->problem);
        free(ctx);
        return NULL;
    }

    // Initialize parameters randomly
    for (size_t i = 0; i < ctx->num_parameters; i++) {
        ctx->parameters[i] = ((double)rand() / RAND_MAX) * M_PI;
    }

    ctx->current_cost = INFINITY;

    return ctx;
}

// Run VQE algorithm
int run_vqe(VQEContext* ctx,
            HybridOrchestrator* orchestrator,
            VQEResult* result) {
    if (!ctx || !orchestrator || !result) return -1;

    // Set up optimization objective
    OptimizationObjective objective = {
        .function = vqe_objective,
        .data = ctx
    };

    // Run optimization
    int status = optimize_parameters(ctx->optimizer,
                                  objective,
                                  orchestrator);

    if (status == 0) {
        // Store results
        result->energy = ctx->current_energy;
        result->num_parameters = ctx->num_parameters;
        result->parameters = malloc(
            ctx->num_parameters * sizeof(double));

        if (result->parameters) {
            memcpy(result->parameters,
                   ctx->parameters,
                   ctx->num_parameters * sizeof(double));
        }
    }

    return status;
}

// Run QAOA algorithm
int run_qaoa(QAOAContext* ctx,
             HybridOrchestrator* orchestrator,
             QAOAResult* result) {
    if (!ctx || !orchestrator || !result) return -1;

    // Set up optimization objective
    OptimizationObjective objective = {
        .function = qaoa_objective,
        .data = ctx
    };

    // Run optimization
    int status = optimize_parameters(ctx->optimizer,
                                  objective,
                                  orchestrator);

    if (status == 0) {
        // Store results
        result->cost = ctx->current_cost;
        result->num_parameters = ctx->num_parameters;
        result->parameters = malloc(
            ctx->num_parameters * sizeof(double));

        if (result->parameters) {
            memcpy(result->parameters,
                   ctx->parameters,
                   ctx->num_parameters * sizeof(double));
        }
    }

    return status;
}

// VQE objective function
static double vqe_objective(const double* parameters,
                          double* gradients,
                          void* data) {
    VQEContext* ctx = (VQEContext*)data;

    // Update ansatz parameters
    update_circuit_parameters(ctx->ansatz, parameters);

    // Compute energy and gradients
    double energy = compute_expectation_value(
        ctx->ansatz,
        ctx->hamiltonian,
        gradients);

    ctx->current_energy = energy;
    return energy;
}

// QAOA objective function
static double qaoa_objective(const double* parameters,
                           double* gradients,
                           void* data) {
    QAOAContext* ctx = (QAOAContext*)data;

    // Build QAOA circuit
    quantum_circuit_t* qaoa_circuit = build_qaoa_circuit(
        ctx->problem,
        ctx->mixer,
        parameters,
        ctx->depth);

    if (!qaoa_circuit) return INFINITY;

    // Compute cost and gradients
    double cost = compute_qaoa_cost(
        qaoa_circuit,
        gradients);

    quantum_circuit_destroy(qaoa_circuit);

    ctx->current_cost = cost;
    return cost;
}

// Helper functions

static quantum_circuit_t* create_mixer_circuit(size_t num_qubits) {
    quantum_circuit_t* mixer = quantum_circuit_create(num_qubits);
    if (!mixer) return NULL;

    // Add X rotations for each qubit
    for (size_t i = 0; i < num_qubits; i++) {
        qgt_error_t err = quantum_circuit_rotation(mixer, i, 0.0, PAULI_X);
        if (err != QGT_SUCCESS) {
            quantum_circuit_destroy(mixer);
            return NULL;
        }
    }

    return mixer;
}

static quantum_circuit_t* build_qaoa_circuit(const quantum_circuit_t* problem,
                                            const quantum_circuit_t* mixer,
                                            const double* parameters,
                                            size_t depth) {
    size_t num_qubits = problem->num_qubits;

    // Initialize circuit
    quantum_circuit_t* qaoa = quantum_circuit_create(num_qubits);
    if (!qaoa) return NULL;

    // Add initial Hadamard layer
    for (size_t i = 0; i < num_qubits; i++) {
        qgt_error_t err = quantum_circuit_hadamard(qaoa, i);
        if (err != QGT_SUCCESS) {
            quantum_circuit_destroy(qaoa);
            return NULL;
        }
    }

    // Add QAOA layers
    for (size_t d = 0; d < depth; d++) {
        // Problem unitary with gamma
        double gamma = parameters[2 * d];
        if (append_parameterized_circuit(qaoa, problem, gamma) != QGT_SUCCESS) {
            quantum_circuit_destroy(qaoa);
            return NULL;
        }

        // Mixer unitary with beta
        double beta = parameters[2 * d + 1];
        if (append_parameterized_circuit(qaoa, mixer, beta) != QGT_SUCCESS) {
            quantum_circuit_destroy(qaoa);
            return NULL;
        }
    }

    return qaoa;
}

// Clean up VQE
void cleanup_vqe(VQEContext* ctx) {
    if (!ctx) return;

    quantum_circuit_destroy(ctx->ansatz);
    cleanup_hamiltonian(ctx->hamiltonian);
    cleanup_classical_optimizer(ctx->optimizer);
    free(ctx->parameters);
    free(ctx);
}

// Clean up QAOA
void cleanup_qaoa(QAOAContext* ctx) {
    if (!ctx) return;

    quantum_circuit_destroy(ctx->problem);
    quantum_circuit_destroy(ctx->mixer);
    cleanup_classical_optimizer(ctx->optimizer);
    free(ctx->parameters);
    free(ctx);
}
