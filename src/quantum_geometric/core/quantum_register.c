#include "quantum_geometric/core/quantum_register.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

quantum_register_t* quantum_register_create(size_t num_qubits, int flags) {
    if (num_qubits == 0 || num_qubits > 30) {
        return NULL;
    }

    quantum_register_t* reg = malloc(sizeof(quantum_register_t));
    if (!reg) {
        return NULL;
    }

    size_t size = 1ULL << num_qubits;
    reg->size = size;
    reg->amplitudes = calloc(size, sizeof(ComplexFloat));
    reg->system = NULL;

    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    // Initialize to |0⟩ state
    reg->amplitudes[0].real = 1.0f;
    reg->amplitudes[0].imag = 0.0f;

    return reg;
}

quantum_register_t* quantum_register_create_empty(size_t size) {
    if (size == 0) {
        return NULL;
    }

    quantum_register_t* reg = malloc(sizeof(quantum_register_t));
    if (!reg) {
        return NULL;
    }

    reg->size = size;
    reg->amplitudes = calloc(size, sizeof(ComplexFloat));
    reg->system = NULL;

    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    return reg;
}

quantum_register_t* quantum_register_create_state(const complex double* amplitudes, size_t size, quantum_system_t* system) {
    if (!amplitudes || size == 0) {
        return NULL;
    }

    quantum_register_t* reg = malloc(sizeof(quantum_register_t));
    if (!reg) {
        return NULL;
    }

    reg->size = size;
    reg->amplitudes = malloc(size * sizeof(ComplexFloat));
    reg->system = system;

    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    // Convert from complex double to ComplexFloat
    for (size_t i = 0; i < size; i++) {
        reg->amplitudes[i].real = (float)creal(amplitudes[i]);
        reg->amplitudes[i].imag = (float)cimag(amplitudes[i]);
    }

    return reg;
}

void quantum_register_destroy(quantum_register_t* reg) {
    if (reg) {
        free(reg->amplitudes);
        free(reg);
    }
}

qgt_error_t quantum_register_initialize(quantum_register_t* reg, const complex double* initial_state) {
    if (!reg || !initial_state) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < reg->size; i++) {
        reg->amplitudes[i].real = (float)creal(initial_state[i]);
        reg->amplitudes[i].imag = (float)cimag(initial_state[i]);
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_register_reset(quantum_register_t* reg) {
    if (!reg) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memset(reg->amplitudes, 0, reg->size * sizeof(ComplexFloat));
    reg->amplitudes[0].real = 1.0f;
    reg->amplitudes[0].imag = 0.0f;

    return QGT_SUCCESS;
}

qgt_error_t quantum_register_apply_gate(quantum_register_t* reg, const quantum_operator_t* gate, size_t target) {
    if (!reg || !gate || !gate->matrix) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify gate dimension is 2 (single-qubit gate)
    if (gate->dimension != 2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate the number of qubits from register size
    size_t num_qubits = 0;
    size_t temp = reg->size;
    while (temp > 1) {
        temp >>= 1;
        num_qubits++;
    }

    // Verify target qubit is valid
    if (target >= num_qubits) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Get gate matrix elements (2x2 stored row-major)
    // G = [[g00, g01], [g10, g11]]
    ComplexFloat g00 = gate->matrix[0];
    ComplexFloat g01 = gate->matrix[1];
    ComplexFloat g10 = gate->matrix[2];
    ComplexFloat g11 = gate->matrix[3];

    size_t stride = 1ULL << target;

    // Apply gate to each pair of amplitudes where target qubit differs
    // States with qubit=0 are paired with states with qubit=1
    for (size_t block = 0; block < reg->size; block += 2 * stride) {
        for (size_t i = block; i < block + stride; i++) {
            size_t j = i + stride;  // j has target qubit = 1, i has target qubit = 0

            ComplexFloat a = reg->amplitudes[i];  // |...0...⟩
            ComplexFloat b = reg->amplitudes[j];  // |...1...⟩

            // Apply gate: [a', b'] = G * [a, b]
            // a' = g00*a + g01*b
            // b' = g10*a + g11*b
            reg->amplitudes[i].real = g00.real * a.real - g00.imag * a.imag
                                    + g01.real * b.real - g01.imag * b.imag;
            reg->amplitudes[i].imag = g00.real * a.imag + g00.imag * a.real
                                    + g01.real * b.imag + g01.imag * b.real;

            reg->amplitudes[j].real = g10.real * a.real - g10.imag * a.imag
                                    + g11.real * b.real - g11.imag * b.imag;
            reg->amplitudes[j].imag = g10.real * a.imag + g10.imag * a.real
                                    + g11.real * b.imag + g11.imag * b.real;
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_register_apply_controlled_gate(quantum_register_t* reg, const quantum_operator_t* gate,
                                                    size_t control, size_t target) {
    if (!reg || !gate || !gate->matrix) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Verify gate dimension is 2 (single-qubit gate to be controlled)
    if (gate->dimension != 2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate the number of qubits from register size
    size_t num_qubits = 0;
    size_t temp = reg->size;
    while (temp > 1) {
        temp >>= 1;
        num_qubits++;
    }

    // Verify control and target qubits are valid and different
    if (control >= num_qubits || target >= num_qubits || control == target) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Get gate matrix elements
    ComplexFloat g00 = gate->matrix[0];
    ComplexFloat g01 = gate->matrix[1];
    ComplexFloat g10 = gate->matrix[2];
    ComplexFloat g11 = gate->matrix[3];

    size_t control_mask = 1ULL << control;
    size_t target_stride = 1ULL << target;

    // Apply gate only when control qubit is |1⟩
    for (size_t block = 0; block < reg->size; block += 2 * target_stride) {
        for (size_t i = block; i < block + target_stride; i++) {
            // Only apply if control qubit is 1
            if (!(i & control_mask)) {
                continue;  // Control qubit is 0, skip this pair
            }

            size_t j = i + target_stride;

            // Also check j has control = 1 (it will since we only changed target bit)
            // Actually we need to reconsider: i has target=0, j has target=1
            // But i might not have control=1. Let's fix the logic.

            // Skip if this state doesn't have control=1
            // We iterate over all states, and for each state with target=0,
            // we pair it with the state with target=1
            // We only apply gate if BOTH states have control=1
            if (!(j & control_mask)) {
                continue;
            }

            ComplexFloat a = reg->amplitudes[i];  // |...control=1...target=0...⟩
            ComplexFloat b = reg->amplitudes[j];  // |...control=1...target=1...⟩

            // Apply gate: [a', b'] = G * [a, b]
            reg->amplitudes[i].real = g00.real * a.real - g00.imag * a.imag
                                    + g01.real * b.real - g01.imag * b.imag;
            reg->amplitudes[i].imag = g00.real * a.imag + g00.imag * a.real
                                    + g01.real * b.imag + g01.imag * b.real;

            reg->amplitudes[j].real = g10.real * a.real - g10.imag * a.imag
                                    + g11.real * b.real - g11.imag * b.imag;
            reg->amplitudes[j].imag = g10.real * a.imag + g10.imag * a.real
                                    + g11.real * b.imag + g11.imag * b.real;
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_register_measure_qubit(quantum_register_t* reg, size_t qubit, int* result) {
    if (!reg || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate probability of measuring |1⟩ using stride optimization
    double prob_one = 0.0;
    size_t stride = 1ULL << qubit;

    // Use stride to efficiently iterate over states where qubit is |1⟩
    // States with qubit=1 are at positions: stride, stride+1, ..., 2*stride-1, 3*stride, ...
    for (size_t block = 0; block < reg->size; block += 2 * stride) {
        for (size_t i = block + stride; i < block + 2 * stride && i < reg->size; i++) {
            prob_one += reg->amplitudes[i].real * reg->amplitudes[i].real +
                       reg->amplitudes[i].imag * reg->amplitudes[i].imag;
        }
    }

    // Random measurement outcome
    double r = (double)rand() / RAND_MAX;
    *result = (r < prob_one) ? 1 : 0;

    // Collapse state
    double norm = 0.0;
    for (size_t i = 0; i < reg->size; i++) {
        if (((i >> qubit) & 1) == (size_t)*result) {
            norm += reg->amplitudes[i].real * reg->amplitudes[i].real +
                   reg->amplitudes[i].imag * reg->amplitudes[i].imag;
        } else {
            reg->amplitudes[i].real = 0.0f;
            reg->amplitudes[i].imag = 0.0f;
        }
    }

    // Normalize
    if (norm > 0) {
        float inv_norm = 1.0f / sqrtf((float)norm);
        for (size_t i = 0; i < reg->size; i++) {
            reg->amplitudes[i].real *= inv_norm;
            reg->amplitudes[i].imag *= inv_norm;
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_register_measure_all(quantum_register_t* reg, size_t* results) {
    if (!reg || !results) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Calculate probabilities
    double* probs = malloc(reg->size * sizeof(double));
    if (!probs) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    double total = 0.0;
    for (size_t i = 0; i < reg->size; i++) {
        probs[i] = reg->amplitudes[i].real * reg->amplitudes[i].real +
                  reg->amplitudes[i].imag * reg->amplitudes[i].imag;
        total += probs[i];
    }

    // Normalize and create cumulative distribution
    for (size_t i = 0; i < reg->size; i++) {
        probs[i] /= total;
    }

    // Sample from distribution
    double r = (double)rand() / RAND_MAX;
    double cumsum = 0.0;
    size_t outcome = 0;

    for (size_t i = 0; i < reg->size; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            outcome = i;
            break;
        }
    }

    *results = outcome;

    // Collapse to measured state
    memset(reg->amplitudes, 0, reg->size * sizeof(ComplexFloat));
    reg->amplitudes[outcome].real = 1.0f;
    reg->amplitudes[outcome].imag = 0.0f;

    free(probs);
    return QGT_SUCCESS;
}

qgt_error_t quantum_register_get_probabilities(quantum_register_t* reg, double* probabilities) {
    if (!reg || !probabilities) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < reg->size; i++) {
        probabilities[i] = reg->amplitudes[i].real * reg->amplitudes[i].real +
                          reg->amplitudes[i].imag * reg->amplitudes[i].imag;
    }

    return QGT_SUCCESS;
}

double quantum_register_fidelity(const quantum_register_t* reg1, const quantum_register_t* reg2) {
    if (!reg1 || !reg2 || reg1->size != reg2->size) {
        return 0.0;
    }

    // Fidelity = |⟨ψ|φ⟩|²
    double real_sum = 0.0, imag_sum = 0.0;

    for (size_t i = 0; i < reg1->size; i++) {
        // Inner product ⟨ψ|φ⟩ = Σ conj(ψ_i) * φ_i
        real_sum += reg1->amplitudes[i].real * reg2->amplitudes[i].real +
                   reg1->amplitudes[i].imag * reg2->amplitudes[i].imag;
        imag_sum += reg1->amplitudes[i].real * reg2->amplitudes[i].imag -
                   reg1->amplitudes[i].imag * reg2->amplitudes[i].real;
    }

    return real_sum * real_sum + imag_sum * imag_sum;
}

double quantum_register_trace_distance(const quantum_register_t* reg1, const quantum_register_t* reg2) {
    if (!reg1 || !reg2 || reg1->size != reg2->size) {
        return 1.0;
    }

    // For pure states: D = sqrt(1 - F)
    double fidelity = quantum_register_fidelity(reg1, reg2);
    return sqrt(1.0 - fidelity);
}

complex double quantum_register_expectation_value(const quantum_register_t* reg, const quantum_operator_t* op) {
    if (!reg || !op || !reg->amplitudes || !op->matrix) {
        return 0.0;
    }

    // Check dimension compatibility
    if (reg->size != op->dimension) {
        return 0.0;
    }

    // Compute expectation value: ⟨ψ|O|ψ⟩ = Σᵢⱼ ψᵢ* Oᵢⱼ ψⱼ
    // First compute O|ψ⟩, then compute ⟨ψ|O|ψ⟩
    complex double result = 0.0;
    size_t dim = reg->size;

    for (size_t i = 0; i < dim; i++) {
        // Compute (O|ψ⟩)ᵢ = Σⱼ Oᵢⱼ ψⱼ
        complex double op_psi_i = 0.0;
        for (size_t j = 0; j < dim; j++) {
            ComplexFloat o_ij = op->matrix[i * dim + j];
            ComplexFloat psi_j = reg->amplitudes[j];
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            op_psi_i += (o_ij.real * psi_j.real - o_ij.imag * psi_j.imag) +
                        I * (o_ij.real * psi_j.imag + o_ij.imag * psi_j.real);
        }

        // Add ψᵢ* · (O|ψ⟩)ᵢ to result
        ComplexFloat psi_i = reg->amplitudes[i];
        // ψᵢ* = (psi_i.real - i * psi_i.imag)
        result += (psi_i.real * creal(op_psi_i) + psi_i.imag * cimag(op_psi_i)) +
                  I * (psi_i.real * cimag(op_psi_i) - psi_i.imag * creal(op_psi_i));
    }

    return result;
}

qgt_error_t quantum_register_apply_error_correction(quantum_register_t* reg) {
    if (!reg) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Error correction implementation would go here
    return QGT_SUCCESS;
}

qgt_error_t quantum_register_syndrome_measurement(quantum_register_t* reg, double* syndrome) {
    if (!reg || !syndrome) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Syndrome measurement implementation would go here
    return QGT_SUCCESS;
}

qgt_error_t quantum_register_to_device(quantum_register_t* reg, int device_type) {
    if (!reg) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Device transfer implementation would go here
    return QGT_SUCCESS;
}

qgt_error_t quantum_register_from_device(quantum_register_t* reg) {
    if (!reg) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    // Device transfer implementation would go here
    return QGT_SUCCESS;
}

void quantum_register_print_state(const quantum_register_t* reg) {
    if (!reg) return;

    for (size_t i = 0; i < reg->size && i < 16; i++) {
        if (reg->amplitudes[i].real != 0.0f || reg->amplitudes[i].imag != 0.0f) {
            // Print would go here
        }
    }
}

int quantum_register_verify_state(const quantum_register_t* reg) {
    if (!reg) return 0;

    // Check normalization
    double norm = 0.0;
    for (size_t i = 0; i < reg->size; i++) {
        norm += reg->amplitudes[i].real * reg->amplitudes[i].real +
               reg->amplitudes[i].imag * reg->amplitudes[i].imag;
    }

    return (fabs(norm - 1.0) < 1e-6) ? 1 : 0;
}

size_t quantum_register_memory_size(const quantum_register_t* reg) {
    if (!reg) return 0;
    return sizeof(quantum_register_t) + reg->size * sizeof(ComplexFloat);
}
