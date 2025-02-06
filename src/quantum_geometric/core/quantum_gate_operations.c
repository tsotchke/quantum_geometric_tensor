#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/numeric_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to compute rotation matrices
static void compute_rotation_matrix(
    double angle,
    char axis,
    ComplexFloat* matrix) {
    
    double cos_half = cos(angle / 2.0);
    double sin_half = sin(angle / 2.0);
    
    switch (axis) {
        case 'x':
            // Rx = [cos(θ/2)   -i*sin(θ/2)]
            //      [-i*sin(θ/2)  cos(θ/2) ]
            matrix[0] = (ComplexFloat){cos_half, 0};
            matrix[1] = (ComplexFloat){0, -sin_half};
            matrix[2] = (ComplexFloat){0, -sin_half};
            matrix[3] = (ComplexFloat){cos_half, 0};
            break;
            
        case 'y':
            // Ry = [cos(θ/2)   -sin(θ/2)]
            //      [sin(θ/2)    cos(θ/2)]
            matrix[0] = (ComplexFloat){cos_half, 0};
            matrix[1] = (ComplexFloat){-sin_half, 0};
            matrix[2] = (ComplexFloat){sin_half, 0};
            matrix[3] = (ComplexFloat){cos_half, 0};
            break;
            
        case 'z':
            // Rz = [e^(-iθ/2)      0    ]
            //      [    0      e^(iθ/2) ]
            matrix[0] = (ComplexFloat){cos_half, -sin_half};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){cos_half, sin_half};
            break;
    }
}

// Helper function to compute fixed gate matrices
static void compute_fixed_matrix(
    gate_type_t type,
    ComplexFloat* matrix) {
    
    switch (type) {
        case GATE_TYPE_I:
            // I = [1 0]
            //     [0 1]
            matrix[0] = (ComplexFloat){1, 0};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){1, 0};
            break;
            
        case GATE_TYPE_X:
            // X = [0 1]
            //     [1 0]
            matrix[0] = (ComplexFloat){0, 0};
            matrix[1] = (ComplexFloat){1, 0};
            matrix[2] = (ComplexFloat){1, 0};
            matrix[3] = (ComplexFloat){0, 0};
            break;
            
        case GATE_TYPE_Y:
            // Y = [0 -i]
            //     [i  0]
            matrix[0] = (ComplexFloat){0, 0};
            matrix[1] = (ComplexFloat){0, -1};
            matrix[2] = (ComplexFloat){0, 1};
            matrix[3] = (ComplexFloat){0, 0};
            break;
            
        case GATE_TYPE_Z:
            // Z = [1  0]
            //     [0 -1]
            matrix[0] = (ComplexFloat){1, 0};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){-1, 0};
            break;
            
        case GATE_TYPE_H:
            // H = 1/√2 [1  1]
            //          [1 -1]
            {
                double inv_sqrt2 = 1.0 / sqrt(2.0);
                matrix[0] = (ComplexFloat){inv_sqrt2, 0};
                matrix[1] = (ComplexFloat){inv_sqrt2, 0};
                matrix[2] = (ComplexFloat){inv_sqrt2, 0};
                matrix[3] = (ComplexFloat){-inv_sqrt2, 0};
            }
            break;
            
        case GATE_TYPE_S:
            // S = [1 0]
            //     [0 i]
            matrix[0] = (ComplexFloat){1, 0};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){0, 1};
            break;
            
        case GATE_TYPE_T:
            // T = [1 0]
            //     [0 e^(iπ/4)]
            matrix[0] = (ComplexFloat){1, 0};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){cos(M_PI/4), sin(M_PI/4)};
            break;
            
        default:
            // Identity matrix as fallback
            matrix[0] = (ComplexFloat){1, 0};
            matrix[1] = (ComplexFloat){0, 0};
            matrix[2] = (ComplexFloat){0, 0};
            matrix[3] = (ComplexFloat){1, 0};
    }
}

// Helper function to compute controlled gate matrix
static void compute_controlled_matrix(
    const ComplexFloat* base_matrix,
    size_t num_qubits,
    ComplexFloat* controlled_matrix) {
    
    size_t dim = 1 << num_qubits;
    
    // Create control projector |1⟩⟨1|
    ComplexFloat* control_proj = malloc(4 * sizeof(ComplexFloat));
    control_proj[0] = (ComplexFloat){0, 0};
    control_proj[1] = (ComplexFloat){0, 0};
    control_proj[2] = (ComplexFloat){0, 0};
    control_proj[3] = (ComplexFloat){1, 0};
    
    // Create identity matrix
    ComplexFloat* identity = malloc(4 * sizeof(ComplexFloat));
    identity[0] = (ComplexFloat){1, 0};
    identity[1] = (ComplexFloat){0, 0};
    identity[2] = (ComplexFloat){0, 0};
    identity[3] = (ComplexFloat){1, 0};
    
    // Compute |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
    ComplexFloat* temp = malloc(dim * dim * sizeof(ComplexFloat));
    
    // First term: (I - |1⟩⟨1|) ⊗ I
    for (size_t i = 0; i < dim * dim; i++) {
        controlled_matrix[i] = (ComplexFloat){0, 0};
    }
    for (size_t i = 0; i < dim/2; i++) {
        controlled_matrix[i * dim + i] = (ComplexFloat){1, 0};
    }
    
    // Second term: |1⟩⟨1| ⊗ U
    matrix_multiply(control_proj, base_matrix, temp, 2, 2, 2);
    
    // Add the terms
    for (size_t i = dim/2; i < dim; i++) {
        for (size_t j = dim/2; j < dim; j++) {
            controlled_matrix[i * dim + j] = base_matrix[(i-dim/2) * 2 + (j-dim/2)];
        }
    }
    
    free(control_proj);
    free(identity);
    free(temp);
}

// Create a deep copy of a quantum gate
quantum_gate_t* copy_quantum_gate(const quantum_gate_t* gate) {
    if (!gate) return NULL;
    
    quantum_gate_t* new_gate = malloc(sizeof(quantum_gate_t));
    if (!new_gate) return NULL;
    
    // Copy basic fields
    new_gate->num_qubits = gate->num_qubits;
    new_gate->num_controls = gate->num_controls;
    new_gate->is_controlled = gate->is_controlled;
    new_gate->type = gate->type;
    new_gate->num_parameters = gate->num_parameters;
    new_gate->is_parameterized = gate->is_parameterized;
    
    // Allocate and copy arrays
    size_t matrix_size = 1 << (2 * gate->num_qubits);  // 2^n x 2^n matrix
    new_gate->matrix = malloc(matrix_size * sizeof(ComplexFloat));
    new_gate->target_qubits = malloc(gate->num_qubits * sizeof(size_t));
    new_gate->control_qubits = gate->num_controls ? malloc(gate->num_controls * sizeof(size_t)) : NULL;
    new_gate->parameters = gate->num_parameters ? malloc(gate->num_parameters * sizeof(double)) : NULL;
    
    if (!new_gate->matrix || !new_gate->target_qubits || 
        (gate->num_controls && !new_gate->control_qubits) ||
        (gate->num_parameters && !new_gate->parameters)) {
        free(new_gate->matrix);
        free(new_gate->target_qubits);
        free(new_gate->control_qubits);
        free(new_gate->parameters);
        free(new_gate);
        return NULL;
    }
    
    // Copy data
    memcpy(new_gate->matrix, gate->matrix, matrix_size * sizeof(ComplexFloat));
    memcpy(new_gate->target_qubits, gate->target_qubits, gate->num_qubits * sizeof(size_t));
    if (gate->num_controls) {
        memcpy(new_gate->control_qubits, gate->control_qubits, gate->num_controls * sizeof(size_t));
    }
    if (gate->num_parameters) {
        memcpy(new_gate->parameters, gate->parameters, gate->num_parameters * sizeof(double));
    }
    
    return new_gate;
}

// Create a new quantum gate
quantum_gate_t* create_quantum_gate(
    gate_type_t type,
    const size_t* qubits,
    size_t num_qubits,
    const double* parameters,
    size_t num_parameters) {
    
    if (!qubits || num_qubits == 0 || 
        (parameters == NULL && num_parameters > 0)) {
        return NULL;
    }
    
    quantum_gate_t* gate = malloc(sizeof(quantum_gate_t));
    if (!gate) return NULL;
    
    gate->type = type;
    gate->num_qubits = num_qubits;
    gate->target_qubits = malloc(num_qubits * sizeof(size_t));
    if (!gate->target_qubits) {
        free(gate);
        return NULL;
    }
    memcpy(gate->target_qubits, qubits, num_qubits * sizeof(size_t));
    gate->num_controls = 0;
    gate->control_qubits = NULL;
    gate->is_controlled = (type == GATE_TYPE_CNOT || type == GATE_TYPE_CZ);
    gate->is_parameterized = false;
    gate->parameters = NULL;
    gate->num_parameters = 0;
    
    // Allocate matrix
    size_t matrix_dim = 1 << num_qubits;
    gate->matrix = malloc(matrix_dim * matrix_dim * sizeof(ComplexFloat));
    if (!gate->matrix) {
        free(gate->target_qubits);
        free(gate);
        return NULL;
    }
    
    // Initialize matrix based on gate type
    ComplexFloat base_matrix[4];  // 2x2 matrix for single qubit gates
    
    switch (type) {
        case GATE_TYPE_RX:
        case GATE_TYPE_RY:
        case GATE_TYPE_RZ:
            if (num_parameters != 1) {
                free(gate->matrix);
                free(gate->target_qubits);
                free(gate);
                return NULL;
            }
            gate->is_parameterized = true;
            gate->parameters = malloc(sizeof(double));
            if (!gate->parameters) {
                free(gate->matrix);
                free(gate->target_qubits);
                free(gate);
                return NULL;
            }
            gate->parameters[0] = parameters[0];
            gate->num_parameters = 1;
            compute_rotation_matrix(parameters[0], 
                                 type == GATE_TYPE_RX ? 'x' :
                                 type == GATE_TYPE_RY ? 'y' : 'z',
                                 base_matrix);
            break;
            
        case GATE_TYPE_CNOT:
        case GATE_TYPE_CZ:
        case GATE_TYPE_SWAP:
            if (num_qubits != 2) {
                free(gate->matrix);
                free(gate->target_qubits);
                free(gate);
                return NULL;
            }
            compute_fixed_matrix(type == GATE_TYPE_CNOT ? GATE_TYPE_X :
                               type == GATE_TYPE_CZ ? GATE_TYPE_Z :
                               GATE_TYPE_I, base_matrix);
            break;
            
        default:
            compute_fixed_matrix(type, base_matrix);
    }
    
    // Convert to full matrix if controlled
    if (type == GATE_TYPE_CNOT || type == GATE_TYPE_CZ) {
        compute_controlled_matrix(base_matrix, num_qubits, gate->matrix);
    } else {
        memcpy(gate->matrix, base_matrix, 4 * sizeof(ComplexFloat));
    }
    
    return gate;
}

// Update gate parameters
bool update_gate_parameters(
    quantum_gate_t* gate,
    const double* parameters,
    size_t num_parameters) {
    
    if (!gate || !gate->is_parameterized || !parameters || num_parameters != gate->num_parameters ||
        (gate->type != GATE_TYPE_RX && gate->type != GATE_TYPE_RY && gate->type != GATE_TYPE_RZ)) {
        return false;
    }
    
    // Recompute gate matrix with new parameters
    ComplexFloat base_matrix[4];
    
    // Update parameters
    memcpy(gate->parameters, parameters, num_parameters * sizeof(double));

    // Recompute matrix based on gate type
    switch (gate->type) {
        case GATE_TYPE_RX:
            compute_rotation_matrix(parameters[0], 'x', base_matrix);
            break;
        case GATE_TYPE_RY:
            compute_rotation_matrix(parameters[0], 'y', base_matrix);
            break;
        case GATE_TYPE_RZ:
            compute_rotation_matrix(parameters[0], 'z', base_matrix);
            break;
        default:
            return false;
    }
    
    if (gate->is_controlled) {
        compute_controlled_matrix(base_matrix, gate->num_qubits, gate->matrix);
    } else {
        memcpy(gate->matrix, base_matrix, 4 * sizeof(ComplexFloat));
    }
    
    return true;
}

// Shift gate parameters
bool shift_gate_parameters(
    quantum_gate_t* gate,
    size_t param_idx,
    double shift_amount) {
    
    if (!gate || !gate->is_parameterized || param_idx >= gate->num_parameters) {
        return false;
    }
    
    // For now, assume single parameter rotation gates
    double shifted_param = gate->parameters[param_idx] + shift_amount;
    return update_gate_parameters(gate, &shifted_param, 1);
}

// Destroy quantum gate
void destroy_quantum_gate(quantum_gate_t* gate) {
    if (!gate) return;
    
    free(gate->matrix);
    free(gate->target_qubits);
    free(gate->control_qubits);
    free(gate->parameters);
    free(gate);
}
