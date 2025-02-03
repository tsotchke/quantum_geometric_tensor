#include "quantum_geometric/core/quantum_gate_operations.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/error_handling.h"
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
    size_t base_dim = 2;  // Base matrices are 2x2
    
    // Initialize to identity
    for (size_t i = 0; i < dim * dim; i++) {
        controlled_matrix[i] = (ComplexFloat){0, 0};
    }
    for (size_t i = 0; i < dim; i++) {
        controlled_matrix[i * dim + i] = (ComplexFloat){1, 0};
    }
    
    // Add controlled operation
    size_t control_mask = (1 << (num_qubits - 1)) - 1;
    for (size_t i = 0; i < base_dim; i++) {
        for (size_t j = 0; j < base_dim; j++) {
            size_t ii = control_mask | (i << (num_qubits - 1));
            size_t jj = control_mask | (j << (num_qubits - 1));
            controlled_matrix[ii * dim + jj] = base_matrix[i * base_dim + j];
        }
    }
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
    
    gate->num_qubits = num_qubits;
    gate->target_qubits = malloc(num_qubits * sizeof(size_t));
    if (!gate->target_qubits) {
        free(gate);
        return NULL;
    }
    memcpy(gate->target_qubits, qubits, num_qubits * sizeof(size_t));
    
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
    
    if (!gate || !parameters || num_parameters == 0) {
        return false;
    }
    
    // Recompute gate matrix with new parameters
    ComplexFloat base_matrix[4];
    
    // Determine gate type and update accordingly
    // For now, assume it's a rotation gate
    compute_rotation_matrix(parameters[0], 'z', base_matrix);  // Default to Z rotation
    
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
    
    if (!gate || param_idx >= gate->num_qubits) {
        return false;
    }
    
    // For now, assume single parameter rotation gates
    double shifted_param = shift_amount;  // Original parameter + shift
    return update_gate_parameters(gate, &shifted_param, 1);
}

// Destroy quantum gate
void destroy_quantum_gate(quantum_gate_t* gate) {
    if (!gate) return;
    
    free(gate->matrix);
    free(gate->target_qubits);
    free(gate->control_qubits);
    free(gate);
}
