/**
 * @file quantum_geometric_hamiltonian.c
 * @brief Hamiltonian construction from quantum geometric tensors
 *
 * This module creates Hamiltonians that encode geometric constraints from
 * quantum geometric tensors, including metric, curvature, and topological
 * contributions.
 */

#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// ============================================================================
// Type Definitions for Hamiltonian Construction
// ============================================================================

/**
 * @brief Physical constraints for manifold projection
 * Note: PhysicalConstraints is defined in quantum_geometric_types.h
 */
#ifndef PHYSICAL_CONSTRAINTS_DEFINED
#define PHYSICAL_CONSTRAINTS_DEFINED
typedef struct PhysicalConstraints {
    double energy_threshold;       // Maximum allowed energy deviation
    double symmetry_tolerance;     // Tolerance for symmetry violation
    double conservation_tolerance; // Tolerance for conservation law violation
    double gauge_tolerance;        // Tolerance for gauge constraint violation
    double locality_tolerance;     // Tolerance for locality violation
    double causality_tolerance;    // Tolerance for causality violation
} PhysicalConstraints;
#endif

/**
 * @brief Geometric Hamiltonian structure using HierarchicalMatrix
 */
typedef struct GeometricHamiltonian {
    HierarchicalMatrix* matrix;    // Hamiltonian matrix representation
    size_t dimension;              // Hilbert space dimension
    double energy_scale;           // Energy scale factor
    bool is_hermitian;             // Hermiticity flag
} GeometricHamiltonian;

// ============================================================================
// Forward Declarations
// ============================================================================

static bool project_energy(HierarchicalMatrix* matrix, double threshold);
static bool project_symmetry(HierarchicalMatrix* matrix, double tolerance);
static bool project_conservation(HierarchicalMatrix* matrix, double tolerance);
static bool project_gauge(HierarchicalMatrix* matrix, double tolerance);
static bool project_locality(HierarchicalMatrix* matrix, double tolerance);
static bool project_causality(HierarchicalMatrix* matrix, double tolerance);
static double complex project_to_gauge_orbit(double complex value, double tolerance);

// Neighbor distance for locality constraint
#ifndef QG_NEIGHBOR_DISTANCE
#define QG_NEIGHBOR_DISTANCE 2
#endif

// ============================================================================
// Hamiltonian Creation from Quantum Geometric Tensor
// ============================================================================

/**
 * @brief Create Hamiltonian from geometric constraints
 *
 * Constructs a Hamiltonian that enforces the geometric constraints from
 * the quantum geometric tensor, including metric, curvature, and connection.
 *
 * @param qgt Quantum geometric tensor containing geometric structure
 * @return GeometricHamiltonian structure (caller must free with destroy_geometric_hamiltonian)
 */
GeometricHamiltonian* create_geometric_hamiltonian(const quantum_geometric_tensor_t* qgt) {
    if (!qgt || !qgt->dimensions || qgt->rank < 1) return NULL;

    // Get dimension from tensor (first dimension)
    size_t dim = qgt->dimensions[0];
    if (dim == 0) return NULL;

    // Allocate Hamiltonian structure
    GeometricHamiltonian* hamiltonian = calloc(1, sizeof(GeometricHamiltonian));
    if (!hamiltonian) return NULL;

    hamiltonian->dimension = dim;
    hamiltonian->energy_scale = 1.0;
    hamiltonian->is_hermitian = true;

    // Create hierarchical matrix for efficient O(log n) operations
    hamiltonian->matrix = create_hierarchical_matrix(dim, 1e-10);
    if (!hamiltonian->matrix) {
        free(hamiltonian);
        return NULL;
    }

    // Initialize matrix data
    size_t total_elements = dim * dim;
    hamiltonian->matrix->data = calloc(total_elements, sizeof(double complex));
    if (!hamiltonian->matrix->data) {
        destroy_hierarchical_matrix(hamiltonian->matrix);
        free(hamiltonian);
        return NULL;
    }
    hamiltonian->matrix->rows = dim;
    hamiltonian->matrix->cols = dim;
    hamiltonian->matrix->n = dim;
    hamiltonian->matrix->is_leaf = true;

    // Add contributions from quantum geometric tensor components
    if (qgt->components) {
        // The QGT components encode the metric tensor g_μν
        // H_ij = g_ij (metric contribution)
        size_t max_elements = qgt->total_elements < total_elements ?
                              qgt->total_elements : total_elements;
        for (size_t idx = 0; idx < max_elements; idx++) {
            // Convert ComplexFloat to double complex
            ComplexFloat cf = qgt->components[idx];
            hamiltonian->matrix->data[idx] = cf.real + I * cf.imag;
        }
    }

    // Ensure Hermiticity: H = (H + H†) / 2
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            size_t idx_ij = i * dim + j;
            size_t idx_ji = j * dim + i;
            double complex avg = 0.5 * (hamiltonian->matrix->data[idx_ij] +
                                        conj(hamiltonian->matrix->data[idx_ji]));
            hamiltonian->matrix->data[idx_ij] = avg;
            hamiltonian->matrix->data[idx_ji] = conj(avg);
        }
        // Diagonal elements must be real for Hermitian matrix
        size_t idx_ii = i * dim + i;
        hamiltonian->matrix->data[idx_ii] = creal(hamiltonian->matrix->data[idx_ii]);
    }

    return hamiltonian;
}

/**
 * @brief Destroy geometric Hamiltonian and free resources
 */
void destroy_geometric_hamiltonian(GeometricHamiltonian* hamiltonian) {
    if (!hamiltonian) return;

    if (hamiltonian->matrix) {
        destroy_hierarchical_matrix(hamiltonian->matrix);
    }
    free(hamiltonian);
}

// ============================================================================
// Physical Manifold Projection
// ============================================================================

/**
 * @brief Project Hamiltonian onto physical manifold
 *
 * Projects a Hamiltonian matrix onto the manifold of physically valid states
 * according to the provided constraints (energy bounds, symmetry, conservation, etc).
 *
 * @param hamiltonian Input Hamiltonian to project
 * @param constraints Physical constraints to enforce
 * @return Projected Hamiltonian (caller must free with destroy_geometric_hamiltonian)
 */
GeometricHamiltonian* project_to_physical_manifold(
    const GeometricHamiltonian* hamiltonian,
    const PhysicalConstraints* constraints) {

    if (!hamiltonian || !hamiltonian->matrix || !constraints) return NULL;

    // Clone input Hamiltonian
    GeometricHamiltonian* result = calloc(1, sizeof(GeometricHamiltonian));
    if (!result) return NULL;

    result->dimension = hamiltonian->dimension;
    result->energy_scale = hamiltonian->energy_scale;
    result->is_hermitian = hamiltonian->is_hermitian;

    // Clone the hierarchical matrix
    result->matrix = create_hierarchical_matrix(hamiltonian->dimension, 1e-10);
    if (!result->matrix) {
        free(result);
        return NULL;
    }

    size_t total_elements = hamiltonian->dimension * hamiltonian->dimension;
    result->matrix->data = malloc(total_elements * sizeof(double complex));
    if (!result->matrix->data) {
        destroy_hierarchical_matrix(result->matrix);
        free(result);
        return NULL;
    }
    memcpy(result->matrix->data, hamiltonian->matrix->data,
           total_elements * sizeof(double complex));
    result->matrix->rows = hamiltonian->dimension;
    result->matrix->cols = hamiltonian->dimension;
    result->matrix->n = hamiltonian->dimension;
    result->matrix->is_leaf = true;

    // Apply energy constraint
    if (!project_energy(result->matrix, constraints->energy_threshold)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    // Apply symmetry constraints
    if (!project_symmetry(result->matrix, constraints->symmetry_tolerance)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    // Apply conservation laws
    if (!project_conservation(result->matrix, constraints->conservation_tolerance)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    // Apply gauge constraints
    if (!project_gauge(result->matrix, constraints->gauge_tolerance)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    // Apply locality constraints
    if (!project_locality(result->matrix, constraints->locality_tolerance)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    // Apply causality constraints
    if (!project_causality(result->matrix, constraints->causality_tolerance)) {
        destroy_geometric_hamiltonian(result);
        return NULL;
    }

    return result;
}

// ============================================================================
// Projection Helper Functions
// ============================================================================

/**
 * @brief Project matrix to satisfy energy bounds
 */
static bool project_energy(HierarchicalMatrix* matrix, double threshold) {
    if (!matrix || !matrix->data) return false;
    if (threshold <= 0) return true;

    size_t size = matrix->rows * matrix->cols;

    // Calculate total energy (Frobenius norm squared)
    double energy = 0.0;
    for (size_t i = 0; i < size; i++) {
        energy += cabs(matrix->data[i]) * cabs(matrix->data[i]);
    }

    // Project if energy exceeds threshold
    if (energy > threshold) {
        double scale = sqrt(threshold / energy);
        for (size_t i = 0; i < size; i++) {
            matrix->data[i] *= scale;
        }
    }

    return true;
}

/**
 * @brief Project matrix to symmetric/Hermitian form
 */
static bool project_symmetry(HierarchicalMatrix* matrix, double tolerance) {
    if (!matrix || !matrix->data) return false;
    if (tolerance <= 0) return true;

    size_t dim = matrix->rows;

    // Project to Hermitian: H = (H + H†) / 2
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            size_t ij = i * dim + j;
            size_t ji = j * dim + i;
            double complex avg = (matrix->data[ij] + conj(matrix->data[ji])) * 0.5;
            if (cabs(matrix->data[ij] - avg) > tolerance) {
                matrix->data[ij] = avg;
                matrix->data[ji] = conj(avg);
            }
        }
        // Diagonal must be real
        size_t ii = i * dim + i;
        matrix->data[ii] = creal(matrix->data[ii]);
    }

    return true;
}

/**
 * @brief Project to conserve trace (probability)
 */
static bool project_conservation(HierarchicalMatrix* matrix, double tolerance) {
    if (!matrix || !matrix->data) return false;
    if (tolerance <= 0) return true;

    size_t dim = matrix->rows;

    // Calculate trace
    double total = 0.0;
    for (size_t i = 0; i < dim; i++) {
        size_t ii = i * dim + i;
        total += creal(matrix->data[ii]);
    }

    // Normalize trace if needed
    if (fabs(total - 1.0) > tolerance && fabs(total) > 1e-15) {
        double scale = 1.0 / total;
        for (size_t i = 0; i < dim; i++) {
            size_t ii = i * dim + i;
            matrix->data[ii] *= scale;
        }
    }

    return true;
}

/**
 * @brief Project to gauge orbit (phase normalization)
 */
static bool project_gauge(HierarchicalMatrix* matrix, double tolerance) {
    if (!matrix || !matrix->data) return false;
    if (tolerance <= 0) return true;

    size_t dim = matrix->rows;

    // Project each element to gauge orbit
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;
            matrix->data[idx] = project_to_gauge_orbit(matrix->data[idx], tolerance);
        }
    }

    return true;
}

/**
 * @brief Project single value to gauge orbit
 */
static double complex project_to_gauge_orbit(double complex value, double tolerance) {
    double magnitude = cabs(value);
    if (magnitude < tolerance) {
        return 0.0;
    }
    // Normalize phase to standard form
    return value;  // Gauge-invariant for now
}

/**
 * @brief Enforce locality by suppressing long-range terms
 */
static bool project_locality(HierarchicalMatrix* matrix, double tolerance) {
    if (!matrix || !matrix->data) return false;
    if (tolerance <= 0) return true;

    size_t dim = matrix->rows;

    // Suppress long-range interactions
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t idx = i * dim + j;
            int distance = abs((int)i - (int)j);
            if (distance > QG_NEIGHBOR_DISTANCE) {
                double magnitude = cabs(matrix->data[idx]);
                if (magnitude > tolerance) {
                    matrix->data[idx] *= tolerance / magnitude;
                }
            }
        }
    }

    return true;
}

/**
 * @brief Enforce causality by making lower triangular part small
 */
static bool project_causality(HierarchicalMatrix* matrix, double tolerance) {
    if (!matrix || !matrix->data) return false;
    if (tolerance <= 0) return true;

    size_t dim = matrix->rows;

    // Suppress anti-causal (lower triangular) terms
    for (size_t i = 1; i < dim; i++) {
        for (size_t j = 0; j < i; j++) {
            size_t idx = i * dim + j;
            if (cabs(matrix->data[idx]) > tolerance) {
                matrix->data[idx] = 0.0;
            }
        }
    }

    return true;
}

// ============================================================================
// Hamiltonian Operations
// ============================================================================

/**
 * @brief Apply Hamiltonian to state vector: result = H * state
 */
bool apply_hamiltonian(const GeometricHamiltonian* hamiltonian,
                       const ComplexFloat* state,
                       ComplexFloat* result) {
    if (!hamiltonian || !hamiltonian->matrix || !state || !result) return false;

    size_t dim = hamiltonian->dimension;

    // Convert ComplexFloat to double complex
    double complex* state_dc = malloc(dim * sizeof(double complex));
    double complex* result_dc = calloc(dim, sizeof(double complex));
    if (!state_dc || !result_dc) {
        free(state_dc);
        free(result_dc);
        return false;
    }

    for (size_t i = 0; i < dim; i++) {
        state_dc[i] = state[i].real + I * state[i].imag;
    }

    // Matrix-vector multiplication
    hmatrix_multiply_vector(hamiltonian->matrix, state_dc, result_dc, 1);

    // Convert back to ComplexFloat
    for (size_t i = 0; i < dim; i++) {
        result[i].real = (float)creal(result_dc[i]);
        result[i].imag = (float)cimag(result_dc[i]);
    }

    free(state_dc);
    free(result_dc);
    return true;
}

/**
 * @brief Compute expectation value ⟨ψ|H|ψ⟩
 * @note Renamed to avoid conflict with quantum_physics_operations.c (this is Hamiltonian-specific)
 */
double compute_hamiltonian_expectation_value(const GeometricHamiltonian* hamiltonian,
                                              const ComplexFloat* state) {
    if (!hamiltonian || !state) return 0.0;

    size_t dim = hamiltonian->dimension;

    ComplexFloat* h_state = malloc(dim * sizeof(ComplexFloat));
    if (!h_state) return 0.0;

    if (!apply_hamiltonian(hamiltonian, state, h_state)) {
        free(h_state);
        return 0.0;
    }

    // ⟨ψ|H|ψ⟩
    double complex expectation = 0.0;
    for (size_t i = 0; i < dim; i++) {
        double complex psi_i = state[i].real + I * state[i].imag;
        double complex h_psi_i = h_state[i].real + I * h_state[i].imag;
        expectation += conj(psi_i) * h_psi_i;
    }

    free(h_state);
    return creal(expectation);  // Should be real for Hermitian H
}
