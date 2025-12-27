/**
 * @file quantum_field_operations.h
 * @brief High-performance quantum field operations
 *
 * Provides optimized field evolution and coupling computations with:
 * - Hierarchical matrix representation (O(log n) complexity)
 * - GPU acceleration when available (Metal/CUDA)
 * - Distributed computing support
 * - Cache-optimized CPU fallbacks
 */

#ifndef QUANTUM_FIELD_OPERATIONS_H
#define QUANTUM_FIELD_OPERATIONS_H

#include <stddef.h>
#include <stdbool.h>
#include <complex.h>
#include "quantum_geometric/core/quantum_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#ifndef QG_GPU_BLOCK_SIZE
#define QG_GPU_BLOCK_SIZE 256
#endif

#ifndef QG_FIELD_CACHE_SIZE
#define QG_FIELD_CACHE_SIZE 4096
#endif

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Gauge transformation specification
 */
typedef struct GaugeTransform {
    double complex* matrix;          /**< Transformation matrix elements */
    size_t dimension;                /**< Dimension of gauge group */
    double phase;                    /**< Global phase factor */
    bool is_abelian;                 /**< Whether transformation is abelian */
} GaugeTransform;

/**
 * @brief Fast approximation state for field computations
 */
typedef struct FastApproximation {
    double complex* coefficients;    /**< Approximation coefficients */
    size_t num_terms;                /**< Number of terms in expansion */
    double tolerance;                /**< Approximation tolerance */
    void* internal_state;            /**< Implementation-specific state */
} FastApproximation;

/**
 * @brief Forward declaration for hierarchical matrix
 */
struct HierarchicalMatrix;

// ============================================================================
// Core Field Operations
// ============================================================================

/**
 * @brief Evolve quantum field under Hamiltonian
 *
 * Uses hierarchical matrix methods for O(log n) complexity.
 *
 * @param field Field values to evolve (modified in place)
 * @param hamiltonian Hamiltonian operator
 * @param n Field dimension
 */
void evolve_quantum_field(double complex* field,
                         const double complex* hamiltonian,
                         size_t n);

/**
 * @brief Compute coupling between two fields
 *
 * Uses GPU acceleration when available.
 *
 * @param coupling Output coupling values
 * @param field1 First field
 * @param field2 Second field
 * @param n Field dimension
 */
void compute_field_coupling(double complex* coupling,
                          const double complex* field1,
                          const double complex* field2,
                          size_t n);

/**
 * @brief Compute field equations of motion
 *
 * Uses distributed computing for large systems.
 *
 * @param equations Output equation values
 * @param field Input field configuration
 * @param n Field dimension
 */
void compute_field_equations(double complex* equations,
                           const double complex* field,
                           size_t n);

// ============================================================================
// Gauge Operations
// ============================================================================

/**
 * @brief Apply gauge transformation to field
 *
 * @param field Field to transform (modified in place)
 * @param transform Gauge transformation specification
 * @param n Field dimension
 */
void apply_gauge_transformation(double complex* field,
                              const GaugeTransform* transform,
                              size_t n);

/**
 * @brief Create gauge transformation
 *
 * @param dimension Gauge group dimension
 * @param is_abelian Whether group is abelian
 * @return New gauge transformation (caller must free)
 */
GaugeTransform* create_gauge_transform(size_t dimension, bool is_abelian);

/**
 * @brief Destroy gauge transformation
 *
 * @param transform Transformation to destroy
 */
void destroy_gauge_transform(GaugeTransform* transform);

// ============================================================================
// Fast Approximation Methods
// ============================================================================

/**
 * @brief Initialize fast approximation for field
 *
 * @param field Field to approximate
 * @param n Field dimension
 * @return Approximation state (caller must free)
 */
FastApproximation* init_fast_approximation(const double complex* field, size_t n);

/**
 * @brief Compute equations using approximation
 *
 * @param approx Approximation state
 * @param equations Output equations
 */
void compute_approximated_equations(FastApproximation* approx, double complex* equations);

/**
 * @brief Destroy fast approximation
 *
 * @param approx Approximation to destroy
 */
void destroy_fast_approximation(FastApproximation* approx);

// ============================================================================
// Hierarchical Operations
// ============================================================================

/**
 * @brief Convert field to hierarchical representation
 *
 * @param field Field data
 * @param n Field dimension
 * @return Hierarchical matrix (caller must free)
 */
struct HierarchicalMatrix* convert_to_hierarchical(const double complex* field, size_t n);

/**
 * @brief Convert from hierarchical representation
 *
 * @param field Output field data
 * @param matrix Hierarchical matrix
 */
void convert_from_hierarchical(double complex* field, const struct HierarchicalMatrix* matrix);

/**
 * @brief Destroy hierarchical matrix
 *
 * @param matrix Matrix to destroy
 */
void destroy_hierarchical_matrix(struct HierarchicalMatrix* matrix);

// ============================================================================
// Cleanup
// ============================================================================

/**
 * @brief Cleanup field operation caches
 */
void cleanup_field_cache(void);

/**
 * @brief Cleanup field operation buffers
 */
void cleanup_field_buffers(void);

/**
 * @brief Complete cleanup of field operations resources
 */
void cleanup_quantum_field_operations(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FIELD_OPERATIONS_H
