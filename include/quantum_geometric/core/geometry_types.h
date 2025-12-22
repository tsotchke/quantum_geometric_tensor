#ifndef GEOMETRY_TYPES_H
#define GEOMETRY_TYPES_H

/**
 * @file geometry_types.h
 * @brief Shared geometry type definitions used across the library
 *
 * This header provides unified geometry type definitions to avoid
 * conflicts between different modules.
 */

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Core Geometry Types (Shared)
// =============================================================================

/**
 * @brief Core geometry classification
 */
typedef enum {
    GEOMETRY_TYPE_EUCLIDEAN = 0,     // Flat Euclidean geometry
    GEOMETRY_TYPE_RIEMANNIAN,        // Riemannian manifold geometry
    GEOMETRY_TYPE_SYMPLECTIC,        // Symplectic geometry (phase space)
    GEOMETRY_TYPE_KAHLER,            // Kahler manifold (complex)
    GEOMETRY_TYPE_FUBINI_STUDY,      // Fubini-Study metric (quantum)
    GEOMETRY_TYPE_PROJECTIVE,        // Complex projective space
    GEOMETRY_TYPE_HYPERBOLIC,        // Hyperbolic geometry
    GEOMETRY_TYPE_COUNT
} core_geometry_type_t;

/**
 * @brief Geometry metric types
 */
typedef enum {
    METRIC_FLAT = 0,                 // Flat metric
    METRIC_CURVED,                   // General curved metric
    METRIC_DIAGONAL,                 // Diagonal metric tensor
    METRIC_BLOCK_DIAGONAL,           // Block diagonal metric
    METRIC_HERMITIAN,                // Hermitian metric (complex)
    METRIC_FISHER,                   // Fisher information metric
    METRIC_BURES,                    // Bures metric (quantum states)
    METRIC_COUNT
} geometry_metric_t;

/**
 * @brief Connection types for parallel transport
 */
typedef enum {
    CONNECTION_LEVI_CIVITA = 0,      // Levi-Civita (metric compatible, torsion-free)
    CONNECTION_CHERN,                // Chern connection (complex manifolds)
    CONNECTION_BERRY,                // Berry connection (quantum systems)
    CONNECTION_QUANTUM_GEOMETRIC,    // Full quantum geometric connection
    CONNECTION_AFFINE,               // General affine connection
    CONNECTION_COUNT
} geometry_connection_t;

/**
 * @brief Curvature types
 */
typedef enum {
    CURVATURE_RIEMANN = 0,           // Riemann curvature tensor
    CURVATURE_RICCI,                 // Ricci curvature
    CURVATURE_SCALAR,                // Scalar curvature
    CURVATURE_WEYL,                  // Weyl curvature (conformal)
    CURVATURE_BERRY,                 // Berry curvature (quantum)
    CURVATURE_SECTIONAL,             // Sectional curvature
    CURVATURE_COUNT
} curvature_type_t;

/**
 * @brief Phase types for quantum geometric phases
 */
typedef enum {
    PHASE_TYPE_BERRY = 0,            // Berry phase (geometric)
    PHASE_TYPE_AHARONOV_BOHM,        // Aharonov-Bohm phase
    PHASE_TYPE_PANCHARATNAM,         // Pancharatnam phase
    PHASE_TYPE_DYNAMIC,              // Dynamic phase
    PHASE_TYPE_TOPOLOGICAL,          // Topological phase
    PHASE_TYPE_COUNT
} geometry_phase_type_t;

/**
 * @brief Basic manifold info structure
 */
typedef struct {
    core_geometry_type_t type;       // Type of geometry
    size_t dimension;                // Manifold dimension
    size_t embedding_dim;            // Embedding space dimension
    geometry_metric_t metric_type;   // Metric type
    bool is_compact;                 // Compact manifold flag
    bool is_orientable;              // Orientable manifold flag
    double characteristic;           // Euler characteristic or similar invariant
} manifold_info_t;

/**
 * @brief Metric tensor structure
 */
typedef struct {
    double* components;              // Metric tensor components (row-major)
    size_t dimension;                // Dimension (n x n tensor)
    geometry_metric_t type;          // Metric type
    bool is_positive_definite;       // Positive definiteness
    double determinant;              // Metric determinant
    double* inverse;                 // Inverse metric (optional, can be NULL)
} metric_tensor_t;

/**
 * @brief Connection coefficients (Christoffel symbols)
 */
typedef struct {
    double* components;              // Gamma^i_jk components
    size_t dimension;                // Manifold dimension
    geometry_connection_t type;      // Connection type
    bool is_metric_compatible;       // Metric compatibility
    bool is_torsion_free;            // Torsion-free property
} connection_coeffs_t;

/**
 * @brief Curvature tensor structure
 */
typedef struct {
    double* components;              // R^i_jkl or contracted components
    size_t dimension;                // Manifold dimension
    curvature_type_t type;           // Which curvature
    double scalar_value;             // For scalar curvature
} curvature_tensor_t;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get name of geometry type
 */
static inline const char* geometry_type_name(core_geometry_type_t type) {
    static const char* names[] = {
        "Euclidean",
        "Riemannian",
        "Symplectic",
        "Kahler",
        "Fubini-Study",
        "Projective",
        "Hyperbolic"
    };
    if (type < GEOMETRY_TYPE_COUNT) {
        return names[type];
    }
    return "Unknown";
}

/**
 * @brief Get name of metric type
 */
static inline const char* metric_type_name(geometry_metric_t type) {
    static const char* names[] = {
        "Flat",
        "Curved",
        "Diagonal",
        "Block-Diagonal",
        "Hermitian",
        "Fisher",
        "Bures"
    };
    if (type < METRIC_COUNT) {
        return names[type];
    }
    return "Unknown";
}

/**
 * @brief Get name of connection type
 */
static inline const char* connection_type_name(geometry_connection_t type) {
    static const char* names[] = {
        "Levi-Civita",
        "Chern",
        "Berry",
        "Quantum-Geometric",
        "Affine"
    };
    if (type < CONNECTION_COUNT) {
        return names[type];
    }
    return "Unknown";
}

/**
 * @brief Get name of curvature type
 */
static inline const char* curvature_type_name(curvature_type_t type) {
    static const char* names[] = {
        "Riemann",
        "Ricci",
        "Scalar",
        "Weyl",
        "Berry",
        "Sectional"
    };
    if (type < CURVATURE_COUNT) {
        return names[type];
    }
    return "Unknown";
}

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_TYPES_H
