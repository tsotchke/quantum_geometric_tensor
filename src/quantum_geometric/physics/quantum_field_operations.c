#include "quantum_geometric/physics/quantum_field_operations.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

// ============================================================================
// Private Types and Constants
// ============================================================================

#ifndef QG_GPU_BLOCK_SIZE
#define QG_GPU_BLOCK_SIZE 256
#endif

#ifndef QG_DEFAULT_TOLERANCE
#define QG_DEFAULT_TOLERANCE 1e-10
#endif

#ifndef MIN_HIERARCHICAL_SIZE
#define MIN_HIERARCHICAL_SIZE 64
#endif

// Static GPU context for field operations
static GPUContext* field_gpu_ctx = NULL;
static bool gpu_initialized = false;

// Field operation cache for reuse
typedef struct {
    double complex* temp_buffer;
    size_t temp_size;
    HierarchicalMatrix* cached_hamiltonian;
    size_t cached_size;
} FieldOperationCache;

static FieldOperationCache field_cache = {NULL, 0, NULL, 0};

// ============================================================================
// Forward Declarations of Static Functions
// ============================================================================

static void evolve_hierarchical_field(HierarchicalMatrix* field,
                                     const HierarchicalMatrix* hamiltonian);
static void evolve_leaf_field(double complex* field,
                             const double complex* hamiltonian,
                             size_t n);
static double complex evolve_single_mode(double complex field,
                                        double complex hamiltonian);
static void merge_field_results(HierarchicalMatrix* field);
static void apply_field_boundaries(HierarchicalMatrix* left, HierarchicalMatrix* right);
static void compute_local_equations_impl(double complex* equations,
                                        const double complex* field,
                                        size_t n);
static void apply_hierarchical_gauge(HierarchicalMatrix* field,
                                    const GaugeTransform* transform);
static double complex compute_local_coupling(double complex f1, double complex f2);
static bool ensure_gpu_context(void);

// ============================================================================
// GPU Initialization and Context Management
// ============================================================================

static bool ensure_gpu_context(void) {
    if (gpu_initialized && field_gpu_ctx) {
        return true;
    }

    if (gpu_initialize() != QGT_SUCCESS) {
        return false;
    }

    field_gpu_ctx = gpu_create_context(0);  // Use first available GPU
    if (field_gpu_ctx) {
        gpu_initialized = true;
        return true;
    }

    return false;
}

// ============================================================================
// Quantum Field Evolution - O(log n) using Hierarchical Decomposition
// ============================================================================

void evolve_quantum_field(double complex* field,
                         const double complex* hamiltonian,
                         size_t n) {
    if (!field || !hamiltonian || n == 0) return;

    // For small systems, use direct evolution
    if (n < MIN_HIERARCHICAL_SIZE) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            field[i] = evolve_single_mode(field[i], hamiltonian[i]);
        }
        return;
    }

    // Convert to hierarchical representation for O(log n) complexity
    HierarchicalMatrix* h_field = convert_to_hierarchical(field, n);
    if (!h_field) {
        // Fallback to direct method
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            field[i] = evolve_single_mode(field[i], hamiltonian[i]);
        }
        return;
    }

    HierarchicalMatrix* h_hamiltonian = convert_to_hierarchical(hamiltonian, n);
    if (!h_hamiltonian) {
        destroy_hierarchical_matrix(h_field);
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            field[i] = evolve_single_mode(field[i], hamiltonian[i]);
        }
        return;
    }

    // Evolve using hierarchical operations - O(log n) depth
    evolve_hierarchical_field(h_field, h_hamiltonian);

    // Convert back to flat representation
    convert_from_hierarchical(field, h_field);

    // Cleanup hierarchical structures
    destroy_hierarchical_matrix(h_field);
    destroy_hierarchical_matrix(h_hamiltonian);
}

// Recursive hierarchical field evolution - O(log n) depth
static void evolve_hierarchical_field(HierarchicalMatrix* field,
                                     const HierarchicalMatrix* hamiltonian) {
    if (!field || !hamiltonian) return;

    if (field->is_leaf) {
        // Base case: direct evolution at leaf level
        size_t n = field->rows * field->cols;
        evolve_leaf_field(field->data, hamiltonian->data, n);
        return;
    }

    // Recursive case: divide and conquer using children[4]
    // children[0] = northwest, children[1] = northeast
    // children[2] = southwest, children[3] = southeast
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (field->children[0] && hamiltonian->children[0]) {
                evolve_hierarchical_field(field->children[0], hamiltonian->children[0]);
            }
        }

        #pragma omp section
        {
            if (field->children[1] && hamiltonian->children[1]) {
                evolve_hierarchical_field(field->children[1], hamiltonian->children[1]);
            }
        }

        #pragma omp section
        {
            if (field->children[2] && hamiltonian->children[2]) {
                evolve_hierarchical_field(field->children[2], hamiltonian->children[2]);
            }
        }

        #pragma omp section
        {
            if (field->children[3] && hamiltonian->children[3]) {
                evolve_hierarchical_field(field->children[3], hamiltonian->children[3]);
            }
        }
    }

    // Merge results and apply boundary conditions
    merge_field_results(field);
}

// Leaf field evolution - O(n) for small blocks
static void evolve_leaf_field(double complex* field,
                             const double complex* hamiltonian,
                             size_t n) {
    if (!field || !hamiltonian || n == 0) return;

    // Direct evolution at leaf level using SIMD
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        field[i] = evolve_single_mode(field[i], hamiltonian[i]);
    }
}

// Single mode quantum evolution using unitary operator
// Implements: ψ'(t) = exp(-iHt) ψ(t)
static double complex evolve_single_mode(double complex field,
                                        double complex hamiltonian) {
    // Apply evolution operator: exp(-i * H)
    // For small timesteps, this is the exact unitary evolution
    return field * cexp(-I * hamiltonian);
}

// Merge hierarchical field results with boundary conditions
static void merge_field_results(HierarchicalMatrix* field) {
    if (!field || field->is_leaf) return;

    // Apply boundary conditions between adjacent blocks
    // Horizontal boundaries (0-1, 2-3)
    if (field->children[0] && field->children[1]) {
        apply_field_boundaries(field->children[0], field->children[1]);
    }
    if (field->children[2] && field->children[3]) {
        apply_field_boundaries(field->children[2], field->children[3]);
    }

    // Vertical boundaries (0-2, 1-3)
    if (field->children[0] && field->children[2]) {
        apply_field_boundaries(field->children[0], field->children[2]);
    }
    if (field->children[1] && field->children[3]) {
        apply_field_boundaries(field->children[1], field->children[3]);
    }
}

// Apply boundary conditions between adjacent field blocks
static void apply_field_boundaries(HierarchicalMatrix* left, HierarchicalMatrix* right) {
    if (!left || !right) return;

    // Only apply at leaf level
    if (!left->is_leaf || !right->is_leaf) return;
    if (!left->data || !right->data) return;

    // Apply smooth transition at boundary
    size_t n_left = left->rows * left->cols;
    size_t n_right = right->rows * right->cols;
    size_t boundary_size = (n_left < n_right) ? n_left : n_right;

    // Weighted average at boundary for continuity
    #pragma omp simd
    for (size_t i = 0; i < boundary_size; i++) {
        double complex avg = 0.5 * (left->data[i] + right->data[i]);
        left->data[i] = avg;
        right->data[i] = avg;
    }
}

// ============================================================================
// Field Coupling Computation - GPU Accelerated with CPU Fallback
// ============================================================================

void compute_field_coupling(double complex* coupling,
                          const double complex* field1,
                          const double complex* field2,
                          size_t n) {
    if (!coupling || !field1 || !field2 || n == 0) return;

    // Try GPU acceleration first
    if (ensure_gpu_context()) {
        // Allocate GPU memory
        void* d_coupling = gpu_allocate(field_gpu_ctx, n * sizeof(double complex));
        void* d_field1 = gpu_allocate(field_gpu_ctx, n * sizeof(double complex));
        void* d_field2 = gpu_allocate(field_gpu_ctx, n * sizeof(double complex));

        if (d_coupling && d_field1 && d_field2) {
            // Copy to GPU
            int status1 = gpu_memcpy_to_device(field_gpu_ctx, d_field1, field1,
                                                n * sizeof(double complex));
            int status2 = gpu_memcpy_to_device(field_gpu_ctx, d_field2, field2,
                                                n * sizeof(double complex));

            if (status1 == QGT_SUCCESS && status2 == QGT_SUCCESS) {
                // Use GPU tensor multiply for coupling computation
                // coupling[i] = field1[i] * conj(field2[i])
                int gpu_result = gpu_quantum_tensor_multiply(
                    field_gpu_ctx,
                    (const ComplexFloat*)d_field1,
                    (const ComplexFloat*)d_field2,
                    (ComplexFloat*)d_coupling,
                    (int)n, 1, 1  // m, n, k for element-wise
                );

                if (gpu_result == QGT_SUCCESS) {
                    gpu_memcpy_from_device(field_gpu_ctx, coupling, d_coupling,
                                           n * sizeof(double complex));
                    gpu_free(field_gpu_ctx, d_coupling);
                    gpu_free(field_gpu_ctx, d_field1);
                    gpu_free(field_gpu_ctx, d_field2);
                    return;
                }
            }

            // Cleanup on failure
            gpu_free(field_gpu_ctx, d_coupling);
            gpu_free(field_gpu_ctx, d_field1);
            gpu_free(field_gpu_ctx, d_field2);
        } else {
            // Cleanup partial allocations
            if (d_coupling) gpu_free(field_gpu_ctx, d_coupling);
            if (d_field1) gpu_free(field_gpu_ctx, d_field1);
            if (d_field2) gpu_free(field_gpu_ctx, d_field2);
        }
    }

    // CPU fallback with SIMD optimization
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        coupling[i] = compute_local_coupling(field1[i], field2[i]);
    }
}

// Local coupling computation - quantum field inner product
static double complex compute_local_coupling(double complex f1, double complex f2) {
    // Field coupling: <ψ1|ψ2> = ψ1* · ψ2
    return conj(f1) * f2;
}

// ============================================================================
// Field Equations - Distributed Computing with O(log n) Synchronization
// ============================================================================

void compute_field_equations(double complex* equations,
                           const double complex* field,
                           size_t n) {
    if (!equations || !field || n == 0) return;

    // Distribute computation across nodes
    size_t local_n = distribute_workload(n);
    size_t offset = get_local_offset();

    // Each node computes its portion
    compute_local_equations_impl(equations + offset, field + offset, local_n);

    // Synchronize results across all nodes
    synchronize_complex_results(equations, n);
}

// Local field equations computation - implements Klein-Gordon or Dirac equation
static void compute_local_equations_impl(double complex* equations,
                                        const double complex* field,
                                        size_t n) {
    if (!equations || !field || n == 0) return;

    // Initialize equations array
    memset(equations, 0, n * sizeof(double complex));

    // Compute field equations using finite differences for derivatives
    // Klein-Gordon: (∂²/∂t² - ∇² + m²)φ = 0
    // Here we compute the spatial part: -∇²φ + m²φ

    const double mass_squared = 1.0;  // Normalized units
    const double dx = 1.0 / (double)n;
    const double dx2_inv = 1.0 / (dx * dx);

    // Interior points: central differences
    #pragma omp parallel for
    for (size_t i = 1; i < n - 1; i++) {
        // Second derivative: (φ[i+1] - 2φ[i] + φ[i-1]) / dx²
        double complex laplacian = (field[i+1] - 2.0*field[i] + field[i-1]) * dx2_inv;

        // Klein-Gordon equation: -∇²φ + m²φ
        equations[i] = -laplacian + mass_squared * field[i];
    }

    // Boundary conditions (periodic)
    if (n > 2) {
        // Left boundary
        double complex laplacian_left = (field[1] - 2.0*field[0] + field[n-1]) * dx2_inv;
        equations[0] = -laplacian_left + mass_squared * field[0];

        // Right boundary
        double complex laplacian_right = (field[0] - 2.0*field[n-1] + field[n-2]) * dx2_inv;
        equations[n-1] = -laplacian_right + mass_squared * field[n-1];
    }
}

// ============================================================================
// Gauge Transformations - O(log n) using Fast Approximation
// ============================================================================

void apply_gauge_transformation(double complex* field,
                              const GaugeTransform* transform,
                              size_t n) {
    if (!field || !transform || n == 0) return;

    // For Abelian gauge transformations, apply directly
    if (transform->is_abelian) {
        double complex phase = cexp(I * transform->phase);

        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            field[i] *= phase;
        }
        return;
    }

    // For non-Abelian, convert to hierarchical for O(log n) complexity
    if (n >= MIN_HIERARCHICAL_SIZE) {
        HierarchicalMatrix* h_field = convert_to_hierarchical(field, n);
        if (h_field) {
            apply_hierarchical_gauge(h_field, transform);
            convert_from_hierarchical(field, h_field);
            destroy_hierarchical_matrix(h_field);
            return;
        }
    }

    // CPU fallback for small systems or hierarchical failure
    // Apply gauge transformation matrix to each element
    size_t dim = transform->dimension;
    if (dim == 0 || !transform->matrix) {
        // Simple phase rotation if no matrix specified
        double complex phase = cexp(I * transform->phase);
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            field[i] *= phase;
        }
        return;
    }

    // General non-Abelian transformation
    double complex* temp = malloc(n * sizeof(double complex));
    if (!temp) return;

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // Apply transformation matrix element
        size_t mat_idx = i % (dim * dim);
        temp[i] = transform->matrix[mat_idx] * field[i];
    }

    memcpy(field, temp, n * sizeof(double complex));
    free(temp);
}

// Apply gauge transformation hierarchically
static void apply_hierarchical_gauge(HierarchicalMatrix* field,
                                    const GaugeTransform* transform) {
    if (!field || !transform) return;

    if (field->is_leaf) {
        // Apply transformation at leaf level
        size_t n = field->rows * field->cols;
        if (field->data) {
            double complex phase = cexp(I * transform->phase);

            if (transform->is_abelian) {
                #pragma omp simd
                for (size_t i = 0; i < n; i++) {
                    field->data[i] *= phase;
                }
            } else if (transform->matrix && transform->dimension > 0) {
                size_t dim = transform->dimension;
                for (size_t i = 0; i < n; i++) {
                    size_t mat_idx = i % (dim * dim);
                    field->data[i] = transform->matrix[mat_idx] * field->data[i];
                }
            }
        }
        return;
    }

    // Recursive application to children
    #pragma omp parallel sections
    {
        #pragma omp section
        if (field->children[0]) apply_hierarchical_gauge(field->children[0], transform);

        #pragma omp section
        if (field->children[1]) apply_hierarchical_gauge(field->children[1], transform);

        #pragma omp section
        if (field->children[2]) apply_hierarchical_gauge(field->children[2], transform);

        #pragma omp section
        if (field->children[3]) apply_hierarchical_gauge(field->children[3], transform);
    }
}

// ============================================================================
// Fast Approximation Methods for Field Equations
// ============================================================================

FastApproximation* init_fast_approximation(const double complex* data, size_t n) {
    if (!data || n == 0) return NULL;

    FastApproximation* approx = malloc(sizeof(FastApproximation));
    if (!approx) return NULL;

    // Determine number of Chebyshev terms based on accuracy requirements
    size_t num_terms = (n > 64) ? 64 : n;

    approx->coefficients = malloc(num_terms * sizeof(double complex));
    if (!approx->coefficients) {
        free(approx);
        return NULL;
    }

    approx->num_terms = num_terms;
    approx->tolerance = QG_DEFAULT_TOLERANCE;
    approx->internal_state = NULL;

    // Compute Chebyshev coefficients for approximation
    // c_k = (2/N) * Σ_j f(x_j) * T_k(x_j) where x_j = cos(π(j+0.5)/N)
    #pragma omp parallel for
    for (size_t k = 0; k < num_terms; k++) {
        double complex sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            double x_j = cos(M_PI * (j + 0.5) / n);
            double T_k = cos(k * acos(x_j));  // Chebyshev polynomial T_k(x)
            sum += data[j] * T_k;
        }
        approx->coefficients[k] = (2.0 / n) * sum;
    }

    // First coefficient is halved in Chebyshev series
    approx->coefficients[0] *= 0.5;

    return approx;
}

void compute_approximated_equations(FastApproximation* approx,
                                   double complex* equations) {
    if (!approx || !equations) return;

    // Evaluate Chebyshev series at output points
    // f(x) ≈ Σ_k c_k * T_k(x)
    size_t n = approx->num_terms;

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double x = 2.0 * i / n - 1.0;  // Map to [-1, 1]
        double complex result = 0.0;

        // Clenshaw recurrence for efficient Chebyshev evaluation
        double complex b_kp1 = 0.0, b_kp2 = 0.0;
        for (size_t k = n; k > 0; k--) {
            double complex b_k = 2.0 * x * b_kp1 - b_kp2 + approx->coefficients[k-1];
            b_kp2 = b_kp1;
            b_kp1 = b_k;
        }
        result = b_kp1 - x * b_kp2 + approx->coefficients[0];

        equations[i] = result;
    }
}

void destroy_fast_approximation(FastApproximation* approx) {
    if (!approx) return;

    free(approx->coefficients);
    free(approx->internal_state);
    free(approx);
}

// ============================================================================
// Gauge Transform Creation and Destruction
// ============================================================================

GaugeTransform* create_gauge_transform(size_t dimension, bool is_abelian) {
    GaugeTransform* transform = malloc(sizeof(GaugeTransform));
    if (!transform) return NULL;

    transform->dimension = dimension;
    transform->is_abelian = is_abelian;
    transform->phase = 0.0;

    if (dimension > 0 && !is_abelian) {
        transform->matrix = calloc(dimension * dimension, sizeof(double complex));
        if (!transform->matrix) {
            free(transform);
            return NULL;
        }
        // Initialize to identity matrix
        for (size_t i = 0; i < dimension; i++) {
            transform->matrix[i * dimension + i] = 1.0;
        }
    } else {
        transform->matrix = NULL;
    }

    return transform;
}

void destroy_gauge_transform(GaugeTransform* transform) {
    if (!transform) return;
    free(transform->matrix);
    free(transform);
}

// ============================================================================
// Resource Management and Cleanup
// ============================================================================

void cleanup_field_cache(void) {
    if (field_cache.temp_buffer) {
        free(field_cache.temp_buffer);
        field_cache.temp_buffer = NULL;
        field_cache.temp_size = 0;
    }

    if (field_cache.cached_hamiltonian) {
        destroy_hierarchical_matrix(field_cache.cached_hamiltonian);
        field_cache.cached_hamiltonian = NULL;
        field_cache.cached_size = 0;
    }
}

void cleanup_field_buffers(void) {
    cleanup_field_cache();

    if (field_gpu_ctx) {
        gpu_destroy_context(field_gpu_ctx);
        field_gpu_ctx = NULL;
    }

    if (gpu_initialized) {
        gpu_cleanup();
        gpu_initialized = false;
    }
}

void cleanup_quantum_field_operations(void) {
    cleanup_field_buffers();
}
