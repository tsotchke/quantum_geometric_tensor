#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// OpenMP support
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#endif

// Physical constants from header
#ifndef QG_HALF
#define QG_HALF 0.5
#endif
#ifndef QG_QUARTER
#define QG_QUARTER 0.25
#endif

// ============================================================================
// Helper Functions (Forward Declarations)
// ============================================================================

static complex double* calculate_field_strength(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z);

// ============================================================================
// Tensor Allocation
// ============================================================================

bool tensor_allocate(Tensor* tensor, size_t rank, const size_t* dims) {
    if (!tensor || !dims || rank == 0 || rank > QG_MAX_TENSOR_RANK) {
        return false;
    }

    tensor->rank = rank;
    tensor->total_size = 1;

    for (size_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
        tensor->total_size *= dims[i];
    }

    // Zero remaining dimensions
    for (size_t i = rank; i < QG_MAX_TENSOR_RANK; i++) {
        tensor->dims[i] = 0;
    }

    tensor->data = calloc(tensor->total_size, sizeof(complex double));
    if (!tensor->data) {
        tensor->total_size = 0;
        return false;
    }

    tensor->is_allocated = true;
    return true;
}

void tensor_free(Tensor* tensor) {
    if (tensor && tensor->is_allocated && tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
        tensor->is_allocated = false;
        tensor->total_size = 0;
    }
}

// ============================================================================
// Derivative Calculations
// ============================================================================

/**
 * Calculate partial derivatives of field at a spacetime point
 * Uses fourth-order central finite differences for accuracy
 */
complex double* calculate_derivatives(
    const Tensor* field_tensor,
    size_t t, size_t x, size_t y, size_t z)
{
    if (!field_tensor || !field_tensor->data) return NULL;

    size_t n = field_tensor->dims[4];  // Number of internal components
    size_t nt = field_tensor->dims[0];
    size_t nx = field_tensor->dims[1];
    size_t ny = field_tensor->dims[2];
    size_t nz = field_tensor->dims[3];

    // Allocate derivative array: QG_SPACETIME_DIMS directions × n components
    complex double* derivatives = calloc(QG_SPACETIME_DIMS * n, sizeof(complex double));
    if (!derivatives) return NULL;

    // Helper macro for index calculation
    #define IDX(tt, xx, yy, zz) \
        (((((tt) % nt) * nx + ((xx) % nx)) * ny + ((yy) % ny)) * nz + ((zz) % nz))

    // Fourth-order central difference coefficients
    const double coeff[3] = {8.0/12.0, -1.0/12.0, 0.0};

    // Calculate derivatives in each direction
    for (size_t i = 0; i < n; i++) {
        // Time derivative (μ = 0)
        size_t tp1 = (t + 1) % nt, tm1 = (t + nt - 1) % nt;
        size_t tp2 = (t + 2) % nt, tm2 = (t + nt - 2) % nt;

        derivatives[0 * n + i] =
            coeff[0] * (field_tensor->data[IDX(tp1, x, y, z) * n + i] -
                       field_tensor->data[IDX(tm1, x, y, z) * n + i]) +
            coeff[1] * (field_tensor->data[IDX(tp2, x, y, z) * n + i] -
                       field_tensor->data[IDX(tm2, x, y, z) * n + i]);

        // X derivative (μ = 1)
        size_t xp1 = (x + 1) % nx, xm1 = (x + nx - 1) % nx;
        size_t xp2 = (x + 2) % nx, xm2 = (x + nx - 2) % nx;

        derivatives[1 * n + i] =
            coeff[0] * (field_tensor->data[IDX(t, xp1, y, z) * n + i] -
                       field_tensor->data[IDX(t, xm1, y, z) * n + i]) +
            coeff[1] * (field_tensor->data[IDX(t, xp2, y, z) * n + i] -
                       field_tensor->data[IDX(t, xm2, y, z) * n + i]);

        // Y derivative (μ = 2)
        size_t yp1 = (y + 1) % ny, ym1 = (y + ny - 1) % ny;
        size_t yp2 = (y + 2) % ny, ym2 = (y + ny - 2) % ny;

        derivatives[2 * n + i] =
            coeff[0] * (field_tensor->data[IDX(t, x, yp1, z) * n + i] -
                       field_tensor->data[IDX(t, x, ym1, z) * n + i]) +
            coeff[1] * (field_tensor->data[IDX(t, x, yp2, z) * n + i] -
                       field_tensor->data[IDX(t, x, ym2, z) * n + i]);

        // Z derivative (μ = 3)
        size_t zp1 = (z + 1) % nz, zm1 = (z + nz - 1) % nz;
        size_t zp2 = (z + 2) % nz, zm2 = (z + nz - 2) % nz;

        derivatives[3 * n + i] =
            coeff[0] * (field_tensor->data[IDX(t, x, y, zp1) * n + i] -
                       field_tensor->data[IDX(t, x, y, zm1) * n + i]) +
            coeff[1] * (field_tensor->data[IDX(t, x, y, zp2) * n + i] -
                       field_tensor->data[IDX(t, x, y, zm2) * n + i]);
    }

    #undef IDX
    return derivatives;
}

/**
 * Calculate covariant derivatives including gauge field coupling
 * D_μ φ = ∂_μ φ - ig A_μ φ
 */
complex double* calculate_covariant_derivatives(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z)
{
    if (!field || !field->field_tensor) return NULL;

    size_t n = field->field_tensor->dims[4];

    // Get partial derivatives
    complex double* partial = calculate_derivatives(field->field_tensor, t, x, y, z);
    if (!partial) return NULL;

    // If no gauge field, just return partial derivatives
    if (!field->has_gauge_field || !field->gauge_field) {
        return partial;
    }

    // Add gauge coupling: D_μ φ = ∂_μ φ - ig A_μ φ
    size_t nt = field->field_tensor->dims[0];
    size_t nx = field->field_tensor->dims[1];
    size_t ny = field->field_tensor->dims[2];
    size_t nz = field->field_tensor->dims[3];
    size_t ng = field->gauge_group_dim;

    // Bounds check for spacetime coordinates
    if (t >= nt || x >= nx || y >= ny || z >= nz) {
        return partial;  // Out of bounds
    }

    size_t idx = ((t * nx + x) * ny + y) * nz + z;
    complex double g = I * field->field_strength;

    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
        for (size_t i = 0; i < n; i++) {
            // Sum over gauge group generators
            complex double gauge_term = 0.0;
            for (size_t a = 0; a < ng; a++) {
                // A_μ^a(x) * T^a * φ(x)
                // For simplicity, using diagonal generator approximation
                size_t gauge_idx = (idx * QG_SPACETIME_DIMS + mu) * ng + a;
                if (gauge_idx < field->gauge_field->total_size) {
                    gauge_term += field->gauge_field->data[gauge_idx] *
                                 field->field_tensor->data[idx * n + i];
                }
            }
            partial[mu * n + i] -= g * gauge_term;
        }
    }

    return partial;
}

// ============================================================================
// Gauge Field Operations
// ============================================================================

/**
 * Transform a single generator component of the gauge field
 */
void transform_generator(
    Tensor* gauge_field,
    const Tensor* transformation,
    size_t t, size_t x, size_t y, size_t z, size_t g)
{
    if (!gauge_field || !transformation || !gauge_field->data) return;

    size_t nx = gauge_field->dims[1];
    size_t ny = gauge_field->dims[2];
    size_t nz = gauge_field->dims[3];
    size_t ng = gauge_field->dims[4];
    size_t nc = gauge_field->dims[5];

    size_t idx = ((t * nx + x) * ny + y) * nz + z;

    // Apply U A_μ U† transformation for each Lorentz component
    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
        complex double original = 0.0;
        size_t comp_idx = (idx * QG_SPACETIME_DIMS + mu) * ng + g;
        if (comp_idx * nc < gauge_field->total_size) {
            original = gauge_field->data[comp_idx * nc];
        }

        // Apply transformation matrix
        complex double transformed = 0.0;
        for (size_t a = 0; a < ng && a < transformation->dims[0]; a++) {
            for (size_t b = 0; b < ng && b < transformation->dims[1]; b++) {
                size_t u_idx = a * transformation->dims[1] + b;
                if (u_idx < transformation->total_size) {
                    // U_{ag} * A * U†_{gb} contribution
                    if (a == g) {
                        transformed += transformation->data[u_idx] * original *
                                      conj(transformation->data[b * transformation->dims[1] + g]);
                    }
                }
            }
        }

        if (comp_idx * nc < gauge_field->total_size) {
            gauge_field->data[comp_idx * nc] = transformed;
        }
    }
}

/**
 * Calculate field strength tensor F_μν at a spacetime point
 * F_μν = ∂_μ A_ν - ∂_ν A_μ + ig[A_μ, A_ν]
 */
static complex double* calculate_field_strength(
    const QuantumField* field,
    size_t t, size_t x, size_t y, size_t z)
{
    if (!field || !field->gauge_field) return NULL;

    // Allocate field strength tensor (4x4 matrix)
    complex double* F = calloc(QG_SPACETIME_DIMS * QG_SPACETIME_DIMS, sizeof(complex double));
    if (!F) return NULL;

    size_t nt = field->gauge_field->dims[0];
    size_t nx = field->gauge_field->dims[1];
    size_t ny = field->gauge_field->dims[2];
    size_t nz = field->gauge_field->dims[3];
    size_t ng = field->gauge_field->dims[4];

    // Helper for gauge field index
    #define GAUGE_IDX(tt, xx, yy, zz, mu) \
        ((((((tt) % nt) * nx + ((xx) % nx)) * ny + ((yy) % ny)) * nz + ((zz) % nz)) * QG_SPACETIME_DIMS + (mu))

    // Calculate F_μν for each component
    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
        for (size_t nu = mu + 1; nu < QG_SPACETIME_DIMS; nu++) {
            complex double F_munu = 0.0;

            // Sum over gauge group generators
            for (size_t a = 0; a < ng; a++) {
                // ∂_μ A_ν - ∂_ν A_μ using central differences
                complex double dmu_Anu = 0.0;
                complex double dnu_Amu = 0.0;

                // Shift indices based on direction
                size_t shifts[4][4] = {
                    {(t + 1) % nt, x, y, z},  // +t
                    {t, (x + 1) % nx, y, z},  // +x
                    {t, x, (y + 1) % ny, z},  // +y
                    {t, x, y, (z + 1) % nz}   // +z
                };
                size_t mshifts[4][4] = {
                    {(t + nt - 1) % nt, x, y, z},  // -t
                    {t, (x + nx - 1) % nx, y, z},  // -x
                    {t, x, (y + ny - 1) % ny, z},  // -y
                    {t, x, y, (z + nz - 1) % nz}   // -z
                };

                size_t idx_p_mu = GAUGE_IDX(shifts[mu][0], shifts[mu][1], shifts[mu][2], shifts[mu][3], nu);
                size_t idx_m_mu = GAUGE_IDX(mshifts[mu][0], mshifts[mu][1], mshifts[mu][2], mshifts[mu][3], nu);
                size_t idx_p_nu = GAUGE_IDX(shifts[nu][0], shifts[nu][1], shifts[nu][2], shifts[nu][3], mu);
                size_t idx_m_nu = GAUGE_IDX(mshifts[nu][0], mshifts[nu][1], mshifts[nu][2], mshifts[nu][3], mu);

                if ((idx_p_mu + 1) * ng <= field->gauge_field->total_size &&
                    (idx_m_mu + 1) * ng <= field->gauge_field->total_size) {
                    dmu_Anu = 0.5 * (field->gauge_field->data[idx_p_mu * ng + a] -
                                   field->gauge_field->data[idx_m_mu * ng + a]);
                }

                if ((idx_p_nu + 1) * ng <= field->gauge_field->total_size &&
                    (idx_m_nu + 1) * ng <= field->gauge_field->total_size) {
                    dnu_Amu = 0.5 * (field->gauge_field->data[idx_p_nu * ng + a] -
                                   field->gauge_field->data[idx_m_nu * ng + a]);
                }

                F_munu += dmu_Anu - dnu_Amu;
            }

            // Non-abelian commutator term ig[A_μ, A_ν] (structure constants)
            // For U(1) this is zero; for non-abelian groups add structure constants
            size_t idx = GAUGE_IDX(t, x, y, z, 0);
            (void)idx;  // Used for non-abelian extension

            F[mu * QG_SPACETIME_DIMS + nu] = F_munu;
            F[nu * QG_SPACETIME_DIMS + mu] = -F_munu;  // Antisymmetric
        }
    }

    #undef GAUGE_IDX
    return F;
}

// ============================================================================
// Field Initialization and Cleanup
// ============================================================================

bool init_quantum_field(
    QuantumField* field,
    const size_t* lattice_dims,
    size_t num_components,
    double mass,
    double coupling)
{
    if (!field || !lattice_dims || num_components == 0) return false;

    memset(field, 0, sizeof(QuantumField));

    // Allocate field tensor
    field->field_tensor = malloc(sizeof(Tensor));
    if (!field->field_tensor) return false;

    size_t dims[5] = {lattice_dims[0], lattice_dims[1], lattice_dims[2],
                      lattice_dims[3], num_components};
    if (!tensor_allocate(field->field_tensor, 5, dims)) {
        free(field->field_tensor);
        return false;
    }

    // Allocate conjugate momentum
    field->conjugate_momentum = malloc(sizeof(Tensor));
    if (!field->conjugate_momentum) {
        tensor_free(field->field_tensor);
        free(field->field_tensor);
        return false;
    }
    if (!tensor_allocate(field->conjugate_momentum, 5, dims)) {
        tensor_free(field->field_tensor);
        free(field->field_tensor);
        free(field->conjugate_momentum);
        return false;
    }

    // Set physical parameters
    field->mass = mass;
    field->coupling = coupling;
    field->num_components = num_components;

    // Initialize Minkowski metric (- + + +)
    memset(field->metric, 0, sizeof(field->metric));
    field->metric[0] = -1.0;  // g_00
    field->metric[5] = 1.0;   // g_11
    field->metric[10] = 1.0;  // g_22
    field->metric[15] = 1.0;  // g_33

    // Default lattice spacing
    for (size_t i = 0; i < QG_SPACETIME_DIMS; i++) {
        field->lattice_spacing[i] = 1.0;
        field->periodic_bc[i] = true;
    }

    field->is_initialized = true;
    field->has_gauge_field = false;
    field->gauge_field = NULL;

    return true;
}

void cleanup_quantum_field(QuantumField* field) {
    if (!field) return;

    if (field->field_tensor) {
        tensor_free(field->field_tensor);
        free(field->field_tensor);
        field->field_tensor = NULL;
    }

    if (field->conjugate_momentum) {
        tensor_free(field->conjugate_momentum);
        free(field->conjugate_momentum);
        field->conjugate_momentum = NULL;
    }

    if (field->gauge_field) {
        tensor_free(field->gauge_field);
        free(field->gauge_field);
        field->gauge_field = NULL;
    }

    field->is_initialized = false;
}

bool init_gauge_field(
    QuantumField* field,
    size_t gauge_group_dim,
    double gauge_coupling)
{
    if (!field || !field->is_initialized || gauge_group_dim == 0) return false;

    field->gauge_field = malloc(sizeof(Tensor));
    if (!field->gauge_field) return false;

    // Gauge field dimensions: [t, x, y, z, μ, generator]
    size_t dims[6] = {
        field->field_tensor->dims[0],
        field->field_tensor->dims[1],
        field->field_tensor->dims[2],
        field->field_tensor->dims[3],
        QG_SPACETIME_DIMS,
        gauge_group_dim
    };

    // Use 6-dimensional tensor for full gauge structure
    Tensor* gauge = field->gauge_field;
    gauge->rank = 6;
    gauge->total_size = 1;
    for (size_t i = 0; i < 6; i++) {
        gauge->dims[i] = dims[i];
        gauge->total_size *= dims[i];
    }

    gauge->data = calloc(gauge->total_size, sizeof(complex double));
    if (!gauge->data) {
        free(field->gauge_field);
        field->gauge_field = NULL;
        return false;
    }
    gauge->is_allocated = true;

    field->gauge_group_dim = gauge_group_dim;
    field->field_strength = gauge_coupling;
    field->has_gauge_field = true;

    return true;
}

// ============================================================================
// Total Hamiltonian
// ============================================================================

double calculate_hamiltonian(const QuantumField* field) {
    if (!field || !field->is_initialized) return 0.0;

    double H = calculate_kinetic_energy(field) + calculate_potential_energy(field);

    if (field->has_gauge_field) {
        H += calculate_gauge_energy(field);
    }

    return H;
}

// Transform field components at a spacetime point
void transform_field_components(
    Tensor* field,
    const Tensor* transformation,
    size_t t,
    size_t x,
    size_t y,
    size_t z) {
    
    size_t num_components = field->dims[4];
    
    // Get field components at point
    complex double* components = malloc(
        num_components * sizeof(complex double)
    );
    
    for (size_t i = 0; i < num_components; i++) {
        size_t idx = ((t * field->dims[1] + x) * field->dims[2] + y)
                    * field->dims[3] + z;
        components[i] = field->data[idx * field->dims[4] + i];
    }
    
    // Apply transformation
    complex double* transformed = malloc(
        num_components * sizeof(complex double)
    );
    
    for (size_t i = 0; i < num_components; i++) {
        transformed[i] = 0;
        for (size_t j = 0; j < num_components; j++) {
            size_t trans_idx = i * num_components + j;
            transformed[i] += transformation->data[trans_idx] * components[j];
        }
    }
    
    // Update field
    for (size_t i = 0; i < num_components; i++) {
        size_t idx = ((t * field->dims[1] + x) * field->dims[2] + y)
                    * field->dims[3] + z;
        field->data[idx * field->dims[4] + i] = transformed[i];
    }
    
    free(components);
    free(transformed);
}

// Transform gauge field
void transform_gauge_field(
    Tensor* gauge_field,
    const Tensor* transformation) {

    size_t num_components = gauge_field->dims[5];
    size_t num_generators = gauge_field->dims[4];

    // Validate transformation tensor has matching structure
    if (transformation->dims[0] < num_generators ||
        transformation->dims[1] < num_components) {
        return;  // Incompatible transformation
    }

    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < gauge_field->dims[0]; t++) {
        for (size_t x = 0; x < gauge_field->dims[1]; x++) {
            for (size_t y = 0; y < gauge_field->dims[2]; y++) {
                for (size_t z = 0; z < gauge_field->dims[3]; z++) {
                    // Transform each generator
                    for (size_t g = 0; g < num_generators; g++) {
                        transform_generator(
                            gauge_field,
                            transformation,
                            t, x, y, z, g
                        );
                    }
                }
            }
        }
    }
}

// Calculate kinetic terms
void calculate_kinetic_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    // Calculate derivatives
                    complex double* derivatives = calculate_derivatives(
                        field->field_tensor,
                        t, x, y, z
                    );
                    
                    // Contract with metric
                    for (size_t i = 0; i < n; i++) {
                        complex double sum = 0;
                        for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                            for (size_t nu = 0; nu < QG_SPACETIME_DIMS; nu++) {
                                sum += field->metric[mu * QG_SPACETIME_DIMS + nu] *
                                      derivatives[mu * n + i] *
                                      conj(derivatives[nu * n + i]);
                            }
                        }
                        
                        size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                    field->field_tensor->dims[2] + y) *
                                    field->field_tensor->dims[3] + z;
                        equations->data[idx * n + i] = sum;
                    }
                    
                    free(derivatives);
                }
            }
        }
    }
}

// Add mass terms
void add_mass_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    double mass_squared = field->mass * field->mass;
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        equations->data[idx * n + i] +=
                            mass_squared * field->field_tensor->data[idx * n + i];
                    }
                }
            }
        }
    }
}

// Add interaction terms
void add_interaction_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    // Calculate field magnitude squared
                    double phi_squared = 0;
                    for (size_t i = 0; i < n; i++) {
                        complex double phi = field->field_tensor->data[idx * n + i];
                        phi_squared += creal(phi * conj(phi));
                    }
                    
                    // Add phi^4 interaction
                    for (size_t i = 0; i < n; i++) {
                        equations->data[idx * n + i] +=
                            field->coupling * phi_squared *
                            field->field_tensor->data[idx * n + i];
                    }
                }
            }
        }
    }
}

// Add gauge coupling
void add_gauge_coupling(
    const QuantumField* field,
    Tensor* equations) {
    
    if (!field->gauge_field) return;
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    // Calculate covariant derivatives
                    complex double* cov_derivatives = calculate_covariant_derivatives(
                        field,
                        t, x, y, z
                    );
                    
                    // Add to equations
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                            equations->data[idx * n + i] +=
                                field->field_strength * cov_derivatives[mu * n + i];
                        }
                    }
                    
                    free(cov_derivatives);
                }
            }
        }
    }
}

// Calculate field energies
double calculate_kinetic_energy(const QuantumField* field) {
    double energy = 0.0;
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        complex double pi = field->conjugate_momentum->data[idx * n + i];
                        energy += creal(pi * conj(pi));
                    }
                }
            }
        }
    }
    
    return QG_HALF * energy;
}

double calculate_potential_energy(const QuantumField* field) {
    double energy = 0.0;
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    double phi_squared = 0;
                    for (size_t i = 0; i < n; i++) {
                        complex double phi = field->field_tensor->data[idx * n + i];
                        phi_squared += creal(phi * conj(phi));
                    }
                    
                    energy += QG_HALF * field->mass * field->mass * phi_squared +
                             QG_QUARTER * field->coupling * phi_squared * phi_squared;
                }
            }
        }
    }
    
    return energy;
}

double calculate_gauge_energy(const QuantumField* field) {
    if (!field->gauge_field) return 0.0;

    double energy = 0.0;
    // n is the number of field components (color indices for gauge fields)
    size_t n = field->gauge_field->dims[5];

    // Energy normalization factor based on field components
    double norm_factor = (n > 0) ? 1.0 / (double)n : 1.0;

    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->gauge_field->dims[0]; t++) {
        for (size_t x = 0; x < field->gauge_field->dims[1]; x++) {
            for (size_t y = 0; y < field->gauge_field->dims[2]; y++) {
                for (size_t z = 0; z < field->gauge_field->dims[3]; z++) {
                    // Calculate field strength tensor
                    complex double* F_munu = calculate_field_strength(
                        field,
                        t, x, y, z
                    );

                    // Calculate energy density: E = Tr(F_μν F^μν) / 4
                    double local_energy = 0.0;
                    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                        for (size_t nu = 0; nu < QG_SPACETIME_DIMS; nu++) {
                            local_energy += creal(F_munu[mu * QG_SPACETIME_DIMS + nu] *
                                         conj(F_munu[mu * QG_SPACETIME_DIMS + nu]));
                        }
                    }
                    energy += local_energy * norm_factor;

                    free(F_munu);
                }
            }
        }
    }
    
    return QG_QUARTER * field->field_strength * field->field_strength * energy;
}
