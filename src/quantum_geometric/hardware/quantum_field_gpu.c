#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/physics/quantum_field_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>

// Include feature detection header if available
#if __has_include("quantum_geometric/core/qgtl_features.h")
#include "quantum_geometric/core/qgtl_features.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Map QGTL feature macros to legacy macros for compatibility
#ifdef QGTL_HAS_METAL
#if QGTL_HAS_METAL && !defined(HAVE_METAL)
#define HAVE_METAL 1
#endif
#endif

#ifdef QGTL_HAS_CUDA
#if QGTL_HAS_CUDA && !defined(HAVE_CUDA)
#define HAVE_CUDA 1
#endif
#endif

// Also check Apple platform for Metal
#if defined(__APPLE__) && !defined(HAVE_METAL)
#define HAVE_METAL 1
#endif

// GPU backend state
static struct {
    GPUBackendType type;
    bool initialized;
    void* context;
} gpu_backend = {
    .type = GPU_BACKEND_NONE,
    .initialized = false,
    .context = NULL
};

// Forward declarations for CPU implementations (defined below)
int apply_rotation_cpu(QuantumField* field, size_t qubit, double theta, double phi);
double calculate_field_energy_cpu(const QuantumField* field);
int calculate_field_equations_cpu(const QuantumField* field, Tensor* equations);

// ============================================================================
// CUDA Backend (stub implementations when CUDA not available)
// ============================================================================

#ifdef HAVE_CUDA
// External CUDA functions - actual implementations in CUDA file
extern int apply_rotation_cuda(QuantumField* field, size_t qubit, double theta, double phi);
extern double calculate_field_energy_cuda(const QuantumField* field);
extern int calculate_field_equations_cuda(const QuantumField* field, Tensor* equations);
#else
// CPU fallback when CUDA not available
int apply_rotation_cuda(QuantumField* field, size_t qubit, double theta, double phi) {
    return apply_rotation_cpu(field, qubit, theta, phi);
}

double calculate_field_energy_cuda(const QuantumField* field) {
    return calculate_field_energy_cpu(field);
}

int calculate_field_equations_cuda(const QuantumField* field, Tensor* equations) {
    return calculate_field_equations_cpu(field, equations);
}
#endif

// ============================================================================
// Metal Backend (stub implementations when Metal not available)
// ============================================================================

#ifdef HAVE_METAL
// External Metal functions - actual implementations in Metal/Objective-C file
extern int apply_rotation_metal(QuantumField* field, size_t qubit, double theta, double phi);
extern double calculate_field_energy_metal(const QuantumField* field);
extern int calculate_field_equations_metal(const QuantumField* field, Tensor* equations);
extern bool init_metal_backend(void** context);
extern void cleanup_metal_backend(void* context);
extern const char* get_metal_device_name(void* context);
#else
// CPU fallback when Metal not available
int apply_rotation_metal(QuantumField* field, size_t qubit, double theta, double phi) {
    return apply_rotation_cpu(field, qubit, theta, phi);
}

double calculate_field_energy_metal(const QuantumField* field) {
    return calculate_field_energy_cpu(field);
}

int calculate_field_equations_metal(const QuantumField* field, Tensor* equations) {
    return calculate_field_equations_cpu(field, equations);
}

bool init_metal_backend(void** context) {
    (void)context;
    return false;
}

void cleanup_metal_backend(void* context) {
    (void)context;
}

const char* get_metal_device_name(void* context) {
    (void)context;
    return "No Metal Device";
}
#endif

// Initialize GPU backend
// Backend availability is determined at build time via CMake
static bool init_gpu_backend(void) {
    if (gpu_backend.initialized) {
        return gpu_backend.type != GPU_BACKEND_NONE;
    }

    // Backend selection is determined at compile time
    #if defined(HAVE_CUDA)
    gpu_backend.type = GPU_BACKEND_CUDA;
    gpu_backend.initialized = true;
    return true;
    #elif defined(HAVE_METAL)
    if (init_metal_backend(&gpu_backend.context)) {
        gpu_backend.type = GPU_BACKEND_METAL;
        gpu_backend.initialized = true;
        return true;
    }
    // Metal init failed
    gpu_backend.type = GPU_BACKEND_NONE;
    gpu_backend.initialized = true;
    return false;
    #else
    // No GPU backend compiled in
    gpu_backend.type = GPU_BACKEND_NONE;
    gpu_backend.initialized = true;
    return false;
    #endif
}

// Apply rotation using GPU
int apply_rotation_gpu(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi) {
    
    if (!init_gpu_backend()) {
        return -1;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return apply_rotation_cuda(field, qubit, theta, phi);
            
        case GPU_BACKEND_METAL:
            return apply_rotation_metal(field, qubit, theta, phi);
            
        default:
            return -1;
    }
}

// Calculate field energy using GPU
double calculate_field_energy_gpu(const QuantumField* field) {
    if (!init_gpu_backend()) {
        return 0.0;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return calculate_field_energy_cuda(field);
            
        case GPU_BACKEND_METAL:
            return calculate_field_energy_metal(field);
            
        default:
            return 0.0;
    }
}

// Calculate field equations using GPU
int calculate_field_equations_gpu(
    const QuantumField* field,
    Tensor* equations) {
    
    if (!init_gpu_backend()) {
        return -1;
    }
    
    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return calculate_field_equations_cuda(field, equations);
            
        case GPU_BACKEND_METAL:
            return calculate_field_equations_metal(field, equations);
            
        default:
            return -1;
    }
}

// Get GPU backend type - internal helper (public API in quantum_field_gpu_monitor.c)
static GPUBackendType get_gpu_backend_type_internal(void) {
    if (!init_gpu_backend()) {
        return GPU_BACKEND_NONE;
    }
    return gpu_backend.type;
}

// Check if GPU acceleration is available
bool has_gpu_acceleration() {
    return init_gpu_backend() && gpu_backend.type != GPU_BACKEND_NONE;
}

// Get GPU device name
const char* get_gpu_device_name(void) {
    if (!init_gpu_backend()) {
        return "No GPU";
    }

    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            return "CUDA Device";

        case GPU_BACKEND_METAL:
            return get_metal_device_name(gpu_backend.context);

        default:
            return "No GPU";
    }
}

// Clean up GPU backend
void cleanup_gpu_backend(void) {
    if (!gpu_backend.initialized) {
        return;
    }

    switch (gpu_backend.type) {
        case GPU_BACKEND_CUDA:
            // CUDA cleanup handled by runtime
            break;

        case GPU_BACKEND_METAL:
            cleanup_metal_backend(gpu_backend.context);
            break;

        default:
            break;
    }

    gpu_backend.type = GPU_BACKEND_NONE;
    gpu_backend.initialized = false;
    gpu_backend.context = NULL;
}

// ============================================================================
// Error and Monitoring Functions
// ============================================================================
// NOTE: Error handling functions (gpu_error_string, get_last_gpu_error,
// clear_gpu_error, is_gpu_available) and performance monitoring functions
// (get_gpu_memory_usage, get_gpu_utilization, get_gpu_temperature,
// get_gpu_power_usage) are implemented in quantum_field_gpu_monitor.c

// ============================================================================
// Tensor Operations for Tests
// ============================================================================

Tensor* init_tensor(const size_t* dims, size_t rank) {
    if (!dims || rank == 0 || rank > QG_MAX_TENSOR_RANK) {
        return NULL;
    }

    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->rank = rank;
    tensor->total_size = 1;

    for (size_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
        tensor->total_size *= dims[i];
    }

    tensor->data = (complex double*)calloc(tensor->total_size, sizeof(complex double));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }

    tensor->is_allocated = true;
    return tensor;
}

void cleanup_tensor(Tensor* tensor) {
    if (!tensor) return;

    if (tensor->is_allocated && tensor->data) {
        free(tensor->data);
    }
    free(tensor);
}

// ============================================================================
// Extended QuantumField for GPU operations
// ============================================================================

typedef struct QuantumFieldExtended {
    QuantumField base;
    complex double* components;
    size_t total_size;
    double* metric_data;
    double* connection_data;
    double* curvature_data;
} QuantumFieldExtended;

// ============================================================================
// Quantum Field Initialization (for test compatibility)
// ============================================================================

QuantumField* init_quantum_field(const FieldConfig* config, const GeometricConfig* geom) {
    if (!config) return NULL;

    QuantumFieldExtended* ext = (QuantumFieldExtended*)calloc(1, sizeof(QuantumFieldExtended));
    if (!ext) return NULL;

    QuantumField* field = &ext->base;

    // Calculate total lattice size (4D lattice)
    size_t lattice_vol = config->lattice_size * config->lattice_size *
                         config->lattice_size * config->lattice_size;
    ext->total_size = lattice_vol * config->num_components;

    // Allocate field tensor
    size_t tensor_dims[5] = {
        config->lattice_size,
        config->lattice_size,
        config->lattice_size,
        config->lattice_size,
        config->num_components
    };

    field->field_tensor = (Tensor*)calloc(1, sizeof(Tensor));
    if (!field->field_tensor) {
        free(ext);
        return NULL;
    }

    if (!tensor_allocate(field->field_tensor, 5, tensor_dims)) {
        free(field->field_tensor);
        free(ext);
        return NULL;
    }

    // Allocate components array
    ext->components = (complex double*)calloc(ext->total_size, sizeof(complex double));
    if (!ext->components) {
        tensor_free(field->field_tensor);
        free(field->field_tensor);
        free(ext);
        return NULL;
    }

    // Initialize with random values
    for (size_t i = 0; i < ext->total_size; i++) {
        double re = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double im = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        ext->components[i] = re + I * im;
        field->field_tensor->data[i] = ext->components[i];
    }

    // Set field parameters
    field->mass = config->mass;
    field->coupling = config->coupling;
    field->field_strength = config->field_strength;
    field->num_components = config->num_components;

    // Set lattice spacing
    for (int i = 0; i < QG_SPACETIME_DIMS; i++) {
        field->lattice_spacing[i] = 1.0;
        field->periodic_bc[i] = true;
    }

    // Copy metric if provided
    if (geom && geom->metric) {
        for (int i = 0; i < QG_SPACETIME_DIMS * QG_SPACETIME_DIMS; i++) {
            field->metric[i] = geom->metric[i];
        }
    } else {
        // Default Minkowski metric
        for (int i = 0; i < QG_SPACETIME_DIMS; i++) {
            field->metric[i * QG_SPACETIME_DIMS + i] = (i == 0) ? -1.0 : 1.0;
        }
    }

    // Initialize gauge field if requested
    if (config->gauge_group && config->num_generators > 0) {
        field->gauge_group_dim = config->num_generators;
        init_gauge_field(field, config->num_generators, config->field_strength);
    }

    field->is_initialized = true;
    return field;
}

// Note: cleanup_quantum_field is defined in quantum_field_calculations.c
// This is the GPU-extended version that cleans up additional GPU resources
void cleanup_quantum_field_gpu(QuantumField* field) {
    if (!field) return;

    QuantumFieldExtended* ext = (QuantumFieldExtended*)field;

    if (field->field_tensor) {
        tensor_free(field->field_tensor);
        free(field->field_tensor);
    }

    if (field->conjugate_momentum) {
        tensor_free(field->conjugate_momentum);
        free(field->conjugate_momentum);
    }

    if (field->gauge_field) {
        tensor_free(field->gauge_field);
        free(field->gauge_field);
    }

    free(ext->components);
    free(ext->metric_data);
    free(ext->connection_data);
    free(ext->curvature_data);
    free(ext);
}

// ============================================================================
// CPU Implementations for Benchmarking
// ============================================================================

int apply_rotation_cpu(QuantumField* field, size_t qubit, double theta, double phi) {
    if (!field || !field->field_tensor) {
        return -1;
    }

    QuantumFieldExtended* ext = (QuantumFieldExtended*)field;
    if (!ext->components) return -1;

    // Apply rotation using RZ(phi) RY(theta) decomposition
    double cos_t = cos(theta / 2.0);
    double sin_t = sin(theta / 2.0);
    complex double phase = cexp(I * phi);

    // Apply to each field component pair (simulated qubit)
    size_t stride = 1ULL << qubit;
    size_t num_pairs = ext->total_size / (2 * stride);

    for (size_t pair = 0; pair < num_pairs; pair++) {
        size_t idx0 = (pair / stride) * 2 * stride + (pair % stride);
        size_t idx1 = idx0 + stride;

        if (idx1 < ext->total_size) {
            complex double a = ext->components[idx0];
            complex double b = ext->components[idx1];

            ext->components[idx0] = cos_t * a - sin_t * phase * b;
            ext->components[idx1] = sin_t * conj(phase) * a + cos_t * b;

            // Also update field_tensor
            field->field_tensor->data[idx0] = ext->components[idx0];
            field->field_tensor->data[idx1] = ext->components[idx1];
        }
    }

    return 0;
}

double calculate_field_energy_cpu(const QuantumField* field) {
    if (!field || !field->field_tensor) {
        return 0.0;
    }

    double energy = 0.0;
    size_t total = field->field_tensor->total_size;

    // Kinetic energy contribution: sum of |amplitude|^2
    for (size_t i = 0; i < total; i++) {
        double amp_sq = cabs(field->field_tensor->data[i]) * cabs(field->field_tensor->data[i]);
        energy += amp_sq;
    }

    // Mass term contribution: m^2 * sum |phi|^2
    double kinetic = energy;
    energy += field->mass * field->mass * kinetic;

    // Interaction term: lambda * sum |phi|^4
    double interaction = 0.0;
    for (size_t i = 0; i < total; i++) {
        double amp_sq = cabs(field->field_tensor->data[i]) * cabs(field->field_tensor->data[i]);
        interaction += amp_sq * amp_sq;
    }
    energy += field->coupling * interaction;

    // Normalize by lattice volume
    double lattice_vol = 1.0;
    for (int d = 0; d < QG_SPACETIME_DIMS; d++) {
        lattice_vol *= field->lattice_spacing[d];
    }

    return energy * lattice_vol / (double)total;
}

int calculate_field_equations_cpu(const QuantumField* field, Tensor* equations) {
    if (!field || !field->field_tensor || !equations || !equations->data) {
        return -1;
    }

    size_t total = equations->total_size;
    if (total != field->field_tensor->total_size) {
        total = (total < field->field_tensor->total_size) ? total : field->field_tensor->total_size;
    }

    // Calculate Klein-Gordon equation: (partial_mu partial^mu + m^2 + lambda|phi|^2) phi = 0
    size_t L = field->field_tensor->dims[0];
    size_t nc = field->num_components;

    for (size_t t = 0; t < L; t++) {
        for (size_t x = 0; x < L; x++) {
            for (size_t y = 0; y < L; y++) {
                for (size_t z = 0; z < L; z++) {
                    for (size_t c = 0; c < nc; c++) {
                        size_t idx = ((((t * L + x) * L + y) * L + z) * nc + c);
                        if (idx >= total) continue;

                        complex double phi = field->field_tensor->data[idx];
                        complex double laplacian = 0.0;

                        // 4D Laplacian using central differences
                        size_t coords[4] = {t, x, y, z};
                        for (int d = 0; d < 4; d++) {
                            size_t coord_p = (coords[d] + 1) % L;
                            size_t coord_m = (coords[d] + L - 1) % L;

                            size_t idx_coords[4];
                            memcpy(idx_coords, coords, 4 * sizeof(size_t));

                            idx_coords[d] = coord_p;
                            size_t idx_p = ((((idx_coords[0] * L + idx_coords[1]) * L +
                                             idx_coords[2]) * L + idx_coords[3]) * nc + c);

                            idx_coords[d] = coord_m;
                            size_t idx_m = ((((idx_coords[0] * L + idx_coords[1]) * L +
                                             idx_coords[2]) * L + idx_coords[3]) * nc + c);

                            if (idx_p < field->field_tensor->total_size &&
                                idx_m < field->field_tensor->total_size) {
                                complex double phi_p = field->field_tensor->data[idx_p];
                                complex double phi_m = field->field_tensor->data[idx_m];
                                double h = field->lattice_spacing[d];
                                double sign = (d == 0) ? -1.0 : 1.0;  // Minkowski metric
                                laplacian += sign * (phi_p + phi_m - 2.0 * phi) / (h * h);
                            }
                        }

                        // Field equation residual
                        double amp_sq = cabs(phi) * cabs(phi);
                        equations->data[idx] = laplacian +
                                              field->mass * field->mass * phi +
                                              field->coupling * amp_sq * phi;
                    }
                }
            }
        }
    }

    return 0;
}
