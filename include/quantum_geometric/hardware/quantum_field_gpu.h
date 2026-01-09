#ifndef QUANTUM_FIELD_GPU_H
#define QUANTUM_FIELD_GPU_H

/**
 * @file quantum_field_gpu.h
 * @brief GPU monitoring and error handling for quantum field operations
 *
 * This header provides error codes, performance monitoring types, and
 * utility functions for GPU-accelerated quantum field computations.
 * It bridges the quantum field subsystem with the core GPU backend.
 */

#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GPU Error Codes
// ============================================================================

/**
 * GPU operation error codes for detailed error reporting.
 * Negative values indicate errors, 0 indicates success.
 */
typedef enum {
    GPU_SUCCESS = 0,              ///< Operation completed successfully
    GPU_ERROR_NO_DEVICE = -1,     ///< No GPU device available
    GPU_ERROR_INIT_FAILED = -2,   ///< GPU initialization failed
    GPU_ERROR_INVALID_ARG = -3,   ///< Invalid argument passed to function
    GPU_ERROR_OUT_OF_MEM = -4,    ///< Out of GPU memory
    GPU_ERROR_LAUNCH_FAILED = -5  ///< Kernel/shader launch failed
} GpuErrorCode;

// ============================================================================
// Type Aliases for Compatibility
// ============================================================================

/**
 * Alias for GPUBackendType to maintain naming consistency
 * across different subsystems
 */
typedef GPUBackendType GpuBackendType;

// ============================================================================
// GPU State Management
// ============================================================================

/**
 * @brief Get the current GPU backend type
 *
 * Returns the type of GPU backend currently in use (Metal, CUDA, or None).
 * This queries the global GPU state to determine which backend is active.
 *
 * @return GpuBackendType The active GPU backend type
 */
GpuBackendType get_gpu_backend_type(void);

/**
 * @brief Check if GPU is available and initialized
 *
 * @return true if GPU is ready for use, false otherwise
 */
bool is_gpu_available(void);

// ============================================================================
// Error Handling Functions
// ============================================================================

/**
 * @brief Get human-readable string for GPU error code
 *
 * @param error The error code to translate
 * @return const char* Human-readable error description
 */
const char* gpu_error_string(int error);

/**
 * @brief Get the last GPU error code
 *
 * Returns the error code from the most recent GPU operation that failed.
 *
 * @return int The last error code (GPU_SUCCESS if no error)
 */
int get_last_gpu_error(void);

/**
 * @brief Clear the last GPU error
 *
 * Resets the error state to GPU_SUCCESS.
 */
void clear_gpu_error(void);

// ============================================================================
// Performance Monitoring Functions
// ============================================================================

/**
 * @brief Get current GPU memory usage in bytes
 *
 * Queries the GPU for current memory allocation. Only available
 * on CUDA backends with NVML support or Metal backends.
 *
 * @return size_t Memory usage in bytes, or 0 if unavailable
 */
size_t get_gpu_memory_usage(void);

/**
 * @brief Get current GPU utilization percentage
 *
 * Queries GPU compute utilization. Requires NVML on CUDA backends.
 * Not available on Metal backends.
 *
 * @return int Utilization percentage (0-100), or -1 if unavailable
 */
int get_gpu_utilization(void);

/**
 * @brief Get current GPU temperature in Celsius
 *
 * Queries GPU temperature. Requires NVML on CUDA backends.
 * Not available on Metal backends.
 *
 * @return int Temperature in Celsius, or -1 if unavailable
 */
int get_gpu_temperature(void);

/**
 * @brief Get current GPU power usage in watts
 *
 * Queries GPU power consumption. Requires NVML on CUDA backends.
 * Not available on Metal backends.
 *
 * @return float Power usage in watts, or -1.0 if unavailable
 */
float get_gpu_power_usage(void);

// ============================================================================
// Internal Error Reporting (for GPU backend implementations)
// ============================================================================

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>

/**
 * @brief Report CUDA error and set last error state
 *
 * @param error The CUDA error code
 * @param operation Description of the operation that failed
 */
void report_cuda_error(cudaError_t error, const char* operation);
#endif

#ifdef HAVE_METAL
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/**
 * @brief Report Metal error and set last error state
 *
 * @param error The NSError from Metal operation
 * @param operation Description of the operation that failed
 */
void report_metal_error(NSError* error, const char* operation);
#endif
#endif

// ============================================================================
// Spacetime Dimensions (for test compatibility)
// ============================================================================

#ifndef SPACETIME_DIMS
#define SPACETIME_DIMS 4
#endif

// ============================================================================
// Include QuantumField and Tensor definitions
// ============================================================================

#include "quantum_geometric/physics/quantum_field_calculations.h"

// ============================================================================
// Field Configuration for Test Compatibility
// ============================================================================

/**
 * @brief Field configuration for initializing quantum fields
 */
typedef struct {
    size_t lattice_size;        /**< Size of each lattice dimension */
    size_t num_components;      /**< Number of field components */
    size_t num_generators;      /**< Number of gauge generators */
    double mass;                /**< Field mass */
    double coupling;            /**< Self-coupling constant */
    double field_strength;      /**< Gauge field strength */
    bool gauge_group;           /**< Whether to include gauge field */
} FieldConfig;

/**
 * @brief Geometric configuration for spacetime
 */
typedef struct {
    double* metric;             /**< Spacetime metric tensor */
    double* connection;         /**< Christoffel connection */
    double* curvature;          /**< Riemann curvature tensor */
} GeometricConfig;

// ============================================================================
// GPU Acceleration Detection
// ============================================================================

/**
 * @brief Check if GPU acceleration is available
 * @return true if GPU is available, false otherwise
 */
bool has_gpu_acceleration(void);

/**
 * @brief Get GPU device name
 * @return Device name string
 */
const char* get_gpu_device_name(void);

/**
 * @brief Cleanup GPU backend resources
 */
void cleanup_gpu_backend(void);

// ============================================================================
// Field Operations (GPU and CPU versions)
// ============================================================================

/**
 * @brief Initialize quantum field for GPU operations
 * @param config Field configuration
 * @param geom Geometric configuration (may be NULL)
 * @return Initialized quantum field or NULL on failure
 */
QuantumField* init_quantum_field(const FieldConfig* config, const GeometricConfig* geom);

/**
 * @brief Cleanup quantum field with GPU-specific resources
 *
 * This is the GPU-extended version that cleans up additional GPU resources
 * allocated by init_quantum_field. For the base cleanup function, use
 * cleanup_quantum_field from quantum_field_calculations.h.
 *
 * @param field Field to cleanup (must have been created with init_quantum_field from this module)
 */
void cleanup_quantum_field_gpu(QuantumField* field);

/**
 * @brief Apply rotation to field qubit (CPU version)
 * @param field Quantum field
 * @param qubit Qubit index
 * @param theta Rotation angle theta
 * @param phi Rotation angle phi
 * @return 0 on success
 */
int apply_rotation_cpu(QuantumField* field, size_t qubit, double theta, double phi);

/**
 * @brief Apply rotation to field qubit (GPU version)
 * @param field Quantum field
 * @param qubit Qubit index
 * @param theta Rotation angle theta
 * @param phi Rotation angle phi
 * @return 0 on success
 */
int apply_rotation_gpu(QuantumField* field, size_t qubit, double theta, double phi);

/**
 * @brief Calculate field energy (CPU version)
 * @param field Quantum field
 * @return Field energy
 */
double calculate_field_energy_cpu(const QuantumField* field);

/**
 * @brief Calculate field energy (GPU version)
 * @param field Quantum field
 * @return Field energy
 */
double calculate_field_energy_gpu(const QuantumField* field);

/**
 * @brief Calculate field equations (CPU version)
 * @param field Quantum field
 * @param equations Output tensor for equations
 * @return 0 on success
 */
int calculate_field_equations_cpu(const QuantumField* field, Tensor* equations);

/**
 * @brief Calculate field equations (GPU version)
 * @param field Quantum field
 * @param equations Output tensor for equations
 * @return 0 on success
 */
int calculate_field_equations_gpu(const QuantumField* field, Tensor* equations);

// ============================================================================
// Tensor Operations for Tests
// ============================================================================

/**
 * @brief Initialize tensor with given dimensions
 * @param dims Array of dimensions
 * @param rank Number of dimensions
 * @return Allocated tensor or NULL on failure
 */
Tensor* init_tensor(const size_t* dims, size_t rank);

/**
 * @brief Cleanup tensor
 * @param tensor Tensor to cleanup
 */
void cleanup_tensor(Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_FIELD_GPU_H
