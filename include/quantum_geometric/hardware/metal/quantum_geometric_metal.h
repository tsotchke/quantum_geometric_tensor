#ifndef QUANTUM_GEOMETRIC_METAL_H
#define QUANTUM_GEOMETRIC_METAL_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stddef.h>
#include <stdbool.h>

// C/C++ complex type compatibility
#ifndef __cplusplus
    #include <complex.h>
#endif

// Type aliases for Metal backend (only if not already defined)
#ifndef QGT_CONTEXT_TYPE_DEFINED
#define QGT_CONTEXT_TYPE_DEFINED
typedef quantum_geometric_hardware_t qgt_context_t;
#endif

#ifndef QGT_STATE_TYPE_DEFINED
#define QGT_STATE_TYPE_DEFINED
typedef quantum_state_t qgt_state_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Metal Device Management
// ===========================================================================

/**
 * Initialize Metal backend
 * @param ctx Quantum geometric context
 * @return Error code
 */
qgt_error_t qgt_metal_initialize(qgt_context_t* ctx);

/**
 * Cleanup Metal backend
 * @param ctx Quantum geometric context
 * @return Error code
 */
qgt_error_t qgt_metal_cleanup(qgt_context_t* ctx);

/**
 * Check if Metal is available
 * @return true if Metal is available, false otherwise
 */
bool qgt_metal_is_available(void);

/**
 * Get Metal device information
 * @param ctx Quantum geometric context
 * @param device_name Buffer for device name (output)
 * @param device_name_size Size of device name buffer
 * @param total_memory Pointer to store total memory in bytes (output)
 * @return Error code
 */
qgt_error_t qgt_metal_get_device_info(qgt_context_t* ctx,
                                     char* device_name,
                                     size_t device_name_size,
                                     size_t* total_memory);

// ===========================================================================
// Metal Memory Management
// ===========================================================================

/**
 * Allocate memory on Metal device
 * @param ptr Pointer to store allocated memory address (output)
 * @param size Size in bytes to allocate
 * @return Error code
 */
qgt_error_t qgt_metal_malloc(void** ptr, size_t size);

/**
 * Free memory on Metal device
 * @param ptr Pointer to free
 * @return Error code
 */
qgt_error_t qgt_metal_free(void* ptr);

/**
 * Copy memory from host to Metal device
 * @param dst Destination pointer on device
 * @param src Source pointer on host
 * @param size Number of bytes to copy
 * @return Error code
 */
qgt_error_t qgt_metal_memcpy_host_to_device(void* dst, const void* src, size_t size);

/**
 * Copy memory from Metal device to host
 * @param dst Destination pointer on host
 * @param src Source pointer on device
 * @param size Number of bytes to copy
 * @return Error code
 */
qgt_error_t qgt_metal_memcpy_device_to_host(void* dst, const void* src, size_t size);

/**
 * Copy memory between Metal device locations
 * @param dst Destination pointer on device
 * @param src Source pointer on device
 * @param size Number of bytes to copy
 * @return Error code
 */
qgt_error_t qgt_metal_memcpy_device_to_device(void* dst, const void* src, size_t size);

/**
 * Free Metal memory allocated from a pool
 * @param pool Memory pool
 * @param ptr Pointer to free
 */
void qgt_metal_free_pooled(MemoryPool* pool, void* ptr);

// ===========================================================================
// Metal State Operations
// ===========================================================================

/**
 * Transfer quantum state to Metal device
 * @param ctx Quantum geometric context
 * @param state Quantum state to transfer
 * @return Error code
 */
qgt_error_t qgt_metal_transfer_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * Retrieve quantum state from Metal device
 * @param ctx Quantum geometric context
 * @param state Quantum state to retrieve
 * @return Error code
 */
qgt_error_t qgt_metal_retrieve_state(qgt_context_t* ctx, qgt_state_t* state);

/**
 * Synchronize Metal operations
 * @param ctx Quantum geometric context
 * @return Error code
 */
qgt_error_t qgt_metal_sync(qgt_context_t* ctx);

/**
 * Create quantum state on Metal device
 * @param ctx Quantum geometric context
 * @param num_qubits Number of qubits
 * @param state Pointer to store created state (output)
 * @return Error code
 */
qgt_error_t qgt_metal_create_state(qgt_context_t* ctx, size_t num_qubits, qgt_state_t** state);

/**
 * Destroy quantum state on Metal device
 * @param ctx Quantum geometric context
 * @param state Quantum state to destroy
 * @return Error code
 */
qgt_error_t qgt_metal_destroy_state(qgt_context_t* ctx, qgt_state_t* state);

// ===========================================================================
// Metal Geometric Operations
// ===========================================================================

/**
 * Perform geometric rotation on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param angle Rotation angle in radians
 * @param axis Rotation axis (3D vector)
 * @return Error code
 */
qgt_error_t qgt_metal_geometric_rotate(qgt_context_t* ctx,
                                      qgt_state_t* state,
                                      double angle,
                                      const double* axis);

/**
 * Perform parallel transport along a path on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param path Path to transport along (3D coordinates)
 * @param num_points Number of points in path
 * @return Error code
 */
qgt_error_t qgt_metal_geometric_parallel_transport(qgt_context_t* ctx,
                                                   qgt_state_t* state,
                                                   const double* path,
                                                   size_t num_points);

/**
 * Compute metric tensor on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param metric Output metric tensor
 * @return Error code
 */
qgt_error_t qgt_metal_compute_metric(qgt_context_t* ctx,
                                    qgt_state_t* state,
                                    double* metric);

/**
 * Compute connection coefficients on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param connection Output connection coefficients
 * @return Error code
 */
qgt_error_t qgt_metal_compute_connection(qgt_context_t* ctx,
                                        qgt_state_t* state,
                                        double* connection);

/**
 * Compute curvature tensor on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param curvature Output curvature tensor
 * @return Error code
 */
qgt_error_t qgt_metal_compute_curvature(qgt_context_t* ctx,
                                       qgt_state_t* state,
                                       double* curvature);

// ===========================================================================
// Metal Quantum Operations
// ===========================================================================

/**
 * Apply quantum gate on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param gate_matrix Gate matrix
 * @param gate_size Size of gate matrix
 * @param target_qubits Target qubit indices
 * @param num_targets Number of target qubits
 * @return Error code
 */
qgt_error_t qgt_metal_apply_gate(qgt_context_t* ctx,
                                 qgt_state_t* state,
                                 const double _Complex* gate_matrix,
                                 size_t gate_size,
                                 const size_t* target_qubits,
                                 size_t num_targets);

/**
 * Apply controlled gate on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param gate_matrix Gate matrix
 * @param gate_size Size of gate matrix
 * @param control_qubits Control qubit indices
 * @param num_controls Number of control qubits
 * @param target_qubits Target qubit indices
 * @param num_targets Number of target qubits
 * @return Error code
 */
qgt_error_t qgt_metal_apply_controlled_gate(qgt_context_t* ctx,
                                           qgt_state_t* state,
                                           const double _Complex* gate_matrix,
                                           size_t gate_size,
                                           const size_t* control_qubits,
                                           size_t num_controls,
                                           const size_t* target_qubits,
                                           size_t num_targets);

/**
 * Measure quantum state on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param qubit Qubit index to measure
 * @param result Measurement result (output)
 * @return Error code
 */
qgt_error_t qgt_metal_measure(qgt_context_t* ctx,
                             qgt_state_t* state,
                             size_t qubit,
                             int* result);

/**
 * Measure all qubits on Metal
 * @param ctx Quantum geometric context
 * @param state Quantum state
 * @param results Array to store measurement results (output)
 * @param num_results Size of results array
 * @return Error code
 */
qgt_error_t qgt_metal_measure_all(qgt_context_t* ctx,
                                  qgt_state_t* state,
                                  int* results,
                                  size_t num_results);

// ===========================================================================
// Metal Tensor Operations
// ===========================================================================

/**
 * Compute tensor contraction on Metal
 * @param ctx Quantum geometric context
 * @param result Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @param dims_a Dimensions of first tensor
 * @param dims_b Dimensions of second tensor
 * @param contract_indices Indices to contract
 * @param num_contract Number of indices to contract
 * @return Error code
 */
qgt_error_t qgt_metal_tensor_contract(qgt_context_t* ctx,
                                     double _Complex* result,
                                     const double _Complex* a,
                                     const double _Complex* b,
                                     const size_t* dims_a,
                                     const size_t* dims_b,
                                     const size_t* contract_indices,
                                     size_t num_contract);

/**
 * Compute matrix-vector multiplication on Metal
 * @param ctx Quantum geometric context
 * @param result Output vector
 * @param matrix Input matrix
 * @param vector Input vector
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Error code
 */
qgt_error_t qgt_metal_matvec(qgt_context_t* ctx,
                             double _Complex* result,
                             const double _Complex* matrix,
                             const double _Complex* vector,
                             size_t rows,
                             size_t cols);

/**
 * Compute matrix-matrix multiplication on Metal
 * @param ctx Quantum geometric context
 * @param result Output matrix
 * @param a First input matrix
 * @param b Second input matrix
 * @param m Number of rows in a
 * @param n Number of columns in a / rows in b
 * @param k Number of columns in b
 * @return Error code
 */
qgt_error_t qgt_metal_matmul(qgt_context_t* ctx,
                             double _Complex* result,
                             const double _Complex* a,
                             const double _Complex* b,
                             size_t m,
                             size_t n,
                             size_t k);

// ===========================================================================
// Metal Performance Optimization
// ===========================================================================

/**
 * Set Metal thread group size
 * @param ctx Quantum geometric context
 * @param width Thread group width
 * @param height Thread group height
 * @param depth Thread group depth
 * @return Error code
 */
qgt_error_t qgt_metal_set_thread_group_size(qgt_context_t* ctx,
                                            size_t width,
                                            size_t height,
                                            size_t depth);

/**
 * Enable/disable Metal fast math
 * @param ctx Quantum geometric context
 * @param enable true to enable, false to disable
 * @return Error code
 */
qgt_error_t qgt_metal_set_fast_math(qgt_context_t* ctx, bool enable);

/**
 * Get Metal performance statistics
 * @param ctx Quantum geometric context
 * @param total_ops Total operations executed (output)
 * @param total_time Total execution time in seconds (output)
 * @param throughput Operations per second (output)
 * @return Error code
 */
qgt_error_t qgt_metal_get_performance_stats(qgt_context_t* ctx,
                                           size_t* total_ops,
                                           double* total_time,
                                           double* throughput);

// ===========================================================================
// Metal AMX (Apple Matrix Coprocessor) Support
// ===========================================================================

/**
 * Check if AMX is available
 * @return true if AMX is available, false otherwise
 */
bool qgt_metal_amx_is_available(void);

/**
 * Initialize AMX for quantum operations
 * @param ctx Quantum geometric context
 * @return Error code
 */
qgt_error_t qgt_metal_amx_initialize(qgt_context_t* ctx);

/**
 * Perform matrix multiplication using AMX
 * @param ctx Quantum geometric context
 * @param result Output matrix
 * @param a First input matrix
 * @param b Second input matrix
 * @param m Number of rows in a
 * @param n Number of columns in a / rows in b
 * @param k Number of columns in b
 * @return Error code
 */
qgt_error_t qgt_metal_amx_matmul(qgt_context_t* ctx,
                                 float* result,
                                 const float* a,
                                 const float* b,
                                 size_t m,
                                 size_t n,
                                 size_t k);

// ===========================================================================
// Metal Error Codes
// ===========================================================================

#define QGT_ERROR_METAL_NOT_AVAILABLE     -200
#define QGT_ERROR_METAL_DEVICE_INIT      -201
#define QGT_ERROR_METAL_OUT_OF_MEMORY    -202
#define QGT_ERROR_METAL_INVALID_VALUE    -203
#define QGT_ERROR_METAL_LAUNCH_FAILED    -204
#define QGT_ERROR_METAL_SYNC_FAILED      -205
#define QGT_ERROR_METAL_INTERNAL         -206
#define QGT_ERROR_METAL_SHADER_COMPILE   -207
#define QGT_ERROR_METAL_PIPELINE_CREATE  -208
#define QGT_ERROR_METAL_BUFFER_CREATE    -209
#define QGT_ERROR_METAL_COMMAND_ENCODE   -210
#define QGT_ERROR_METAL_AMX_NOT_AVAILABLE -211

// ===========================================================================
// Metal Stabilizer Types and Operations
// ===========================================================================

/**
 * @brief 2D float vector for GPU operations
 */
typedef struct float2 {
    float x;
    float y;
} float2;

/**
 * @brief 4D float vector for GPU operations
 */
typedef struct float4 {
    float x;
    float y;
    float z;
    float w;
} float4;

/**
 * @brief Stabilizer qubit representation for error correction
 */
typedef struct StabilizerQubit {
    float2 amplitude;           // Complex amplitude (x=real, y=imag)
    float error_rate;           // Per-qubit error rate
    uint32_t flags;             // Status flags (measured, corrected, etc.)
    uint32_t index;             // Qubit index in lattice
    float phase;                // Phase angle
    float coherence;            // Coherence factor
} StabilizerQubit;

/**
 * @brief Stabilizer measurement configuration (Metal GPU version)
 */
#ifndef METAL_STABILIZER_CONFIG_DEFINED
#define METAL_STABILIZER_CONFIG_DEFINED
typedef struct MetalStabilizerConfig {
    uint32_t type;              // Stabilizer type: 1=X, 2=Z, 3=XZ
    uint32_t num_qubits;        // Number of qubits per stabilizer
    float weight;               // Measurement weight
    float confidence;           // Confidence threshold
    float error_threshold;      // Error detection threshold
    uint32_t measurement_rounds;// Number of measurement rounds
} MetalStabilizerConfig;
// Alias for backward compatibility in Metal context
typedef MetalStabilizerConfig StabilizerConfig;
#endif // METAL_STABILIZER_CONFIG_DEFINED

/**
 * @brief Stabilizer measurement result
 */
typedef struct StabilizerResult {
    float2* syndromes;          // Syndrome measurements
    uint32_t* error_locations;  // Detected error locations
    size_t num_errors;          // Number of detected errors
    float total_fidelity;       // Overall measurement fidelity
    double execution_time;      // Measurement time in seconds
} StabilizerResult;

/**
 * @brief Error correction configuration
 */
typedef struct CorrectionConfig {
    uint32_t decoder_type;      // 0=MWPM, 1=Union-Find, 2=Neural
    uint32_t max_iterations;    // Maximum decoder iterations
    float success_threshold;    // Correction success threshold
    bool adaptive;              // Use adaptive correction
} CorrectionConfig;

// ===========================================================================
// Metal Stabilizer Functions (Simplified API)
// ===========================================================================

/**
 * Initialize Metal backend (simplified)
 * @return 0 on success, error code on failure
 */
int metal_initialize(void);

/**
 * Create Metal context
 * @param device_id Metal device ID (0 for default)
 * @return Context pointer or NULL on failure
 */
void* metal_create_context(int device_id);

/**
 * Destroy Metal context
 * @param ctx Context to destroy
 */
void metal_destroy_context(void* ctx);

/**
 * Cleanup Metal backend
 */
void metal_cleanup(void);

/**
 * Measure stabilizers using Metal acceleration
 * @param ctx Metal context
 * @param qubits Array of stabilizer qubits
 * @param indices Stabilizer qubit indices
 * @param config Measurement configuration
 * @param results Output results array
 * @param num_stabilizers Number of stabilizers to measure
 * @return 0 on success, error code on failure
 */
int metal_measure_stabilizers(void* ctx,
                              const StabilizerQubit* qubits,
                              const uint32_t* indices,
                              const StabilizerConfig* config,
                              float2* results,
                              size_t num_stabilizers);

/**
 * Apply error correction based on syndrome measurements
 * @param ctx Metal context
 * @param qubits Array of stabilizer qubits (modified in place)
 * @param syndromes Syndrome measurement results
 * @param num_syndromes Number of syndromes
 * @param config Correction configuration
 * @return 0 on success, error code on failure
 */
int metal_apply_correction(void* ctx,
                           StabilizerQubit* qubits,
                           const float2* syndromes,
                           size_t num_syndromes,
                           const CorrectionConfig* config);

/**
 * Compute error correlations between stabilizers
 * @param ctx Metal context
 * @param qubits Array of stabilizer qubits
 * @param indices Stabilizer qubit indices
 * @param num_stabilizers Number of stabilizers
 * @param correlations Output correlation matrix
 * @return 0 on success, error code on failure
 */
int metal_compute_correlations(void* ctx,
                               const StabilizerQubit* qubits,
                               const uint32_t* indices,
                               size_t num_stabilizers,
                               float* correlations);

/**
 * Detect errors using syndrome measurements
 * @param ctx Metal context
 * @param syndromes Syndrome measurements
 * @param num_syndromes Number of syndromes
 * @param error_locations Output array of error locations
 * @param max_errors Maximum errors to detect
 * @return Number of errors detected
 */
size_t metal_detect_errors(void* ctx,
                           const float2* syndromes,
                           size_t num_syndromes,
                           uint32_t* error_locations,
                           size_t max_errors);

/**
 * Perform minimum weight perfect matching for decoding
 * @param ctx Metal context
 * @param syndromes Syndrome measurements
 * @param num_syndromes Number of syndromes
 * @param weights Edge weights for matching graph
 * @param matching Output matching result
 * @return 0 on success, error code on failure
 */
int metal_mwpm_decode(void* ctx,
                      const float2* syndromes,
                      size_t num_syndromes,
                      const float* weights,
                      uint32_t* matching);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_METAL_H
