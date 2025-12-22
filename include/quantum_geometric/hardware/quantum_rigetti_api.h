/**
 * @file quantum_rigetti_api.h
 * @brief Rigetti QCS API types and functions for quantum hardware access
 *
 * This module provides the interface to the Rigetti Quantum Cloud Services (QCS)
 * platform, enabling execution of Quil programs on Rigetti quantum processors
 * and simulators.
 */

#ifndef QUANTUM_RIGETTI_API_H
#define QUANTUM_RIGETTI_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// QCS Connection Types
// ============================================================================

/**
 * @brief Handle for QCS API connection
 */
typedef struct rigetti_qcs_handle rigetti_qcs_handle_t;

/**
 * @brief QCS execution targets
 */
typedef enum {
    QCS_TARGET_QPU,        /**< Real quantum processing unit */
    QCS_TARGET_QVM,        /**< Quantum Virtual Machine (simulator) */
    QCS_TARGET_QVM_NOISY   /**< QVM with noise model */
} qcs_target_type_t;

/**
 * @brief Job status on QCS
 */
typedef enum {
    QCS_JOB_PENDING,
    QCS_JOB_QUEUED,
    QCS_JOB_RUNNING,
    QCS_JOB_COMPLETED,
    QCS_JOB_FAILED,
    QCS_JOB_CANCELLED,
    QCS_JOB_UNKNOWN
} qcs_job_status_t;

// ============================================================================
// QCS Configuration
// ============================================================================

/**
 * @brief QCS authentication configuration
 */
typedef struct {
    const char* api_key;           /**< QCS API key */
    const char* user_id;           /**< QCS user ID */
    const char* qcs_url;           /**< QCS endpoint URL (NULL for default) */
    const char* quilc_url;         /**< Quil compiler URL (NULL for default) */
    const char* qvm_url;           /**< QVM URL (NULL for default) */
    bool use_client_configuration; /**< Use ~/.qcs/settings.toml */
} qcs_auth_config_t;

/**
 * @brief QPU execution options
 */
typedef struct {
    qcs_target_type_t target;
    const char* qpu_name;          /**< QPU name (e.g., "Aspen-M-3") */
    size_t shots;                  /**< Number of measurement shots */
    bool use_quilc;                /**< Compile with quilc */
    bool use_parametric;           /**< Use parametric compilation */
    bool use_active_reset;         /**< Use active qubit reset */
    int timeout_seconds;           /**< Job timeout (0 = default) */
} qcs_execution_options_t;

/**
 * @brief QPU calibration data
 */
typedef struct {
    size_t num_qubits;
    double* t1_times;              /**< T1 relaxation times per qubit */
    double* t2_times;              /**< T2 dephasing times per qubit */
    double* readout_fidelities;    /**< Readout fidelity per qubit */
    double* gate_fidelities;       /**< Gate fidelity per qubit */
    double** cz_fidelities;        /**< CZ gate fidelity matrix */
    size_t* qubit_ids;             /**< Physical qubit IDs */
    bool* qubit_dead;              /**< Dead qubit flags */
    int64_t timestamp;             /**< Calibration timestamp */
} qcs_calibration_data_t;

/**
 * @brief Job result from QCS execution
 */
typedef struct {
    uint8_t** bitstrings;          /**< Raw measurement bitstrings [shots][qubits] */
    size_t num_shots;              /**< Number of shots executed */
    size_t num_qubits;             /**< Number of qubits measured */
    uint64_t* counts;              /**< Histogram of measurement outcomes */
    size_t num_outcomes;           /**< Number of unique outcomes */
    double* probabilities;         /**< Probability distribution */
    double execution_time;         /**< Total execution time (seconds) */
    double compile_time;           /**< Quilc compilation time (seconds) */
    qcs_job_status_t status;       /**< Final job status */
    char* error_message;           /**< Error message if failed */
    void* metadata;                /**< Additional metadata (JSON) */
} qcs_job_result_t;

// ============================================================================
// QCS Connection Functions
// ============================================================================

/**
 * @brief Initialize connection to Rigetti QCS
 *
 * @param config Authentication configuration
 * @return Handle to QCS connection, or NULL on failure
 */
rigetti_qcs_handle_t* qcs_connect(const qcs_auth_config_t* config);

/**
 * @brief Initialize connection using default configuration files
 *
 * Uses ~/.qcs/settings.toml and ~/.qcs/secrets.toml
 *
 * @return Handle to QCS connection, or NULL on failure
 */
rigetti_qcs_handle_t* qcs_connect_default(void);

/**
 * @brief Close QCS connection and release resources
 *
 * @param handle QCS connection handle
 */
void qcs_disconnect(rigetti_qcs_handle_t* handle);

/**
 * @brief Test QCS connection
 *
 * @param handle QCS connection handle
 * @return true if connection is active, false otherwise
 */
bool qcs_test_connection(rigetti_qcs_handle_t* handle);

// ============================================================================
// QPU Information Functions
// ============================================================================

/**
 * @brief Get list of available QPUs
 *
 * @param handle QCS connection handle
 * @param qpu_names Output array of QPU names (caller must free)
 * @param num_qpus Output number of QPUs
 * @return true on success, false on failure
 */
bool qcs_list_qpus(rigetti_qcs_handle_t* handle, char*** qpu_names, size_t* num_qpus);

/**
 * @brief Get calibration data for a QPU
 *
 * @param handle QCS connection handle
 * @param qpu_name Name of the QPU
 * @param cal_data Output calibration data (caller must free with qcs_free_calibration)
 * @return true on success, false on failure
 */
bool qcs_get_calibration(rigetti_qcs_handle_t* handle, const char* qpu_name,
                         qcs_calibration_data_t* cal_data);

/**
 * @brief Get instruction set architecture for a QPU
 *
 * @param handle QCS connection handle
 * @param qpu_name Name of the QPU
 * @return ISA as Quil string, or NULL on failure (caller must free)
 */
char* qcs_get_isa(rigetti_qcs_handle_t* handle, const char* qpu_name);

/**
 * @brief Get qubit connectivity map
 *
 * @param handle QCS connection handle
 * @param qpu_name Name of the QPU
 * @param edges Output array of [qubit1, qubit2] pairs (caller must free)
 * @param num_edges Output number of edges
 * @return true on success, false on failure
 */
bool qcs_get_connectivity(rigetti_qcs_handle_t* handle, const char* qpu_name,
                          size_t** edges, size_t* num_edges);

// ============================================================================
// Program Compilation Functions
// ============================================================================

/**
 * @brief Compile Quil program with quilc
 *
 * @param handle QCS connection handle
 * @param quil_program Input Quil program
 * @param qpu_name Target QPU name
 * @param compiled_program Output compiled program (caller must free)
 * @return true on success, false on failure
 */
bool qcs_compile_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const char* qpu_name, char** compiled_program);

/**
 * @brief Get native Quil for a program (compiled for specific QPU)
 *
 * @param handle QCS connection handle
 * @param quil_program Input Quil program
 * @param qpu_name Target QPU name
 * @return Native Quil string, or NULL on failure (caller must free)
 */
char* qcs_get_native_quil(rigetti_qcs_handle_t* handle, const char* quil_program,
                          const char* qpu_name);

// ============================================================================
// Program Execution Functions
// ============================================================================

/**
 * @brief Submit Quil program for execution
 *
 * @param handle QCS connection handle
 * @param quil_program Quil program to execute
 * @param options Execution options
 * @return Job ID string, or NULL on failure (caller must free)
 */
char* qcs_submit_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const qcs_execution_options_t* options);

/**
 * @brief Execute Quil program and wait for results (blocking)
 *
 * @param handle QCS connection handle
 * @param quil_program Quil program to execute
 * @param options Execution options
 * @param result Output result structure (caller must free with qcs_free_result)
 * @return true on success, false on failure
 */
bool qcs_execute_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const qcs_execution_options_t* options, qcs_job_result_t* result);

/**
 * @brief Execute on QVM (Quantum Virtual Machine) locally
 *
 * @param handle QCS connection handle
 * @param quil_program Quil program to execute
 * @param num_shots Number of measurement shots
 * @param result Output result structure
 * @return true on success, false on failure
 */
bool qcs_execute_on_qvm(rigetti_qcs_handle_t* handle, const char* quil_program,
                        size_t num_shots, qcs_job_result_t* result);

/**
 * @brief Get status of a submitted job
 *
 * @param handle QCS connection handle
 * @param job_id Job ID
 * @return Job status
 */
qcs_job_status_t qcs_get_job_status(rigetti_qcs_handle_t* handle, const char* job_id);

/**
 * @brief Get result of a completed job
 *
 * @param handle QCS connection handle
 * @param job_id Job ID
 * @param result Output result structure
 * @return true on success, false on failure
 */
bool qcs_get_job_result(rigetti_qcs_handle_t* handle, const char* job_id,
                        qcs_job_result_t* result);

/**
 * @brief Cancel a submitted job
 *
 * @param handle QCS connection handle
 * @param job_id Job ID
 * @return true if cancellation was successful
 */
bool qcs_cancel_job(rigetti_qcs_handle_t* handle, const char* job_id);

/**
 * @brief Wait for job completion with timeout
 *
 * @param handle QCS connection handle
 * @param job_id Job ID
 * @param timeout_seconds Maximum wait time (0 = no timeout)
 * @return Final job status
 */
qcs_job_status_t qcs_wait_for_job(rigetti_qcs_handle_t* handle, const char* job_id,
                                   int timeout_seconds);

// ============================================================================
// Parametric Compilation Functions
// ============================================================================

/**
 * @brief Create a parametric compiled program
 *
 * @param handle QCS connection handle
 * @param quil_program Quil program with parameters (e.g., "RX(%theta) 0")
 * @param qpu_name Target QPU name
 * @return Parametric program handle, or NULL on failure
 */
void* qcs_create_parametric_program(rigetti_qcs_handle_t* handle,
                                    const char* quil_program, const char* qpu_name);

/**
 * @brief Execute parametric program with specific values
 *
 * @param handle QCS connection handle
 * @param parametric_program Parametric program handle
 * @param param_names Array of parameter names
 * @param param_values Array of parameter values
 * @param num_params Number of parameters
 * @param shots Number of shots
 * @param result Output result
 * @return true on success
 */
bool qcs_execute_parametric(rigetti_qcs_handle_t* handle, void* parametric_program,
                            const char** param_names, const double* param_values,
                            size_t num_params, size_t shots, qcs_job_result_t* result);

/**
 * @brief Free parametric program resources
 */
void qcs_free_parametric_program(void* parametric_program);

// ============================================================================
// Memory Management Functions
// ============================================================================

/**
 * @brief Free calibration data structure
 */
void qcs_free_calibration(qcs_calibration_data_t* cal_data);

/**
 * @brief Free job result structure
 */
void qcs_free_result(qcs_job_result_t* result);

/**
 * @brief Free QPU names array
 */
void qcs_free_qpu_names(char** names, size_t num);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * @brief Get last error message
 *
 * @param handle QCS connection handle
 * @return Error message string (do not free)
 */
const char* qcs_get_last_error(rigetti_qcs_handle_t* handle);

/**
 * @brief Get last error code
 *
 * @param handle QCS connection handle
 * @return Error code (0 = no error)
 */
int qcs_get_last_error_code(rigetti_qcs_handle_t* handle);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_RIGETTI_API_H
