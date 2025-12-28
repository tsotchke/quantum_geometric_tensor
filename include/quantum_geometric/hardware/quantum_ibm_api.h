/**
 * @file quantum_ibm_api.h
 * @brief IBM Quantum API types and functions
 */

#ifndef QUANTUM_IBM_API_H
#define QUANTUM_IBM_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_result.h"
#include "quantum_geometric/physics/stabilizer_types.h"

// Forward declarations
struct quantum_circuit;
struct quantum_state;


// IBM backend status
typedef enum {
    IBM_STATUS_IDLE,
    IBM_STATUS_QUEUED,
    IBM_STATUS_RUNNING,
    IBM_STATUS_COMPLETED,
    IBM_STATUS_ERROR,
    IBM_STATUS_CANCELLED
} IBMJobStatus;

// IBM job result
typedef struct {
    uint64_t* counts;
    size_t num_counts;
    double* probabilities;
    double fidelity;
    double error_rate;
    IBMJobStatus status;
    char* error_message;
    void* raw_data;
} IBMJobResult;

// IBM backend configuration - may be defined in quantum_hardware_types.h
#ifndef IBM_BACKEND_CONFIG_DEFINED
#define IBM_BACKEND_CONFIG_DEFINED
typedef struct IBMBackendConfig {
    char* backend_name;
    char* hub;
    char* group;
    char* project;
    char* token;
    int optimization_level;
    bool error_mitigation;
    bool dynamic_decoupling;
    bool readout_error_mitigation;
    bool measurement_error_mitigation;
} IBMBackendConfig;
#endif

// Error correction graph structures
#ifndef MATCHING_VERTEX_DEFINED
#define MATCHING_VERTEX_DEFINED
typedef struct MatchingVertex {
    size_t id;
    double weight;
    size_t* neighbors;
    size_t num_neighbors;
} MatchingVertex;
#endif

#ifndef MATCHING_GRAPH_DEFINED
#define MATCHING_GRAPH_DEFINED
typedef struct MatchingGraph {
    MatchingVertex* vertices;
    size_t num_vertices;
} MatchingGraph;
#endif

// IBM-specific syndrome configuration (different from stabilizer_types.h SyndromeConfig)
typedef struct IBMSyndromeConfig {
    size_t num_qubits;
    size_t num_stabilizers;
    double threshold;
} IBMSyndromeConfig;

// IBM calibration data
typedef struct {
    double gate_errors[128];    // Per-qubit gate error rates
    double readout_errors[128]; // Per-qubit readout error rates
} IBMCalibrationData;

// Function declarations

// API initialization and connection
void* ibm_api_init(const char* token);
bool ibm_api_connect_backend(void* api_handle, const char* backend_name);
bool ibm_api_get_calibration(void* api_handle, IBMCalibrationData* cal_data);
bool ibm_api_init_job_queue(void* api_handle);

// Job management
char* ibm_api_submit_job(void* api_handle, const char* qasm);
IBMJobStatus ibm_api_get_job_status(void* api_handle, const char* job_id);
char* ibm_api_get_job_error(void* api_handle, const char* job_id);
IBMJobResult* ibm_api_get_job_result(void* api_handle, const char* job_id);
void ibm_api_cancel_pending_jobs(void* api_handle);

// Session management
void ibm_api_close_session(void* api_handle);
void ibm_api_clear_credentials(void* api_handle);
void ibm_api_destroy(void* api_handle);

// Result cleanup
void cleanup_ibm_result(IBMJobResult* result);

// ============================================================================
// IBM Backend Optimized Types
// ============================================================================

// Maximum number of qubits supported
#ifndef MAX_QUBITS
#define MAX_QUBITS 127
#endif

// IBM backend info structure for querying backend properties
typedef struct {
    size_t num_qubits;                // Number of qubits on the backend
    double* gate_errors;              // Per-qubit gate error rates
    double* readout_errors;           // Per-qubit readout error rates
    double* qubit_status;             // Qubit availability status (>0.5 = available)
    double* t1_times;                 // T1 relaxation times
    double* t2_times;                 // T2 coherence times
} ibm_backend_info;

// IBM coupling info for qubit pairs
typedef struct {
    size_t qubit1;                    // First qubit index
    size_t qubit2;                    // Second qubit index
    double strength;                  // Coupling strength
    double gate_error;                // Two-qubit gate error rate
    double gate_time;                 // Two-qubit gate time
} ibm_coupling_info;

// Feedback configuration
typedef struct {
    bool measurement_feedback;        // Enable measurement feedback
    bool conditional_ops;             // Enable conditional operations
    bool dynamic_decoupling;          // Enable dynamic decoupling
    double feedback_delay;            // Feedback loop delay
    size_t max_feedback_depth;        // Maximum feedback depth
} feedback_config;

// IBM feedback setup structure
typedef struct {
    bool measurement_feedback;        // Enable measurement feedback
    bool conditional_ops;             // Enable conditional operations
    bool dynamic_decoupling;          // Enable dynamic decoupling
    double* decoupling_sequence;      // Decoupling pulse sequence
    size_t sequence_length;           // Length of decoupling sequence
} ibm_feedback_setup;

// Parallel execution configuration
typedef struct {
    size_t max_parallel_gates;        // Maximum gates to execute in parallel
    size_t max_parallel_measurements; // Maximum measurements in parallel
    size_t* measurement_order;        // Order of measurements
    bool enable_gate_fusion;          // Enable gate fusion optimization
    bool enable_commutation;          // Enable gate commutation
} parallel_config;

// IBM parallel execution setup
typedef struct {
    size_t max_gates;                 // Maximum parallel gates
    size_t max_measurements;          // Maximum parallel measurements
    size_t* measurement_order;        // Measurement order
    double timing_constraints;        // Timing constraints in nanoseconds
} ibm_parallel_setup;

// IBM-specific gate structure for circuit optimization
typedef struct ibm_quantum_gate {
    gate_type_t type;                 // Gate type (GATE_X, GATE_H, etc.)
    size_t num_qubits;                // Number of qubits this gate operates on
    size_t* qubits;                   // Qubit indices this gate operates on
    size_t control;                   // Control qubit for controlled gates
    size_t target;                    // Target qubit for controlled gates
    double* params;                   // Gate parameters (rotation angles, etc.)
    size_t num_params;                // Number of parameters
    bool cancelled;                   // Whether gate has been cancelled by optimization
    double error_rate;                // Expected error rate for this gate
} ibm_quantum_gate;

// IBM-specific circuit structure for backend optimization
typedef struct ibm_quantum_circuit {
    ibm_quantum_gate* gates;          // Array of gates
    size_t num_gates;                 // Number of gates in circuit
    size_t capacity;                  // Allocated capacity for gates
    size_t num_qubits;                // Number of qubits in circuit
    size_t num_classical_bits;        // Number of classical bits
    double* initial_state;            // Initial state (if specified)
    char* name;                       // Circuit name
} ibm_quantum_circuit;

// Note: quantum_result is defined in quantum_result.h which is included via quantum_result.h

// ============================================================================
// IBM Backend Optimized Function Declarations
// ============================================================================

// Backend property querying
bool query_ibm_backend(const char* backend_name, ibm_backend_info* info);
bool query_ibm_coupling(const char* backend_name, size_t qubit1, size_t qubit2, ibm_coupling_info* coupling);
void cleanup_ibm_backend_info(ibm_backend_info* info);

// Configuration validation
bool validate_ibm_config(const IBMBackendConfig* config);

// Feedback and parallel execution
bool configure_ibm_feedback(const char* backend_name, const ibm_feedback_setup* setup);
bool execute_ibm_parallel(const char* backend_name, ibm_quantum_circuit* circuit,
                          IBMJobResult* result, const ibm_parallel_setup* setup);

// IBM circuit optimization functions
bool ibm_cancel_redundant_gates(ibm_quantum_circuit* circuit);
bool ibm_fuse_compatible_gates(ibm_quantum_circuit* circuit);
bool ibm_reorder_gates_parallel(ibm_quantum_circuit* circuit);
bool ibm_optimize_qubit_mapping(ibm_quantum_circuit* circuit, double** coupling_map, size_t num_qubits);
bool ibm_optimize_measurements(ibm_quantum_circuit* circuit, size_t* measurement_order, size_t num_qubits);
void ibm_optimize_measurement_order(size_t* measurement_order, double* readout_errors, size_t num_qubits);

// Fast feedback and parallel execution
bool configure_fast_feedback(const char* backend_name, ibm_quantum_circuit* circuit);
bool execute_parallel_circuit(const char* backend_name, ibm_quantum_circuit* circuit,
                              IBMJobResult* result, size_t* measurement_order, size_t num_qubits);

// Result processing
void ibm_process_measurement_results(IBMJobResult* result, double* readout_errors, size_t num_qubits);

// Error mitigation
bool ibm_mitigate_readout_errors(IBMJobResult* result, double* readout_errors, size_t num_qubits);
bool ibm_mitigate_measurement_errors(IBMJobResult* result, double* error_rates, size_t num_qubits);
bool ibm_extrapolate_zero_noise(IBMJobResult* result, double* error_rates, size_t num_qubits);

// Cleanup functions
void cleanup_ibm_quantum_gate(ibm_quantum_gate* gate);
void cleanup_ibm_quantum_circuit(ibm_quantum_circuit* circuit);
// Note: cleanup_quantum_result is defined in quantum_result.h

// Fallback status query functions
// These functions allow callers to check if local simulation is being used
// instead of real IBM Quantum hardware, and why.
bool ibm_api_is_using_fallback(void* api_handle);
const char* ibm_api_get_fallback_reason(void* api_handle);
void ibm_api_clear_fallback_status(void* api_handle);

#endif // QUANTUM_IBM_API_H
