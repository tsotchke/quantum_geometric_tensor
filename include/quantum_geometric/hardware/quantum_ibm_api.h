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

#endif // QUANTUM_IBM_API_H
