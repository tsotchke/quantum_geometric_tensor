/**
 * @file test_integration_api.h
 * @brief API layer for integration tests - provides simplified interfaces
 */

#ifndef TEST_INTEGRATION_API_H
#define TEST_INTEGRATION_API_H

#include "quantum_geometric/physics/surface_code.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_backend_types.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/distributed/distributed_training_manager.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type aliases for test compatibility
// ============================================================================

// Surface code and error correction
typedef SurfaceCode surface_code;

// IBM backend types - use actual backend structures for proper API unification
typedef struct IBMConfig IBMConfig;
typedef IBMBackendState IBMState;  // Unify with real API

// Rigetti backend types
typedef struct RigettiConfig RigettiConfig;
typedef struct RigettiState RigettiState;  // Keep as-is, already unified

// D-Wave backend types
typedef struct DWaveConfig DWaveConfig;
typedef struct DWaveState DWaveState;  // Keep as-is, already unified

// Circuit and problem types
typedef struct quantum_circuit quantum_circuit;
typedef quantum_problem quantum_problem;

// Distributed system types
typedef distributed_config_t distributed_config;
typedef struct workload_spec workload_spec;
typedef struct computation_result computation_result;

// ============================================================================
// Node and communication types
// ============================================================================

typedef enum {
    COMPUTE_NODE,
    STORAGE_NODE,
    HYBRID_NODE
} NodeType;

typedef enum {
    SYNC,
    ASYNC,
    HYBRID_COMM
} CommunicationMode;

// ============================================================================
// Syndrome detector and error corrector
// ============================================================================

typedef struct syndrome_detector {
    SurfaceCode* surface_code;
    SyndromeVertex* vertices;
    size_t num_vertices;
    size_t capacity;
    bool initialized;
} syndrome_detector;

typedef struct error_corrector {
    SurfaceCode* surface_code;
    size_t* correction_history;
    size_t history_size;
    size_t history_capacity;
    bool initialized;
} error_corrector;

typedef struct syndrome_result {
    SyndromeVertex* syndromes;
    size_t num_syndromes;
    double detection_confidence;
    bool has_errors;
} syndrome_result;

// ============================================================================
// Workload and distributed computation types
// ============================================================================

struct workload_spec {
    size_t num_tasks;
    size_t* task_sizes;
    void** task_data;
    uint32_t priority;
    bool requires_synchronization;
    double estimated_compute_time;
    distributed_config_t config;
};

struct computation_result {
    void* data;
    size_t data_size;
    double execution_time;
    double communication_overhead;
    bool success;
    char* error_message;
    uint32_t node_id;
};

// ============================================================================
// WRAPPER FUNCTIONS - These wrap the real API with test-friendly signatures
// The real functions have different signatures, these are convenience wrappers
// ============================================================================

// Wrapper for init_surface_code - takes width/height instead of full config
#define init_surface_code init_surface_code_wrapper
surface_code* init_surface_code_wrapper(size_t width, size_t height);

// Wrapper for backend initialization - takes state pointer + config
#define init_rigetti_backend init_rigetti_backend_wrapper
bool init_rigetti_backend_wrapper(RigettiState* state, const RigettiConfig* config);

#define init_dwave_backend init_dwave_backend_wrapper
bool init_dwave_backend_wrapper(DWaveState* state, const DWaveConfig* config);

// Wrapper for workload distribution - different signature from error_correction version
#define distribute_workload distribute_workload_wrapper
bool distribute_workload_wrapper(workload_spec* workload);

// ============================================================================
// Error correction pipeline functions
// ============================================================================

syndrome_detector* init_syndrome_detector(surface_code* code);
error_corrector* init_error_corrector(surface_code* code);
void perform_quantum_operation(surface_code* code);
syndrome_result* detect_error_syndromes(syndrome_detector* detector);
bool apply_surface_code_correction(error_corrector* corrector, syndrome_result* syndromes);  // Renamed to avoid collision
bool verify_quantum_state(surface_code* code);
void cleanup_syndrome_result(syndrome_result* result);
void cleanup_error_corrector(error_corrector* corrector);
void cleanup_syndrome_detector(syndrome_detector* detector);

// ============================================================================
// Hardware backend functions
// ============================================================================

bool execute_problem(DWaveState* state, quantum_problem* problem, quantum_result* result);
bool verify_quantum_result(quantum_result* result);
void cleanup_ibm_backend(IBMState* state);
void cleanup_rigetti_backend(RigettiState* state);
void cleanup_dwave_backend(DWaveState* state);

// ============================================================================
// Circuit and problem management
// ============================================================================

quantum_circuit* create_test_circuit(void);
quantum_problem* create_test_problem(void);
void cleanup_quantum_circuit(quantum_circuit* circuit);
void cleanup_quantum_problem(quantum_problem* problem);

// ============================================================================
// Performance monitoring helpers
// ============================================================================

void perform_test_operation(void);
void allocate_test_resources(void);

// ============================================================================
// Distributed system functions
// ============================================================================

bool init_distributed_system(distributed_config* config);
workload_spec* create_test_workload(void);
computation_result* perform_distributed_computation(void);
bool test_synchronize_computation_result(computation_result* result);  // Renamed to avoid collision
bool verify_distributed_state(void);
void cleanup_workload(workload_spec* workload);
void cleanup_distributed_system(void);
void cleanup_computation_result(computation_result* result);

// ============================================================================
// Fault tolerance functions
// ============================================================================

bool init_fault_tolerance_system(void);
bool verify_system_recovery(void);
bool verify_data_recovery(void);
bool verify_node_recovery(void);
bool verify_system_stability(void);
void cleanup_fault_tolerance_system(void);

// ============================================================================
// Error simulation and resilience testing
// ============================================================================

void simulate_resource_exhaustion(void);
bool verify_graceful_degradation(void);
void simulate_concurrent_failures(void);
bool verify_system_resilience(void);

// ============================================================================
// System initialization and cleanup
// ============================================================================

void init_quantum_system(void);
void init_hardware_backends(void);
void cleanup_quantum_system(void);
void cleanup_hardware_backends(void);

// ============================================================================
// State verification functions
// ============================================================================

bool verify_quantum_state_integrity(void);
bool verify_classical_state_integrity(void);
bool verify_resource_state(void);

// ============================================================================
// Error injection functions
// ============================================================================

void inject_qubit_errors(void);
void inject_gate_errors(void);
void inject_measurement_errors(void);

// ============================================================================
// Network simulation functions
// ============================================================================

void simulate_network_latency(void);
void simulate_packet_loss(void);
void simulate_connection_drops(void);

// ============================================================================
// Data corruption and node failure simulation
// ============================================================================

void corrupt_test_data(void);
void simulate_node_failures(void);

#ifdef __cplusplus
}
#endif

#endif // TEST_INTEGRATION_API_H
