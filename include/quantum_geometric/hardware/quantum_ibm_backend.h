/**
 * @file quantum_ibm_backend.h
 * @brief IBM Quantum backend with semi-classical emulation fallback
 */

#ifndef QUANTUM_IBM_BACKEND_H
#define QUANTUM_IBM_BACKEND_H

#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_circuit.h"

// Forward declarations
struct NoiseModel;
struct MitigationParams;
struct SimulatorConfig;
struct ExecutionResult;

// Constants
#define IBM_MAX_QUBITS 127

// IBM backend types
typedef enum {
    IBM_BACKEND_REAL,     // Real quantum hardware
    IBM_BACKEND_SIMULATOR // Semi-classical emulation
} IBMBackendType;

// IBM backend status
typedef enum {
    IBM_STATUS_IDLE,
    IBM_STATUS_QUEUED,
    IBM_STATUS_RUNNING,
    IBM_STATUS_COMPLETED,
    IBM_STATUS_ERROR,
    IBM_STATUS_CANCELLED
} IBMJobStatus;

// IBM backend capabilities
typedef struct {
    uint32_t max_qubits;
    uint32_t max_shots;
    bool supports_conditional;
    bool supports_reset;
    bool supports_midcircuit_measurement;
    double t1_time;  // Relaxation time
    double t2_time;  // Dephasing time
    double readout_error;
    double gate_error;
    double connectivity[IBM_MAX_QUBITS][IBM_MAX_QUBITS];
} IBMCapabilities;

// Forward declarations from quantum_hardware_types.h
struct IBMBackendConfig;
struct IBMBackendState;

// IBM job configuration
typedef struct IBMJobConfig {
    struct QuantumCircuit* circuit;
    uint32_t shots;
    bool optimize;
    bool use_error_mitigation;
    char* job_tags[8];
    void* custom_options;
} IBMJobConfig;

// Extended IBM backend configuration
struct IBMBackendConfigExt {
    IBMBackendType type;
    char* hub;
    char* group;
    char* project;
    char* token;
    char* backend_name;
    uint32_t max_shots;
    uint32_t optimization_level;
    bool error_mitigation;
    bool dynamic_decoupling;
    bool readout_error_mitigation;
    bool measurement_error_mitigation;
    struct NoiseModel noise_model;
    struct MitigationParams error_mitigation_params;
    struct SimulatorConfig simulator_config;
    void* custom_config;
};

// Extended IBM job configuration
struct IBMBackendStateExt {
    struct QuantumCircuit* circuit;
    uint32_t shots;
    bool optimize;
    bool use_error_mitigation;
    char* job_tags[8];
    void* custom_options;
    uint32_t status;
    char* job_id;
    double* results;
    size_t result_size;
};

// IBM job result
typedef struct {
    uint64_t* counts;
    double* probabilities;
    double fidelity;
    double error_rate;
    IBMJobStatus status;
    char* error_message;
    void* raw_data;
} IBMJobResult;

// Initialize IBM backend
struct IBMBackendConfig* init_ibm_backend(const struct IBMBackendConfig* config);

// Create quantum circuit
struct QuantumCircuit* create_ibm_circuit(uint32_t num_qubits, uint32_t num_classical_bits);

// Add quantum gate
bool add_ibm_gate(struct QuantumCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, double* parameters);

// Add controlled gate
bool add_ibm_controlled_gate(struct QuantumCircuit* circuit, gate_type_t type, uint32_t target, uint32_t control, uint32_t control2, double* parameters);

// Submit job to IBM backend
char* submit_ibm_job(struct IBMBackendConfig* config, const IBMJobConfig* job_config);

// Get job status
IBMJobStatus get_ibm_job_status(struct IBMBackendConfig* config, const char* job_id);

// Get job result
IBMJobResult* get_ibm_job_result(struct IBMBackendConfig* config, const char* job_id);

// Cancel job
bool cancel_ibm_job(struct IBMBackendConfig* config, const char* job_id);

// Get backend capabilities
IBMCapabilities* get_ibm_capabilities(struct IBMBackendConfig* config);

// Get available backends
char** get_ibm_backends(struct IBMBackendConfig* config, size_t* num_backends);

// Get backend properties
char* get_ibm_backend_properties(struct IBMBackendConfig* config, const char* backend_name);

// Get queue information
size_t get_ibm_queue_position(struct IBMBackendConfig* config, const char* job_id);

// Get estimated runtime
double get_ibm_estimated_runtime(struct IBMBackendConfig* config, const struct QuantumCircuit* circuit);

// Optimize circuit for backend
bool optimize_ibm_circuit(struct QuantumCircuit* circuit, const IBMCapabilities* capabilities);

// Error mitigation functions
bool apply_ibm_error_mitigation(IBMJobResult* result, const struct MitigationParams* params);
double richardson_extrapolate(double value, const double* scale_factors, size_t num_factors);
bool apply_zne_mitigation(struct ExecutionResult* result, const struct MitigationParams* params);

// Convert to QASM format
char* circuit_to_qasm(const struct QuantumCircuit* circuit);

// Load from QASM format
struct QuantumCircuit* qasm_to_circuit(const char* qasm);

// Validate circuit for backend
bool validate_ibm_circuit(const struct QuantumCircuit* circuit, const IBMCapabilities* capabilities);

// Get error information
char* get_ibm_error_info(struct IBMBackendConfig* config, const char* job_id);

// Clean up resources
void cleanup_ibm_config(struct IBMBackendConfig* config);
void cleanup_ibm_result(IBMJobResult* result);
void cleanup_ibm_capabilities(IBMCapabilities* capabilities);

// Utility functions
bool save_ibm_credentials(const char* token, const char* filename);
char* load_ibm_credentials(const char* filename);
bool test_ibm_connection(const char* token);
void set_ibm_log_level(int level);
char* get_ibm_version(void);

#endif // QUANTUM_IBM_BACKEND_H
