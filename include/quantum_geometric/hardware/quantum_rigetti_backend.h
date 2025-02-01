/**
 * @file quantum_rigetti_backend.h
 * @brief Rigetti Quantum backend with semi-classical emulation fallback
 */

#ifndef QUANTUM_RIGETTI_BACKEND_H
#define QUANTUM_RIGETTI_BACKEND_H

#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"

// Rigetti backend types
typedef enum {
    RIGETTI_BACKEND_REAL,     // Real quantum hardware
    RIGETTI_BACKEND_SIMULATOR // Semi-classical emulation
} RigettiBackendType;

// Rigetti backend status
typedef enum {
    RIGETTI_STATUS_IDLE,
    RIGETTI_STATUS_QUEUED,
    RIGETTI_STATUS_RUNNING,
    RIGETTI_STATUS_COMPLETED,
    RIGETTI_STATUS_ERROR,
    RIGETTI_STATUS_CANCELLED
} RigettiJobStatus;

// Rigetti backend capabilities
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
    double connectivity[MAX_QUBITS][MAX_QUBITS];
    char* instruction_set[64];
    uint32_t num_instructions;
    bool supports_pyquil;
    bool supports_quil_t;
} RigettiCapabilities;

// Rigetti backend configuration
typedef struct {
    RigettiBackendType type;
    char* api_key;
    char* api_secret;
    char* backend_name;
    uint32_t max_shots;
    bool optimize_mapping;
    NoiseModel noise_model;
    MitigationParams error_mitigation;
    SimulatorConfig simulator_config; // For emulation mode
    void* custom_config;
} RigettiBackendConfig;

// Rigetti job configuration
typedef struct {
    QuantumCircuit* circuit;
    uint32_t shots;
    bool optimize;
    bool use_error_mitigation;
    bool use_parametric_compilation;
    char* job_tags[8];
    void* custom_options;
} RigettiJobConfig;

// Rigetti job result
typedef struct {
    uint64_t* counts;
    double* probabilities;
    double fidelity;
    double error_rate;
    RigettiJobStatus status;
    char* error_message;
    void* raw_data;
    double* parametric_values;
} RigettiJobResult;

// Initialize Rigetti backend
RigettiConfig* init_rigetti_backend(const RigettiBackendConfig* config);

// Create quantum circuit
QuantumCircuit* create_rigetti_circuit(uint32_t num_qubits, uint32_t num_classical_bits);

// Add quantum gate
bool add_rigetti_gate(QuantumCircuit* circuit, GateType type, uint32_t target, uint32_t control, double* parameters);

// Add controlled gate
bool add_rigetti_controlled_gate(QuantumCircuit* circuit, GateType type, uint32_t target, uint32_t control, uint32_t control2, double* parameters);

// Submit job to Rigetti backend
char* submit_rigetti_job(RigettiConfig* config, const RigettiJobConfig* job_config);

// Get job status
RigettiJobStatus get_rigetti_job_status(RigettiConfig* config, const char* job_id);

// Get job result
RigettiJobResult* get_rigetti_job_result(RigettiConfig* config, const char* job_id);

// Cancel job
bool cancel_rigetti_job(RigettiConfig* config, const char* job_id);

// Get backend capabilities
RigettiCapabilities* get_rigetti_capabilities(RigettiConfig* config);

// Get available backends
char** get_rigetti_backends(RigettiConfig* config, size_t* num_backends);

// Get backend properties
char* get_rigetti_backend_properties(RigettiConfig* config, const char* backend_name);

// Get queue information
size_t get_rigetti_queue_position(RigettiConfig* config, const char* job_id);

// Get estimated runtime
double get_rigetti_estimated_runtime(RigettiConfig* config, const QuantumCircuit* circuit);

// Optimize circuit for backend
bool optimize_rigetti_circuit(QuantumCircuit* circuit, const RigettiCapabilities* capabilities);

// Apply error mitigation
bool apply_rigetti_error_mitigation(RigettiJobResult* result, const MitigationParams* params);

// Convert to Quil format
char* circuit_to_quil(const QuantumCircuit* circuit);

// Load from Quil format
QuantumCircuit* quil_to_circuit(const char* quil);

// Validate circuit for backend
bool validate_rigetti_circuit(const QuantumCircuit* circuit, const RigettiCapabilities* capabilities);

// Get error information
char* get_rigetti_error_info(RigettiConfig* config, const char* job_id);

// Clean up resources
void cleanup_rigetti_config(RigettiConfig* config);
void cleanup_rigetti_result(RigettiJobResult* result);
void cleanup_rigetti_capabilities(RigettiCapabilities* capabilities);

// Utility functions
bool save_rigetti_credentials(const char* api_key, const char* api_secret, const char* filename);
char* load_rigetti_credentials(const char* filename);
bool test_rigetti_connection(const char* api_key, const char* api_secret);
void set_rigetti_log_level(int level);
char* get_rigetti_version(void);

// Parametric compilation functions
bool add_parametric_gate(QuantumCircuit* circuit, GateType type, uint32_t target, const char* parameter_name);
bool set_parameter_value(RigettiConfig* config, const char* parameter_name, double value);
bool get_parameter_value(RigettiConfig* config, const char* parameter_name, double* value);

#endif // QUANTUM_RIGETTI_BACKEND_H
