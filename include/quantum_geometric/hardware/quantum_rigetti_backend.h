/**
 * @file quantum_rigetti_backend.h
 * @brief Rigetti Quantum backend with semi-classical emulation fallback
 */

#ifndef QUANTUM_RIGETTI_BACKEND_H
#define QUANTUM_RIGETTI_BACKEND_H

#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_backend_types.h"
#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/core/quantum_circuit_types.h"

// GateType alias for gate_type_t (defined in quantum_base_types.h)
typedef gate_type_t GateType;

// Forward declaration for simulator config
struct SimulatorConfig;

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
typedef struct RigettiBackendConfig {
    RigettiBackendType type;
    char* api_key;
    char* api_secret;
    char* backend_name;
    uint32_t max_shots;
    bool optimize_mapping;
    struct NoiseModel noise_model;
    struct MitigationParams error_mitigation;
    struct SimulatorConfig* simulator_config; // For emulation mode (pointer to avoid incomplete type)
    void* custom_config;
} RigettiBackendConfig;

// RigettiConfig is defined in quantum_backend_types.h
// This typedef provides backward compatibility
// struct RigettiConfig contains: api_key, url, backend_name, max_shots, etc.

// Rigetti job configuration
typedef struct RigettiJobConfig {
    struct QuantumCircuit* circuit;
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

// Initialize Rigetti backend - legacy interface with state
bool init_rigetti_backend_with_state(struct RigettiState* state, const struct RigettiConfig* config);

// Initialize Rigetti backend - returns config directly (test-compatible signature)
struct RigettiConfig* init_rigetti_backend(const RigettiBackendConfig* config);

// Submit circuit and get results directly
int submit_rigetti_circuit(struct RigettiConfig* config,
                          struct QuantumCircuit* circuit,
                          void* options,
                          struct ExecutionResult* result);

// Create quantum circuit
struct QuantumCircuit* create_rigetti_circuit(uint32_t num_qubits, uint32_t num_classical_bits);

// Add quantum gate
bool add_rigetti_gate(struct QuantumCircuit* circuit, GateType type, uint32_t target, uint32_t control, double* parameters);

// Add controlled gate
bool add_rigetti_controlled_gate(struct QuantumCircuit* circuit, GateType type, uint32_t target, uint32_t control, uint32_t control2, double* parameters);

// Submit job to Rigetti backend
char* submit_rigetti_job(struct RigettiConfig* config, const struct RigettiJobConfig* job_config);

// Get job status
RigettiJobStatus get_rigetti_job_status(struct RigettiConfig* config, const char* job_id);

// Get job result
RigettiJobResult* get_rigetti_job_result(struct RigettiConfig* config, const char* job_id);

// Cancel job
bool cancel_rigetti_job(struct RigettiConfig* config, const char* job_id);

// Get backend capabilities
RigettiCapabilities* get_rigetti_capabilities(struct RigettiConfig* config);

// Get available backends
char** get_rigetti_backends(struct RigettiConfig* config, size_t* num_backends);

// Get backend properties
char* get_rigetti_backend_properties(struct RigettiConfig* config, const char* backend_name);

// Get queue information
size_t get_rigetti_queue_position(struct RigettiConfig* config, const char* job_id);

// Get estimated runtime
double get_rigetti_estimated_runtime(struct RigettiConfig* config, const struct QuantumCircuit* circuit);

// Optimize circuit for backend
bool optimize_rigetti_circuit(struct QuantumCircuit* circuit, const RigettiCapabilities* capabilities);

// Apply error mitigation
bool apply_rigetti_error_mitigation(RigettiJobResult* result, const struct MitigationParams* params);

// Convert to Quil format
char* circuit_to_quil(const struct QuantumCircuit* circuit);

// Load from Quil format
struct QuantumCircuit* quil_to_circuit(const char* quil);

// Validate circuit for backend
bool validate_rigetti_circuit(const struct QuantumCircuit* circuit, const RigettiCapabilities* capabilities);

// Get error information
char* get_rigetti_error_info(struct RigettiConfig* config, const char* job_id);

// Clean up resources
void cleanup_rigetti_config(struct RigettiConfig* config);
void cleanup_rigetti_result(RigettiJobResult* result);
void cleanup_rigetti_capabilities(RigettiCapabilities* capabilities);

// Utility functions
bool save_rigetti_credentials(const char* api_key, const char* api_secret, const char* filename);
char* load_rigetti_credentials(const char* filename);
bool test_rigetti_connection(const char* api_key, const char* api_secret);
void set_rigetti_log_level(int level);
char* get_rigetti_version(void);

// Parametric compilation functions
bool add_parametric_gate(struct QuantumCircuit* circuit, GateType type, uint32_t target, const char* parameter_name);
bool set_parameter_value(struct RigettiConfig* config, const char* parameter_name, double value);
bool get_parameter_value(struct RigettiConfig* config, const char* parameter_name, double* value);

#endif // QUANTUM_RIGETTI_BACKEND_H
