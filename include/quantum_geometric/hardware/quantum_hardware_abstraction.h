/**
 * @file quantum_hardware_abstraction.h
 * @brief Unified hardware abstraction layer for quantum backends
 */

#ifndef QUANTUM_HARDWARE_ABSTRACTION_H
#define QUANTUM_HARDWARE_ABSTRACTION_H

#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Forward declarations
struct NoiseModel;
struct MitigationParams;
struct SimulatorState;
struct QuantumCircuit;
struct ExecutionResult;
struct IBMCapabilities;
struct IBMConfig;
struct RigettiConfig;
struct DWaveConfig;
struct SimulatorConfig;

// Hardware capabilities
typedef struct {
    uint32_t max_qubits;
    uint32_t max_shots;
    bool supports_gates;           // Gate-based quantum computing
    bool supports_annealing;       // Quantum annealing
    bool supports_measurement;     // Mid-circuit measurement
    bool supports_reset;          // Qubit reset
    bool supports_conditional;    // Conditional operations
    bool supports_parallel;       // Parallel execution
    bool supports_error_correction; // Quantum error correction
    double coherence_time;        // Coherence time in microseconds
    double gate_time;            // Gate operation time
    double readout_time;         // Measurement readout time
    double* connectivity;        // Qubit connectivity matrix
    size_t connectivity_size;    // Size of connectivity matrix
    char** available_gates;      // List of available quantum gates
    size_t num_gates;           // Number of available gates
    void* backend_specific;     // Backend-specific capabilities
} HardwareCapabilities;

// Hardware state structure
typedef struct {
    HardwareBackendType type;
    HardwareCapabilities capabilities;
    union {
        struct IBMConfig* ibm;
        struct RigettiConfig* rigetti;
        struct DWaveConfig* dwave;
        struct SimulatorConfig* simulator;
    } backend;
    int optimization_level;
    double error_budget;
    bool use_acceleration;
    struct MitigationParams error_mitigation;
} HardwareState;

// Hardware execution mode
typedef enum {
    EXECUTION_REAL,      // Real quantum hardware
    EXECUTION_SIMULATOR, // Semi-classical emulation
    EXECUTION_HYBRID     // Hybrid quantum-classical
} ExecutionMode;

// Hardware configuration
typedef struct {
    HardwareBackendType type;
    ExecutionMode mode;
    char* backend_name;
    union {
        struct IBMConfig ibm;
        struct RigettiConfig rigetti;
        struct DWaveConfig dwave;
        struct SimulatorConfig simulator;
    } config;
    struct NoiseModel noise_model;
    struct MitigationParams error_mitigation;
    void* custom_config;
} HardwareConfig;

// Quantum operation types
typedef enum {
    OPERATION_GATE,      // Quantum gate
    OPERATION_MEASURE,   // Measurement
    OPERATION_RESET,     // Qubit reset
    OPERATION_BARRIER,   // Synchronization barrier
    OPERATION_ANNEAL,    // Quantum annealing
    OPERATION_CUSTOM     // Custom operation
} OperationType;

// Quantum operation
typedef struct {
    OperationType type;
    union {
        QuantumGate gate;
        struct {
            uint32_t qubit;
            uint32_t classical_bit;
        } measure;
        struct {
            uint32_t qubit;
        } reset;
        struct {
            uint32_t* qubits;
            size_t num_qubits;
        } barrier;
        struct {
            double* schedule;
            size_t schedule_points;
        } anneal;
        void* custom;
    } op;
} QuantumOperation;

// Quantum program
typedef struct {
    QuantumOperation* operations;
    size_t num_operations;
    size_t capacity;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool optimize;
    bool use_error_mitigation;
    void* backend_specific;
} QuantumProgram;

// Initialize hardware backend
void* init_hardware(const HardwareConfig* config);

// Create quantum program
QuantumProgram* create_program(uint32_t num_qubits, uint32_t num_classical_bits);

// Add quantum operation
bool add_operation(QuantumProgram* program, const QuantumOperation* operation);

// Execute quantum program
struct ExecutionResult* execute_program(void* hardware, const QuantumProgram* program);

// Get execution status
char* get_execution_status(void* hardware, const char* execution_id);

// Cancel execution
bool cancel_execution(void* hardware, const char* execution_id);

// Get hardware capabilities
HardwareCapabilities* get_hardware_capabilities(void* hardware);

// Get available backends
char** get_available_backends(HardwareBackendType type, size_t* num_backends);

// Optimize program for backend
bool optimize_program(QuantumProgram* program, const HardwareCapabilities* capabilities);

// Apply error mitigation for simulator
bool apply_simulator_error_mitigation(struct SimulatorState* state, const struct MitigationParams* params);

// Apply error mitigation for execution results
bool apply_error_mitigation(struct ExecutionResult* result, const struct MitigationParams* params);

// Convert between representations
char* program_to_qasm(const QuantumProgram* program);
char* program_to_quil(const QuantumProgram* program);
char* program_to_json(const QuantumProgram* program);

// Validate program
bool validate_program(const QuantumProgram* program, const HardwareCapabilities* capabilities);

// Clean up resources
void cleanup_hardware(void* hardware);
void cleanup_program(QuantumProgram* program);
void cleanup_result(struct ExecutionResult* result);
void cleanup_capabilities(HardwareCapabilities* capabilities);

// Utility functions
bool save_credentials(HardwareBackendType type, const char* credentials, const char* filename);
char* load_credentials(HardwareBackendType type, const char* filename);
bool test_connection(void* hardware);
void set_log_level(int level);
char* get_version(void);

// Advanced control
bool set_execution_options(void* hardware, const void* options);
bool set_optimization_level(void* hardware, int level);
bool set_error_budget(void* hardware, double budget);
bool enable_hardware_acceleration(void* hardware, bool enable);

#endif // QUANTUM_HARDWARE_ABSTRACTION_H
