/**
 * @file quantum_hardware_abstraction.h
 * @brief Unified hardware abstraction layer for quantum backends
 */

#ifndef QUANTUM_HARDWARE_ABSTRACTION_H
#define QUANTUM_HARDWARE_ABSTRACTION_H

#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Forward declarations for backend configs (defined in their respective headers)
struct NoiseModel;
struct MitigationParams;
struct SimulatorState;
struct IBMCapabilities;
struct IBMConfig;
struct DWaveConfig;
struct SimulatorConfig;

// ============================================================================
// ExecutionResult Definition
// ============================================================================

/**
 * @brief Result of quantum program execution
 */
typedef struct ExecutionResult {
    double* measurements;        // Measurement results
    double* probabilities;       // State probabilities
    uint64_t* counts;           // Shot counts
    size_t num_results;         // Number of results
    size_t num_shots;           // Total shots executed
    double fidelity;            // Estimated fidelity
    double error_rate;          // Error rate
    double execution_time;      // Execution time in seconds
    char* error_message;        // Error message if failed
    bool success;               // Whether execution succeeded
    void* backend_data;         // Backend-specific data
} ExecutionResult;

// ============================================================================
// QuantumCircuit Definition (for hardware backend operations)
// ============================================================================

/**
 * @brief Gate structure for hardware circuits
 * Uses gate_type_t from quantum_base_types.h
 */
typedef struct HardwareGate {
    gate_type_t type;      // Gate type (GATE_X, GATE_H, etc.)
    uint32_t target;       // Target qubit
    uint32_t control;      // Control qubit (for 2-qubit gates)
    uint32_t target1;      // First target (for multi-qubit gates)
    uint32_t target2;      // Second target (for multi-qubit gates)
    double parameter;      // Gate parameter (for rotation gates)
    double* parameters;    // Multiple parameters (optional)
    size_t num_params;     // Number of parameters
} HardwareGate;

/**
 * @brief Quantum circuit for hardware backends
 *
 * This structure represents a quantum circuit optimized for
 * hardware execution across different backends (IBM, Rigetti, etc.)
 */
typedef struct QuantumCircuit {
    HardwareGate* gates;           // Array of gates
    size_t num_gates;              // Number of gates
    size_t capacity;               // Allocated capacity
    size_t num_qubits;             // Number of qubits
    size_t num_classical_bits;     // Number of classical bits
    size_t depth;                  // Circuit depth
    bool* measured;                // Track measured qubits
    double max_circuit_depth;      // Maximum allowed depth
    void* optimization_data;       // Backend-specific optimization data
    void* metadata;                // Circuit metadata
} QuantumCircuit;

// Circuit manipulation functions (prefixed with hw_ to avoid conflicts)
void hw_insert_gate(QuantumCircuit* circuit, size_t index, HardwareGate gate);
void hw_replace_gate(QuantumCircuit* circuit, size_t index, HardwareGate gate);
void hw_remove_gate(QuantumCircuit* circuit, size_t index);
void hw_add_gate(QuantumCircuit* circuit, HardwareGate gate);

// Aliases for backward compatibility within hardware module
#define insert_gate hw_insert_gate
#define replace_gate hw_replace_gate
// Note: remove_gate and add_gate are too common - use hw_ prefix explicitly

// IonQ configuration
typedef struct IonQConfig {
    bool use_native_gates;
    double max_gate_time;
    void* custom_options;
} IonQConfig;

// Hardware optimization structures
typedef struct HardwareOptimizations {
    void (*ibm_optimize)(QuantumCircuit*, const IBMBackendConfig*);
    void (*rigetti_optimize)(QuantumCircuit*, const struct RigettiConfig*);
    void (*ionq_optimize)(QuantumCircuit*, const IonQConfig*);
    void (*dwave_optimize)(QuantumCircuit*, const struct DWaveConfig*);
} HardwareOptimizations;

// Hardware optimization functions
HardwareOptimizations* init_hardware_optimizations(const char* backend_type);
void cleanup_hardware_optimizations(HardwareOptimizations* opts);

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

// ============================================================================
// Backend-Specific Initialization Functions
// ============================================================================

// IBM Backend
bool init_ibm_capabilities(HardwareCapabilities* caps, const struct IBMConfig* config);
struct IBMConfig* init_ibm_backend(const struct IBMConfig* config);
void cleanup_ibm_backend(struct IBMConfig* backend);

// Rigetti Backend
bool init_rigetti_capabilities(HardwareCapabilities* caps, const struct RigettiConfig* config);
void cleanup_rigetti_backend(struct RigettiConfig* backend);

// D-Wave Backend
bool init_dwave_capabilities(HardwareCapabilities* caps, const struct DWaveConfig* config);
struct DWaveConfig* init_dwave_backend(const struct DWaveConfig* config);
void cleanup_dwave_backend(struct DWaveConfig* backend);

// Simulator Backend
bool init_simulator_capabilities(HardwareCapabilities* caps, const struct SimulatorConfig* config);
struct SimulatorConfig* init_simulator(const struct SimulatorConfig* config);
void cleanup_simulator(struct SimulatorConfig* backend);

#endif // QUANTUM_HARDWARE_ABSTRACTION_H
