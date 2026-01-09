/**
 * @file quantum_hardware_types.h
 * @brief Common type definitions for quantum hardware backends
 */

#ifndef QUANTUM_HARDWARE_TYPES_H
#define QUANTUM_HARDWARE_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Include base types for HardwareType (defined there)
#include "quantum_geometric/core/quantum_base_types.h"

#include "hardware_capabilities.h"
#include "quantum_backend_types.h"
#include "quantum_geometric/supercomputer/compute_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for types used in this header
struct QuantumCircuit;
struct QuantumProgram;
struct ExecutionResult;
struct HardwareGate;
struct IBMConfig;
struct RigettiConfig;
struct DWaveConfig;
struct SimulatorConfig;
struct MitigationParams;
struct NoiseModel;

// Hardware capability flags
typedef enum {
    CAP_NONE = 0,
    CAP_GPU = 1 << 0,
    CAP_METAL = 1 << 1,
    CAP_CUDA = 1 << 2,
    CAP_OPENMP = 1 << 3,
    CAP_MPI = 1 << 4,
    CAP_DISTRIBUTED = 1 << 5,
    CAP_FEEDBACK = 1 << 6,
    CAP_RESET = 1 << 7,
    CAP_HUGE_PAGES = 1 << 8,
    CAP_FMA = 1 << 9,
    CAP_AVX = 1 << 10,
    CAP_AVX2 = 1 << 11,
    CAP_AVX512 = 1 << 12,
    CAP_NEON = 1 << 13,
    CAP_SVE = 1 << 14,
    CAP_AMX = 1 << 15
} HardwareCapabilityFlags;

// Hardware backend types (quantum hardware only)
typedef enum {
    HARDWARE_NONE,
    HARDWARE_IBM,
    HARDWARE_RIGETTI,
    HARDWARE_DWAVE,
    HARDWARE_SIMULATOR
} HardwareBackendType;

// HardwareType is defined in quantum_base_types.h - do not redefine

// Backend type enum
typedef enum {
    BACKEND_IBM,
    BACKEND_RIGETTI,
    BACKEND_DWAVE,
    BACKEND_SIMULATOR
} BackendType;

// Error mitigation types
typedef enum {
    MITIGATION_NONE,
    MITIGATION_RICHARDSON,
    MITIGATION_ZNE,
    MITIGATION_PROBABILISTIC,
    MITIGATION_CUSTOM
} MitigationType;

// IBM backend configuration
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

// IBM backend state (extended for optimized backend)
typedef struct IBMBackendState {
    bool initialized;                 // Backend initialization status
    bool connected;                   // Connection status
    IBMBackendConfig config;          // Backend configuration
    double* error_rates;              // Per-qubit error rates
    double* readout_errors;           // Per-qubit readout error rates
    size_t num_qubits;                // Number of qubits
    void* api_handle;                 // API handle
    struct IBMResultData* last_result_data;  // Last execution result data
    // Extended fields for optimized backend
    double* calibration_data;         // Full calibration data array
    bool* qubit_availability;         // Per-qubit availability status
    size_t* measurement_order;        // Optimized measurement order
    double** coupling_map;            // Qubit coupling strength matrix
} IBMBackendState;

// Quantum hardware capabilities
typedef struct QuantumHardwareCapabilities {
    bool supports_gpu;                // GPU acceleration support
    bool supports_metal;              // Metal API support
    bool supports_cuda;               // CUDA support
    bool supports_openmp;             // OpenMP support
    bool supports_mpi;                // MPI support
    bool supports_distributed;        // Distributed computing support
    bool supports_feedback;           // Real-time feedback support
    bool supports_reset;              // Qubit reset support
    bool supports_gates;              // Gate-based quantum operations
    bool supports_measurement;        // Measurement support
    bool supports_annealing;          // Quantum annealing support
    bool supports_conditional;        // Conditional operations
    bool supports_parallel;           // Parallel execution
    bool supports_error_correction;   // Quantum error correction
    uint32_t max_qubits;              // Maximum number of qubits
    uint32_t max_gates;               // Maximum number of gates
    uint32_t max_depth;               // Maximum circuit depth
    uint32_t max_shots;               // Maximum shots per job
    uint32_t max_parallel_jobs;       // Maximum parallel jobs
    size_t memory_size;               // Device memory size
    double coherence_time;            // Qubit coherence time
    double gate_time;                 // Gate operation time
    double readout_time;              // Measurement readout time
    double* connectivity;             // Qubit connectivity matrix
    size_t connectivity_size;         // Size of connectivity matrix
    size_t num_gates;                 // Number of available gates
    char** available_gates;           // Names of available gates
    void* extensions;                 // Backend-specific extensions
    void* device_specific;            // Device-specific data
    void* backend_specific;           // Backend-specific capabilities
} QuantumHardwareCapabilities;

// Alias for HardwareCapabilities used in abstraction layer
typedef QuantumHardwareCapabilities HardwareCapabilities;

// Function declarations for runtime capability detection
SystemCapabilities detect_system_capabilities(void);
QuantumHardwareCapabilities detect_quantum_capabilities(HardwareType type);

struct MitigationParams {
    MitigationType type;
    double* scale_factors;
    size_t num_factors;
    double mitigation_factor;
    void* custom_parameters;
};

// Noise model structure
struct NoiseModel {
    double* gate_errors;           // Gate error rates
    double* readout_errors;        // Readout error rates
    double* decoherence_rates;     // Decoherence rates
    void* backend_specific_noise;  // Backend-specific noise parameters
};

// ============================================================================
// Unified HardwareProfile - single source of truth for all hardware profiling
// ============================================================================

#ifndef HARDWARE_PROFILE_DEFINED
#define HARDWARE_PROFILE_DEFINED

/**
 * @brief Comprehensive hardware profile for quantum operations
 *
 * This is the unified struct for all hardware profiling needs.
 * All files should include quantum_hardware_types.h and use this definition.
 */
typedef struct HardwareProfile {
    // Basic hardware info
    size_t num_qubits;                  // Number of qubits

    // Fidelity metrics (per-qubit arrays)
    double* gate_fidelities;            // Gate fidelity per qubit
    double* measurement_fidelities;     // Measurement fidelity per qubit
    double* readout_fidelities;         // Alias for measurement_fidelities (readout)

    // Single-value fidelity metrics (averages)
    double gate_fidelity;               // Average gate fidelity
    double measurement_fidelity;        // Average measurement fidelity

    // Coherence times (per-qubit arrays)
    double* t1_times;                   // T1 relaxation times per qubit
    double* t2_times;                   // T2 dephasing times per qubit
    double coherence_time;              // Average coherence time

    // Coupling and crosstalk
    double* coupling_map;               // Coupling strength matrix
    double* crosstalk_matrix;           // Crosstalk coefficients

    // Noise parameters
    double thermal_noise;               // Thermal noise level
    double readout_noise;               // Readout noise level
    double gate_noise;                  // Gate operation noise
    double noise_scale;                 // Overall noise scaling factor

    // Timing parameters
    double measurement_time;            // Measurement duration (us)
    double gate_time;                   // Single-qubit gate time (ns)
    double two_qubit_gate_time;         // Two-qubit gate time (ns)

    // Calibration data
    double last_calibration_time;       // Timestamp of last calibration
    double phase_calibration;           // Phase calibration factor
    bool calibration_valid;             // Whether calibration is current

    // Error mitigation parameters
    double min_reliability_threshold;   // Minimum reliability for corrections
    double min_confidence_threshold;    // Minimum confidence threshold
    double error_rate_learning_rate;    // Learning rate for error updates
    double confidence_scale_factor;     // Scaling factor for confidence

    // Pattern detection parameters
    double learning_rate;               // General learning rate for updates
    double spatial_scale;               // Spatial decay scale for correlations
    double pattern_scale_factor;        // Scaling factor for pattern detection

    // Capability flags
    bool supports_parallel_measurement; // Supports parallel measurement
} HardwareProfile;

#endif // HARDWARE_PROFILE_DEFINED

// Generic gate structure (for named gates)
typedef struct {
    char name[64];                    // Gate name
    uint32_t num_qubits;             // Number of qubits
    uint32_t* qubit_indices;         // Target qubit indices
    double* parameters;              // Gate parameters
    void* custom_data;               // Custom gate data
} GenericQuantumGate;

// Gate structure for hardware abstraction (type-based)
typedef struct {
    gate_type_t type;                // Gate type (GATE_X, GATE_H, etc.)
    uint32_t qubit;                  // Target qubit for single-qubit gates
    uint32_t control_qubit;          // Control qubit for 2-qubit gates
    uint32_t target_qubit;           // Target qubit for 2-qubit gates
    double parameter;                // Rotation parameter
    double* parameters;              // Multiple parameters (optional)
    size_t num_parameters;           // Number of parameters
} QuantumGate;

// Hardware configuration
typedef struct {
    HardwareType type;                // Hardware backend type
    uint32_t num_qubits;              // Number of qubits
    uint32_t num_classical_bits;      // Number of classical bits
    SystemCapabilities sys_caps;      // System capabilities
    QuantumHardwareCapabilities caps; // Quantum capabilities
    char device_name[256];            // Device name
    void* device_data;                // Device-specific data
    uint32_t device_id;               // Device identifier
    void* device_handle;              // Device handle
    void* context;                    // Device context
    void* command_queue;              // Command queue
} quantum_hardware_t;

// Helper functions for capability checking
static inline bool has_system_capability(const SystemCapabilities* sys_caps, HardwareCapabilityFlags flag) {
    return (sys_caps->feature_flags & flag) != 0;
}

static inline bool has_quantum_capability(const QuantumHardwareCapabilities* caps, HardwareCapabilityFlags flag) {
    switch (flag) {
        case CAP_GPU:
            return caps->supports_gpu;
        case CAP_METAL:
            return caps->supports_metal;
        case CAP_CUDA:
            return caps->supports_cuda;
        case CAP_OPENMP:
            return caps->supports_openmp;
        case CAP_MPI:
            return caps->supports_mpi;
        case CAP_DISTRIBUTED:
            return caps->supports_distributed;
        case CAP_FEEDBACK:
            return caps->supports_feedback;
        case CAP_RESET:
            return caps->supports_reset;
        default:
            return false;
    }
}

static inline bool has_capability(const quantum_hardware_t* hw, HardwareCapabilityFlags flag) {
    return has_system_capability(&hw->sys_caps, flag) || 
           has_quantum_capability(&hw->caps, flag);
}

// Performance metrics
typedef struct {
    uint64_t page_faults;
    uint64_t cache_misses;
    uint64_t tlb_misses;
    double efficiency;
    struct {
        uint64_t total_cycles;
        uint64_t stall_cycles;
        uint64_t branch_misses;
        uint64_t instructions;
    } cpu;
    struct {
        uint64_t allocations;
        uint64_t deallocations;
        double fragmentation;
        double utilization;
    } memory;

    // Production monitoring metrics
    double error_rate;
    double avg_latency;
    double peak_memory_usage;
    double avg_cpu_utilization;
    double avg_gpu_utilization;
    double success_rate;
    double false_positive_rate;
    double recovery_success_rate;
} PerformanceMetrics;

typedef struct {
    BackendType type;
    union {
        IBMBackendConfig ibm;
        struct RigettiConfig rigetti;
        struct DWaveConfig dwave;
        struct SimulatorConfig simulator;
    } config;
} QuantumBackendConfig;

typedef struct {
    BackendType type;
    union {
        IBMBackendState ibm;
        struct RigettiState rigetti;
        struct DWaveState dwave;
        struct SimulatorState simulator;
    } state;
} QuantumBackendState;

// ============================================================================
// Hardware Abstraction Types
// ============================================================================

// Circuit validation result
typedef struct ValidationResult {
    bool is_valid;                    // Whether the circuit is valid
    char* error_message;              // Error message if invalid
    size_t error_location;            // Location of error in circuit
    int error_code;                   // Error code
} ValidationResult;

// Error mitigation strategy
typedef struct ErrorMitigationStrategy {
    MitigationType type;              // Mitigation type
    double* noise_amplification;      // Noise amplification factors for ZNE
    size_t num_amplification;         // Number of amplification factors
    double* quasi_probabilities;      // Quasi-probabilities for PEC
    size_t num_qp;                    // Number of quasi-probability entries
    void* custom_data;                // Custom mitigation data
} ErrorMitigationStrategy;

// Optimized circuit structure
typedef struct OptimizedCircuit {
    struct QuantumCircuit* circuit;   // The optimized circuit
    struct MitigationParams* error_mitigation;  // Error mitigation parameters
    int optimization_level;           // Optimization level applied
    double estimated_fidelity;        // Estimated fidelity after optimization
    double* qubit_mapping;            // Logical to physical qubit mapping
    size_t num_qubits;                // Number of qubits
} OptimizedCircuit;

// D-Wave QUBO (Quadratic Unconstrained Binary Optimization) problem
typedef struct QUBO {
    double* linear;                   // Linear terms (bias)
    double* quadratic;                // Quadratic terms (coupling)
    size_t num_variables;             // Number of binary variables
    size_t num_couplings;             // Number of quadratic terms
    uint32_t* variable_indices;       // Variable indices for quadratic terms
    double offset;                    // Constant offset
} QUBO;

// D-Wave QUBO solution result
typedef struct QUBOResult {
    int* solutions;                   // Array of solutions (binary values)
    double* energies;                 // Energy of each solution
    size_t num_solutions;             // Number of solutions
    size_t num_variables;             // Number of variables per solution
    int* num_occurrences;             // Occurrence count of each solution
    double timing_total;              // Total execution time
    double timing_sampling;           // Sampling time
    void* raw_data;                   // Raw backend data
} QUBOResult;

// Crosstalk mitigation data
typedef struct CrosstalkMitigation {
    double** crosstalk_matrix;        // Crosstalk coefficient matrix
    size_t num_qubits;                // Number of qubits
    double** compensation_pulses;     // Compensation pulse parameters
} CrosstalkMitigation;

// Crosstalk map
typedef struct CrosstalkMap {
    double** coefficients;            // Crosstalk coefficients between qubit pairs
    size_t num_qubits;                // Number of qubits
    CrosstalkMitigation* mitigation_strategies;  // Mitigation strategies
} CrosstalkMap;

// Qubit connectivity map
typedef struct ConnectivityMap {
    bool** connected;                 // Adjacency matrix of connected qubits
    size_t num_qubits;                // Number of qubits
    double** coupling_strengths;      // Coupling strengths between qubits
    double** gate_fidelities;         // Two-qubit gate fidelities
} ConnectivityMap;

// Error rate data
typedef struct ErrorRates {
    double* single_qubit_errors;      // Single-qubit gate errors
    double* two_qubit_errors;         // Two-qubit gate errors
    double* readout_errors;           // Readout/measurement errors
    double* t1_times;                 // T1 relaxation times
    double* t2_times;                 // T2 coherence times
    size_t num_qubits;                // Number of qubits
} ErrorRates;

// Unified quantum hardware structure
typedef struct QuantumHardware {
    HardwareBackendType type;         // Quantum backend type (IBM, Rigetti, etc.)
    ComputeBackendType compute_backend;  // Compute acceleration (Metal, CUDA, etc.)
    QuantumHardwareCapabilities capabilities;  // Hardware capabilities
    ConnectivityMap connectivity;     // Qubit connectivity
    ErrorRates error_rates;           // Error rates
    struct NoiseModel noise_model;    // Noise model
    CrosstalkMap crosstalk;           // Crosstalk information
    union {
        struct IBMConfig* ibm;
        struct RigettiConfig* rigetti;
        struct DWaveConfig* dwave;
        struct SimulatorConfig* simulator;
    } backend;
    void* device_data;                // Device-specific data
} QuantumHardware;

// Helper function prototypes for new types
void cleanup_optimized_circuit(OptimizedCircuit* circuit);
void cleanup_qubo(QUBO* qubo);
void cleanup_qubo_result(QUBOResult* result);
void cleanup_connectivity(ConnectivityMap* connectivity);
void cleanup_noise_model(struct NoiseModel* noise_model);
void cleanup_crosstalk(CrosstalkMap* crosstalk);
QUBO* program_to_qubo(const struct QuantumProgram* program);
QUBO* circuit_to_qubo(const struct QuantumCircuit* circuit);
void convert_qubo_result(const QUBOResult* qubo_result, struct ExecutionResult* result);
bool is_qubo_circuit(const struct QuantumCircuit* circuit);
size_t compute_circuit_depth(const struct QuantumCircuit* circuit);
bool is_gate_supported(const struct HardwareGate* gate, const QuantumHardwareCapabilities* caps);
bool check_connectivity(const struct QuantumCircuit* circuit, const ConnectivityMap* connectivity);
double estimate_circuit_fidelity(const struct QuantumCircuit* circuit, const ErrorRates* rates, const struct NoiseModel* noise);
ErrorMitigationStrategy select_error_mitigation(const struct QuantumCircuit* circuit, const ErrorRates* rates, const struct NoiseModel* noise);

// Backend-specific functions
CrosstalkMap get_rigetti_crosstalk(struct RigettiConfig* config);
CrosstalkMitigation* get_rigetti_crosstalk_mitigation(struct RigettiConfig* config);
CrosstalkMap get_ibm_crosstalk(struct IBMConfig* config);
CrosstalkMap get_dwave_crosstalk(struct DWaveConfig* config);

// Circuit submission functions
int submit_rigetti_circuit(struct RigettiConfig* config, struct QuantumCircuit* circuit, void* options, struct ExecutionResult* result);
int submit_ibm_circuit(struct IBMConfig* config, struct QuantumCircuit* circuit, struct ExecutionResult* result);
bool submit_dwave_problem(struct DWaveConfig* config, QUBO* qubo, QUBOResult* result);
int simulate_circuit(struct QuantumCircuit* circuit, struct ExecutionResult* result);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HARDWARE_TYPES_H
