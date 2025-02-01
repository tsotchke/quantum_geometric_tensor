/**
 * @file quantum_hardware_types.h
 * @brief Common type definitions for quantum hardware backends
 */

#ifndef QUANTUM_HARDWARE_TYPES_H
#define QUANTUM_HARDWARE_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "hardware_capabilities.h"
#include "quantum_backend_types.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Hardware backend types
typedef enum {
    HARDWARE_NONE,
    HARDWARE_IBM,
    HARDWARE_RIGETTI,
    HARDWARE_DWAVE,
    HARDWARE_SIMULATOR
} HardwareBackendType;

// Hardware type enum
typedef enum {
    HARDWARE_TYPE_CPU,
    HARDWARE_TYPE_GPU,
    HARDWARE_TYPE_QPU,
    HARDWARE_TYPE_SIMULATOR,
    HARDWARE_TYPE_METAL
} HardwareType;

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
    uint32_t max_qubits;              // Maximum number of qubits
    uint32_t max_gates;               // Maximum number of gates
    uint32_t max_depth;               // Maximum circuit depth
    uint32_t max_shots;               // Maximum shots per job
    uint32_t max_parallel_jobs;       // Maximum parallel jobs
    size_t memory_size;               // Device memory size
    double coherence_time;            // Qubit coherence time
    double gate_time;                 // Gate operation time
    double readout_time;              // Measurement readout time
    void* extensions;                 // Backend-specific extensions
    void* device_specific;            // Device-specific data
} QuantumHardwareCapabilities;

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

typedef struct {
    char name[64];                    // Gate name
    uint32_t num_qubits;             // Number of qubits
    uint32_t* qubit_indices;         // Target qubit indices
    double* parameters;              // Gate parameters
    void* custom_data;               // Custom gate data
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
} PerformanceMetrics;

typedef struct {
    BackendType type;
    union {
        struct IBMConfig ibm;
        struct RigettiConfig rigetti;
        struct DWaveConfig dwave;
        struct SimulatorConfig simulator;
    } config;
} QuantumBackendConfig;

typedef struct {
    BackendType type;
    union {
        struct IBMState ibm;
        struct RigettiState rigetti;
        struct DWaveState dwave;
        struct SimulatorState simulator;
    } state;
} QuantumBackendState;

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_HARDWARE_TYPES_H
