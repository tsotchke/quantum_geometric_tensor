/**
 * @file quantum_backend_types.h
 * @brief Type definitions for quantum hardware backends
 */

#ifndef QUANTUM_BACKEND_TYPES_H
#define QUANTUM_BACKEND_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct MemoryPool;

// Backend state structures
struct IBMState {
    double* amplitudes;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool* measurement_results;
    double fidelity;
    double error_rate;
    void* backend_specific_data;
};

struct RigettiState {
    double* amplitudes;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool* measurement_results;
    double fidelity;
    double error_rate;
    void* backend_specific_data;
};

struct DWaveState {
    double* amplitudes;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool* measurement_results;
    double fidelity;
    double error_rate;
    void* backend_specific_data;
};

struct SimulatorState {
    double* amplitudes;
    uint32_t num_qubits;
    uint32_t num_classical_bits;
    bool* measurement_results;
    double fidelity;
    double error_rate;
    void* backend_specific_data;
};

// Backend configuration structures
struct IBMConfig {
    char* api_key;              // IBM Quantum API key
    char* url;                  // IBM Quantum endpoint URL
    char* backend_name;         // Backend name (e.g., "ibmq_manila")
    char* hub;                  // IBM Quantum hub
    char* group;                // IBM Quantum group
    char* project;              // IBM Quantum project
    uint32_t max_shots;         // Maximum shots per job
    uint32_t max_qubits;        // Maximum qubits supported
    double coupling_map[64][64]; // Qubit connectivity
    double* noise_model;        // Noise model parameters
    bool optimize_mapping;      // Enable qubit mapping optimization
    int optimization_level;     // Transpilation optimization level (0-3)
    void* backend_specific_config;
};

struct RigettiConfig {
    char* api_key;              // Rigetti API key
    char* url;                  // Rigetti Quantum Cloud endpoint
    char* backend_name;         // Backend name (e.g., "Aspen-M-3")
    uint32_t max_shots;         // Maximum shots per job
    uint32_t max_qubits;        // Maximum qubits supported
    double coupling_map[64][64]; // Qubit connectivity
    double* noise_model;        // Noise model parameters
    bool optimize_mapping;      // Enable qubit mapping optimization
    bool use_pulse_control;     // Enable pulse-level control
    double t1_time;             // T1 relaxation time (us)
    double t2_time;             // T2 dephasing time (us)
    void* backend_specific_config;
};

struct DWaveConfig {
    char* api_key;              // D-Wave API token
    char* url;                  // D-Wave SAPI endpoint URL
    char* backend_name;         // Solver name (e.g., "Advantage_system6.3")
    uint32_t max_shots;         // Number of reads/samples
    uint32_t max_qubits;        // Maximum qubits (working qubits)
    double coupling_map[64][64]; // Qubit connectivity (Pegasus/Chimera graph)
    double* noise_model;        // Noise characteristics
    bool optimize_mapping;      // Enable minor embedding optimization
    size_t annealing_time;      // Annealing time in microseconds
    double chain_strength;      // Chain strength for embedding
    double energy_scale;        // Energy scale factor
    void* backend_specific_config;
};

struct SimulatorConfig {
    char* backend_name;         // Simulator name
    uint32_t max_shots;         // Maximum shots
    uint32_t max_qubits;        // Maximum qubits to simulate
    double coupling_map[64][64]; // Simulated connectivity
    double* noise_model;        // Noise model for noisy simulation
    bool optimize_mapping;      // Enable optimization
    bool use_gpu;               // Use GPU acceleration
    size_t memory_limit;        // Memory limit in bytes
    void* backend_specific_config;
};

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_BACKEND_TYPES_H
