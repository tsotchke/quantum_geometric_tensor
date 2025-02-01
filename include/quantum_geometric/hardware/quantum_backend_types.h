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
    char* backend_name;
    uint32_t max_shots;
    double coupling_map[64][64];
    double* noise_model;
    bool optimize_mapping;
    void* backend_specific_config;
};

struct RigettiConfig {
    char* backend_name;
    uint32_t max_shots;
    double coupling_map[64][64];
    double* noise_model;
    bool optimize_mapping;
    void* backend_specific_config;
};

struct DWaveConfig {
    char* backend_name;
    uint32_t max_shots;
    double coupling_map[64][64];
    double* noise_model;
    bool optimize_mapping;
    void* backend_specific_config;
};

struct SimulatorConfig {
    char* backend_name;
    uint32_t max_shots;
    double coupling_map[64][64];
    double* noise_model;
    bool optimize_mapping;
    void* backend_specific_config;
};

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_BACKEND_TYPES_H
