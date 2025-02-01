#ifndef QUANTUM_HARDWARE_CONSTANTS_H
#define QUANTUM_HARDWARE_CONSTANTS_H

#include <string.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Helper function to safely initialize device name
static inline void init_device_name(char device_name[256], const char* name) {
    strncpy(device_name, name, 255);
    device_name[255] = '\0';
}

// Default hardware configurations
static quantum_hardware_t create_default_cpu_config(void) {
    quantum_hardware_t config = {
        .type = HARDWARE_TYPE_CPU,
        .num_qubits = 0,
        .num_classical_bits = 0,
        .sys_caps = {
            .feature_flags = 0,
            .num_physical_cores = 1,
            .num_logical_cores = 1,
            .vector_register_width = 256,
            .max_vector_elements = 8,
            .cache_config = {{0}},
            .has_fma = true,
            .has_avx = true,
            .has_avx2 = true,
            .has_avx512 = false,
            .has_neon = false,
            .has_sve = false,
            .has_amx = false,
            .page_size = 4096,
            .huge_page_size = 2097152,
            .supports_huge_pages = false
        },
        .caps = {
            .supports_gpu = false,
            .supports_metal = false,
            .supports_cuda = false,
            .supports_openmp = true,
            .supports_mpi = false,
            .supports_distributed = false,
            .supports_feedback = false,
            .supports_reset = true,
            .max_qubits = 32,
            .max_gates = 1000000,
            .max_depth = 1000,
            .max_shots = 10000,
            .max_parallel_jobs = 1,
            .memory_size = 0,
            .coherence_time = 0.0,
            .gate_time = 0.0,
            .readout_time = 0.0,
            .extensions = NULL,
            .device_specific = NULL
        },
        .device_data = NULL,
        .device_id = 0,
        .device_handle = NULL,
        .context = NULL,
        .command_queue = NULL
    };
    init_device_name(config.device_name, "CPU");
    return config;
}

static quantum_hardware_t create_default_gpu_config(void) {
    quantum_hardware_t config = {
        .type = HARDWARE_TYPE_GPU,
        .num_qubits = 0,
        .num_classical_bits = 0,
        .sys_caps = {
            .feature_flags = 0,
            .num_physical_cores = 1,
            .num_logical_cores = 1,
            .vector_register_width = 256,
            .max_vector_elements = 8,
            .cache_config = {{0}},
            .has_fma = true,
            .has_avx = true,
            .has_avx2 = true,
            .has_avx512 = false,
            .has_neon = false,
            .has_sve = false,
            .has_amx = false,
            .page_size = 4096,
            .huge_page_size = 2097152,
            .supports_huge_pages = false
        },
        .caps = {
            .supports_gpu = true,
            .supports_metal = false,
            .supports_cuda = true,
            .supports_openmp = false,
            .supports_mpi = false,
            .supports_distributed = false,
            .supports_feedback = false,
            .supports_reset = true,
            .max_qubits = 32,
            .max_gates = 1000000,
            .max_depth = 1000,
            .max_shots = 10000,
            .max_parallel_jobs = 1,
            .memory_size = 0,
            .coherence_time = 0.0,
            .gate_time = 0.0,
            .readout_time = 0.0,
            .extensions = NULL,
            .device_specific = NULL
        },
        .device_data = NULL,
        .device_id = 0,
        .device_handle = NULL,
        .context = NULL,
        .command_queue = NULL
    };
    init_device_name(config.device_name, "GPU");
    return config;
}

static quantum_hardware_t create_default_metal_config(void) {
    quantum_hardware_t config = {
        .type = HARDWARE_TYPE_METAL,
        .num_qubits = 0,
        .num_classical_bits = 0,
        .sys_caps = {
            .feature_flags = 0,
            .num_physical_cores = 1,
            .num_logical_cores = 1,
            .vector_register_width = 256,
            .max_vector_elements = 8,
            .cache_config = {{0}},
            .has_fma = true,
            .has_avx = true,
            .has_avx2 = true,
            .has_avx512 = false,
            .has_neon = false,
            .has_sve = false,
            .has_amx = false,
            .page_size = 4096,
            .huge_page_size = 2097152,
            .supports_huge_pages = false
        },
        .caps = {
            .supports_gpu = true,
            .supports_metal = true,
            .supports_cuda = false,
            .supports_openmp = false,
            .supports_mpi = false,
            .supports_distributed = false,
            .supports_feedback = false,
            .supports_reset = true,
            .max_qubits = 32,
            .max_gates = 1000000,
            .max_depth = 1000,
            .max_shots = 10000,
            .max_parallel_jobs = 1,
            .memory_size = 0,
            .coherence_time = 0.0,
            .gate_time = 0.0,
            .readout_time = 0.0,
            .extensions = NULL,
            .device_specific = NULL
        },
        .device_data = NULL,
        .device_id = 0,
        .device_handle = NULL,
        .context = NULL,
        .command_queue = NULL
    };
    init_device_name(config.device_name, "Metal");
    return config;
}

// Helper functions to get default configs
static inline quantum_hardware_t get_default_cpu_config(void) {
    return create_default_cpu_config();
}

static inline quantum_hardware_t get_default_gpu_config(void) {
    return create_default_gpu_config();
}

static inline quantum_hardware_t get_default_metal_config(void) {
    return create_default_metal_config();
}

#endif // QUANTUM_HARDWARE_CONSTANTS_H
