#include "quantum_geometric/config/mpi_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Version string
#define MPI_CONFIG_VERSION "1.0.0"

int qg_mpi_config_init(mpi_config_t* config) {
    if (!config) {
        return -1;
    }

    // Initialize with default values
    config->num_processes = 1;
    config->process_rank = 0;
    config->chunk_size = 1024;
    config->use_gpu = 0;
    config->debug_level = 0;

    return 0;
}

void qg_mpi_config_cleanup(mpi_config_t* config) {
    if (!config) return;
    // Nothing to clean up since we don't have any dynamically allocated members
}

int qg_mpi_config_load(mpi_config_t* config, const char* config_file) {
    if (!config || !config_file) {
        return -1;
    }

    FILE* file = fopen(config_file, "r");
    if (!file) {
        return -1;
    }

    // Read configuration from file
    int result = fscanf(file, "%d %d %zu %d %d",
        &config->num_processes,
        &config->process_rank,
        &config->chunk_size,
        &config->use_gpu,
        &config->debug_level);

    fclose(file);
    return (result == 5) ? 0 : -1;
}

int qg_mpi_config_save(const mpi_config_t* config, const char* config_file) {
    if (!config || !config_file) {
        return -1;
    }

    FILE* file = fopen(config_file, "w");
    if (!file) {
        return -1;
    }

    // Write configuration to file
    int result = fprintf(file, "%d %d %zu %d %d\n",
        config->num_processes,
        config->process_rank,
        config->chunk_size,
        config->use_gpu,
        config->debug_level);

    fclose(file);
    return (result > 0) ? 0 : -1;
}

// Getters
int qg_mpi_config_get_num_processes(const mpi_config_t* config) {
    return config ? config->num_processes : -1;
}

int qg_mpi_config_get_process_rank(const mpi_config_t* config) {
    return config ? config->process_rank : -1;
}

size_t qg_mpi_config_get_chunk_size(const mpi_config_t* config) {
    return config ? config->chunk_size : 0;
}

int qg_mpi_config_get_use_gpu(const mpi_config_t* config) {
    return config ? config->use_gpu : -1;
}

int qg_mpi_config_get_debug_level(const mpi_config_t* config) {
    return config ? config->debug_level : -1;
}

// Setters
int qg_mpi_config_set_num_processes(mpi_config_t* config, int num_processes) {
    if (!config || num_processes < 1) {
        return -1;
    }
    config->num_processes = num_processes;
    return 0;
}

int qg_mpi_config_set_process_rank(mpi_config_t* config, int process_rank) {
    if (!config || process_rank < 0) {
        return -1;
    }
    config->process_rank = process_rank;
    return 0;
}

int qg_mpi_config_set_chunk_size(mpi_config_t* config, size_t chunk_size) {
    if (!config || chunk_size == 0) {
        return -1;
    }
    config->chunk_size = chunk_size;
    return 0;
}

int qg_mpi_config_set_use_gpu(mpi_config_t* config, int use_gpu) {
    if (!config) {
        return -1;
    }
    config->use_gpu = use_gpu;
    return 0;
}

int qg_mpi_config_set_debug_level(mpi_config_t* config, int debug_level) {
    if (!config || debug_level < 0) {
        return -1;
    }
    config->debug_level = debug_level;
    return 0;
}

// Utility functions
int qg_mpi_config_validate(const mpi_config_t* config) {
    if (!config) {
        return -1;
    }

    // Validate configuration values
    if (config->num_processes < 1) return -1;
    if (config->process_rank < 0 || config->process_rank >= config->num_processes) return -1;
    if (config->chunk_size == 0) return -1;
    if (config->debug_level < 0) return -1;

    return 0;
}

int qg_mpi_config_print(const mpi_config_t* config) {
    if (!config) {
        return -1;
    }

    printf("MPI Configuration:\n");
    printf("  Number of Processes: %d\n", config->num_processes);
    printf("  Process Rank: %d\n", config->process_rank);
    printf("  Chunk Size: %zu\n", config->chunk_size);
    printf("  Use GPU: %s\n", config->use_gpu ? "Yes" : "No");
    printf("  Debug Level: %d\n", config->debug_level);

    return 0;
}

const char* qg_mpi_config_get_version(void) {
    return MPI_CONFIG_VERSION;
}
