#ifndef MEMORY_OPTIMIZATION_H
#define MEMORY_OPTIMIZATION_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/config/mpi_config.h"
#include <stddef.h>
#include <stdbool.h>

// Initialize distributed memory system
int qg_distributed_memory_init(const distributed_memory_config_t* config);

// Allocate memory in specified region
void* qg_distributed_malloc(size_t size, memory_region_type_t region_type);

// Free distributed memory
void qg_distributed_free(void* ptr);

// Get current memory distribution
const memory_distribution_t* qg_get_memory_distribution(void);

// Optimize memory distribution
int qg_optimize_memory_distribution(void);

// Clean up distributed memory system
void qg_distributed_memory_cleanup(void);

#endif // MEMORY_OPTIMIZATION_H
