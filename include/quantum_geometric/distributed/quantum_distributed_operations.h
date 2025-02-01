#ifndef QUANTUM_DISTRIBUTED_OPERATIONS_H
#define QUANTUM_DISTRIBUTED_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/distributed/distributed_training_manager.h"
#include <stdbool.h>

// Initialize distributed environment
qgt_error_t geometric_distribute(quantum_geometric_state_t* state,
                               const distributed_config_t* config);

// Gather distributed state to primary process
qgt_error_t geometric_gather(quantum_geometric_state_t* result,
                           const quantum_geometric_state_t* state);

// Check if state is distributed
bool geometric_is_distributed(const quantum_geometric_state_t* state);

#endif // QUANTUM_DISTRIBUTED_OPERATIONS_H
