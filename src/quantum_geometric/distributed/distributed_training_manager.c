/**
 * @file distributed_training_manager.c
 * @brief Implementation of distributed training management
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// MPI guard
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs for non-MPI builds
typedef int MPI_Comm;
typedef int MPI_Status;
typedef int MPI_Datatype;
typedef int MPI_Info;
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL 0
#define MPI_SUCCESS 0
#define MPI_IN_PLACE ((void*)1)
#define MPI_DOUBLE 0
#define MPI_BYTE 0
#define MPI_UNSIGNED_LONG 0
#define MPI_INT 0
#define MPI_SUM 0
#define MPI_INFO_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_UNDEFINED (-1)
#define MPI_MAX 0

// Stub MPI functions
static inline int MPI_Initialized(int* flag) { *flag = 1; return 0; }
static inline int MPI_Init(int* argc, char*** argv) { return 0; }
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { *size = 1; return 0; }
static inline int MPI_Comm_split_type(MPI_Comm comm, int type, int key, MPI_Info info, MPI_Comm* newcomm) { *newcomm = 0; return 0; }
static inline int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm) { *newcomm = 0; return 0; }
static inline int MPI_Comm_free(MPI_Comm* comm) { *comm = 0; return 0; }
static inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, int op, MPI_Comm comm) { return 0; }
static inline int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { return 0; }
static inline int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { return 0; }
static inline int MPI_Barrier(MPI_Comm comm) { return 0; }
#endif

#include "quantum_geometric/distributed/distributed_training_manager.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include "quantum_geometric/distributed/gradient_optimizer.h"
#include "quantum_geometric/distributed/communication_optimizer.h"

// Internal state for distributed manager
typedef struct {
    MPI_Comm world_comm;           // Global communicator
    MPI_Comm local_comm;           // Node-local communicator
    MPI_Comm model_comm;           // Model parallel communicator
    MPI_Comm data_comm;            // Data parallel communicator
    void* gradient_buffer;         // Buffer for gradient synchronization
    size_t buffer_size;            // Size of gradient buffer
    size_t current_step;           // Current training step
    bool initialized;              // Whether MPI is initialized
    char* checkpoint_path;         // Path to latest checkpoint
    quantum_pipeline_t* pipeline;  // Reference to active pipeline
} distributed_state_t;

// Forward declarations for wrapper functions (implemented at end of file)
workload_manager_t* workload_manager_create(void);
void workload_manager_destroy(workload_manager_t* manager);
int workload_manager_configure(workload_manager_t* manager, const workload_config_t* config);
gradient_optimizer_t* gradient_optimizer_create(void);
void gradient_optimizer_destroy(gradient_optimizer_t* optimizer);
communication_optimizer_t* communication_optimizer_create(void);
void communication_optimizer_destroy(communication_optimizer_t* optimizer);

distributed_manager_t* distributed_manager_create(const distributed_config_t* config) {
    if (!config) return NULL;
    
    distributed_manager_t* manager = malloc(sizeof(distributed_manager_t));
    if (!manager) return NULL;
    
    // Copy configuration
    memcpy(&manager->config, config, sizeof(distributed_config_t));
    if (config->checkpoint_dir) {
        manager->config.checkpoint_dir = strdup(config->checkpoint_dir);
    }
    
    // Initialize internal state
    distributed_state_t* state = malloc(sizeof(distributed_state_t));
    if (!state) {
        free(manager);
        return NULL;
    }
    memset(state, 0, sizeof(distributed_state_t));
    manager->internal_state = state;
    
    // Create component managers
    manager->workload_manager = workload_manager_create();
    manager->gradient_optimizer = gradient_optimizer_create();
    manager->comm_optimizer = communication_optimizer_create();
    
    if (!manager->workload_manager || !manager->gradient_optimizer || 
        !manager->comm_optimizer) {
        distributed_manager_destroy(manager);
        return NULL;
    }
    
    return manager;
}

void distributed_manager_destroy(distributed_manager_t* manager) {
    if (!manager) return;
    
    distributed_state_t* state = manager->internal_state;
    if (state) {
        if (state->gradient_buffer) {
            free(state->gradient_buffer);
        }
        if (state->checkpoint_path) {
            free(state->checkpoint_path);
        }
        free(state);
    }
    
    if (manager->workload_manager) {
        workload_manager_destroy(manager->workload_manager);
    }
    if (manager->gradient_optimizer) {
        gradient_optimizer_destroy(manager->gradient_optimizer);
    }
    if (manager->comm_optimizer) {
        communication_optimizer_destroy(manager->comm_optimizer);
    }
    
    if (manager->config.checkpoint_dir) {
        free(manager->config.checkpoint_dir);
    }
    
    free(manager);
}

int distributed_manager_init_environment(distributed_manager_t* manager) {
    if (!manager || !manager->internal_state) return -1;
    
    distributed_state_t* state = manager->internal_state;
    int initialized;
    
    // Initialize MPI if not already initialized
    MPI_Initialized(&initialized);
    if (!initialized) {
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS) {
            return -1;
        }
        state->initialized = true;
    }
    
    // Get world communicator
    state->world_comm = MPI_COMM_WORLD;
    
    // Create node-local communicator
    if (MPI_Comm_split_type(state->world_comm, MPI_COMM_TYPE_SHARED, 0,
                           MPI_INFO_NULL, &state->local_comm) != MPI_SUCCESS) {
        return -1;
    }
    
    // Create model and data parallel communicators if needed
    if (manager->config.use_model_parallel) {
        int color = manager->config.local_rank % manager->config.num_gpus_per_node;
        if (MPI_Comm_split(state->world_comm, color, manager->config.local_rank,
                          &state->model_comm) != MPI_SUCCESS) {
            return -1;
        }
    }
    
    if (manager->config.use_data_parallel) {
        int color = manager->config.local_rank / manager->config.num_gpus_per_node;
        if (MPI_Comm_split(state->world_comm, color, manager->config.local_rank,
                          &state->data_comm) != MPI_SUCCESS) {
            return -1;
        }
    }
    
    return 0;
}

int distributed_manager_prepare_training(distributed_manager_t* manager,
                                      quantum_pipeline_t* pipeline,
                                      size_t total_samples) {
    if (!manager || !pipeline) return -1;
    
    // Store pipeline reference
    distributed_state_t* state = manager->internal_state;
    state->pipeline = pipeline;
    
    // Configure workload distribution
    workload_config_t wl_config = {
        .world_size = manager->config.world_size,
        .local_rank = manager->config.local_rank,
        .batch_size = manager->config.batch_size,
        .micro_batch_size = manager->config.micro_batch_size,
        .use_data_parallel = manager->config.use_data_parallel,
        .use_model_parallel = manager->config.use_model_parallel
    };
    
    if (workload_manager_configure(manager->workload_manager, &wl_config) != 0) {
        return -1;
    }
    
    // Allocate gradient buffer
    size_t model_size = dist_pipeline_get_parameter_count(pipeline);
    state->buffer_size = model_size * sizeof(double);
    state->gradient_buffer = malloc(state->buffer_size);
    if (!state->gradient_buffer) {
        return -1;
    }
    
    return 0;
}

int distributed_manager_train_step(distributed_manager_t* manager,
                                 quantum_pipeline_t* pipeline,
                                 const void* batch_data,
                                 size_t batch_size,
                                 size_t step,
                                 training_metrics_t* metrics) {
    if (!manager || !pipeline || !batch_data || !metrics) return -1;
    
    distributed_state_t* state = manager->internal_state;
    state->current_step = step;
    
    // Update learning rate
    float lr = distributed_manager_update_learning_rate(manager, step);
    dist_pipeline_set_learning_rate(pipeline, lr);

    // Forward pass
    if (dist_pipeline_forward(pipeline, batch_data, batch_size) != 0) {
        return -1;
    }

    // Backward pass
    if (dist_pipeline_backward(pipeline) != 0) {
        return -1;
    }
    
    // Synchronize gradients
    if (distributed_manager_sync_gradients(manager, pipeline) != 0) {
        return -1;
    }
    
    // Update parameters
    if (dist_pipeline_update_parameters(pipeline) != 0) {
        return -1;
    }

    // Update metrics
    dist_pipeline_get_metrics(pipeline, metrics);
    
    // Save checkpoint if needed
    if (step % manager->config.save_interval == 0) {
        if (distributed_manager_save_checkpoint(manager, pipeline, step) != 0) {
            return -1;
        }
    }
    
    return 0;
}

int distributed_manager_sync_gradients(distributed_manager_t* manager,
                                     quantum_pipeline_t* pipeline) {
    if (!manager || !pipeline) return -1;
    
    distributed_state_t* state = manager->internal_state;
    
    // Get gradients from pipeline
    if (dist_pipeline_get_gradients(pipeline, state->gradient_buffer,
                                   state->buffer_size) != 0) {
        return -1;
    }
    
    // All-reduce gradients
    if (MPI_Allreduce(MPI_IN_PLACE, state->gradient_buffer, state->buffer_size,
                      MPI_DOUBLE, MPI_SUM, state->world_comm) != MPI_SUCCESS) {
        return -1;
    }
    
    // Scale gradients by world size
    double scale = 1.0 / manager->config.world_size;
    for (size_t i = 0; i < state->buffer_size / sizeof(double); i++) {
        ((double*)state->gradient_buffer)[i] *= scale;
    }
    
    // Set scaled gradients back to pipeline
    if (dist_pipeline_set_gradients(pipeline, state->gradient_buffer,
                                   state->buffer_size) != 0) {
        return -1;
    }
    
    return 0;
}

int distributed_manager_save_checkpoint(distributed_manager_t* manager,
                                      quantum_pipeline_t* pipeline,
                                      size_t step) {
    if (!manager || !pipeline) return -1;
    
    // Only primary process saves checkpoint
    if (!distributed_manager_is_primary(manager)) {
        return 0;
    }
    
    distributed_state_t* state = manager->internal_state;
    char checkpoint_path[1024];
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/checkpoint_%zu",
             manager->config.checkpoint_dir, step);
    
    // Save model state
    if (dist_pipeline_save_state(pipeline, checkpoint_path) != 0) {
        return -1;
    }
    
    // Update latest checkpoint path
    if (state->checkpoint_path) {
        free(state->checkpoint_path);
    }
    state->checkpoint_path = strdup(checkpoint_path);
    
    return 0;
}

int distributed_manager_load_checkpoint(distributed_manager_t* manager,
                                      quantum_pipeline_t* pipeline,
                                      size_t step) {
    if (!manager || !pipeline) return -1;
    
    char checkpoint_path[1024];
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/checkpoint_%zu",
             manager->config.checkpoint_dir, step);
    
    // Load model state
    if (dist_pipeline_load_state(pipeline, checkpoint_path) != 0) {
        return -1;
    }
    
    // Broadcast loaded state to all processes
    distributed_state_t* state = manager->internal_state;
    void* model_buffer = NULL;
    size_t buffer_size = 0;
    
    if (distributed_manager_is_primary(manager)) {
        dist_pipeline_serialize(pipeline, &model_buffer, &buffer_size);
    }
    
    // Broadcast buffer size
    MPI_Bcast(&buffer_size, 1, MPI_UNSIGNED_LONG, 0, state->world_comm);
    
    if (!distributed_manager_is_primary(manager)) {
        model_buffer = malloc(buffer_size);
    }
    
    // Broadcast model state
    MPI_Bcast(model_buffer, buffer_size, MPI_BYTE, 0, state->world_comm);
    
    if (!distributed_manager_is_primary(manager)) {
        dist_pipeline_deserialize(pipeline, model_buffer, buffer_size);
    }
    
    free(model_buffer);
    return 0;
}

int distributed_manager_get_local_batch(distributed_manager_t* manager,
                                      size_t total_samples,
                                      size_t* start_idx,
                                      size_t* end_idx) {
    if (!manager || !start_idx || !end_idx) return -1;
    
    size_t samples_per_rank = total_samples / manager->config.world_size;
    size_t remainder = total_samples % manager->config.world_size;
    
    *start_idx = (size_t)manager->config.local_rank * samples_per_rank;
    *start_idx += ((size_t)manager->config.local_rank < remainder) ?
                  (size_t)manager->config.local_rank : remainder;

    *end_idx = *start_idx + samples_per_rank;
    if ((size_t)manager->config.local_rank < remainder) {
        (*end_idx)++;
    }
    
    return 0;
}

bool distributed_manager_is_primary(const distributed_manager_t* manager) {
    return manager && manager->config.local_rank == 0;
}

float distributed_manager_update_learning_rate(distributed_manager_t* manager,
                                            size_t step) {
    if (!manager) return 0.0f;
    
    float lr = manager->config.learning_rate;
    
    // Apply warmup
    if (step < manager->config.warmup_steps) {
        lr *= (float)step / manager->config.warmup_steps;
    }
    
    // Apply decay
    if (step > manager->config.warmup_steps) {
        float progress = (float)(step - manager->config.warmup_steps) /
                        (manager->config.max_steps - manager->config.warmup_steps);
        lr *= (1.0f - progress);
    }
    
    return lr;
}

// ============================================================================
// Wrapper function implementations for component managers
// ============================================================================

// Type alias for WorkloadManager as workload_manager_t
typedef WorkloadManager workload_manager_t_impl;

workload_manager_t* workload_manager_create(void) {
    return (workload_manager_t*)init_workload_manager();
}

void workload_manager_destroy(workload_manager_t* manager) {
    if (manager) {
        cleanup_workload_manager((WorkloadManager*)manager);
    }
}

int workload_manager_configure(workload_manager_t* manager, const workload_config_t* config) {
    if (!manager || !config) return -1;

    WorkloadManager* wm = (WorkloadManager*)manager;
    wm->rank = config->local_rank;
    wm->world_size = config->world_size;

    // Additional configuration could be applied here
    // For now, basic configuration is sufficient
    return 0;
}

// gradient_optimizer_create() - Canonical implementation in gradient_optimizer.c
// (removed: this was a simple wrapper, use init_gradient_optimizer directly)

// gradient_optimizer_destroy() - Canonical implementation in gradient_optimizer.c
// (removed: this was a simple wrapper, use cleanup_gradient_optimizer directly)

communication_optimizer_t* communication_optimizer_create(void) {
    // Create with default configuration
    CommConfig default_config = {
        .buffer_size = 64 * 1024 * 1024,  // 64MB
        .min_message_size = 4096,
        .max_concurrent = 4,
        .enable_compression = false,
        .enable_topology_aware = true,
        .use_pinned_memory = false,
        .numa_aware = true,
        .topology_aware = true,
        .numa_policy = 0
    };
    return (communication_optimizer_t*)init_communication_optimizer(&default_config);
}

void communication_optimizer_destroy(communication_optimizer_t* optimizer) {
    if (optimizer) {
        cleanup_communication_optimizer((CommunicationOptimizer*)optimizer);
    }
}

// ============================================================================
// Failure handling
// ============================================================================

int distributed_manager_handle_failure(distributed_manager_t* manager,
                                     size_t failed_rank) {
    if (!manager) return -1;
    
    distributed_state_t* state = manager->internal_state;
    int rank, size;
    MPI_Comm_rank(state->world_comm, &rank);
    MPI_Comm_size(state->world_comm, &size);
    
    // Step 1: Detect process failure using MPI_Iprobe
    int err_code;
    MPI_Status status;
    int flag = 0;

    // Check if the suspected failed rank can respond
    // Use MPI_Iprobe to check for any pending messages from the rank
    err_code = MPI_Iprobe((int)failed_rank, MPI_ANY_TAG, state->world_comm, &flag, &status);

    // Also send heartbeat to failed rank
    err_code = MPI_Send(&rank, 1, MPI_INT, (int)failed_rank, 0, state->world_comm);

    // If probe found a message, the rank might still be alive - check its source
    if (flag && status.MPI_SOURCE == (int)failed_rank) {
        // Rank responded, not actually failed
        return 0;
    }

    if (err_code != MPI_SUCCESS) {
        // Confirmed failure, proceed with recovery
        
        // Step 2: Notify all processes about failure
        int failure_detected = 1;
        MPI_Allreduce(MPI_IN_PLACE, &failure_detected, 1, MPI_INT, MPI_MAX,
                      state->world_comm);
        
        if (failure_detected) {
            // Step 3: Reconstruct communicators
            
            // Create new world communicator excluding failed rank
            MPI_Comm new_world_comm;
            int color = ((size_t)rank != failed_rank) ? 0 : MPI_UNDEFINED;
            err_code = MPI_Comm_split(state->world_comm, color, rank,
                                    &new_world_comm);
            if (err_code != MPI_SUCCESS) {
                return -1;
            }
            
            // Free old communicators
            if (state->local_comm != MPI_COMM_NULL) {
                MPI_Comm_free(&state->local_comm);
            }
            if (state->model_comm != MPI_COMM_NULL) {
                MPI_Comm_free(&state->model_comm);
            }
            if (state->data_comm != MPI_COMM_NULL) {
                MPI_Comm_free(&state->data_comm);
            }
            
            // Update world communicator
            state->world_comm = new_world_comm;
            
            // Create new node-local communicator
            err_code = MPI_Comm_split_type(state->world_comm,
                                         MPI_COMM_TYPE_SHARED, 0,
                                         MPI_INFO_NULL,
                                         &state->local_comm);
            if (err_code != MPI_SUCCESS) {
                return -1;
            }
            
            // Recreate model and data parallel communicators if needed
            if (manager->config.use_model_parallel) {
                int color = manager->config.local_rank % 
                           manager->config.num_gpus_per_node;
                err_code = MPI_Comm_split(state->world_comm, color,
                                        manager->config.local_rank,
                                        &state->model_comm);
                if (err_code != MPI_SUCCESS) {
                    return -1;
                }
            }
            
            if (manager->config.use_data_parallel) {
                int color = manager->config.local_rank / 
                           manager->config.num_gpus_per_node;
                err_code = MPI_Comm_split(state->world_comm, color,
                                        manager->config.local_rank,
                                        &state->data_comm);
                if (err_code != MPI_SUCCESS) {
                    return -1;
                }
            }
            
            // Step 4: Update configuration
            manager->config.world_size--;
            if ((size_t)rank > failed_rank) {
                manager->config.local_rank--;
            }
            
            // Step 5: Reload from last checkpoint
            if (state->checkpoint_path && state->pipeline) {
                if (distributed_manager_load_checkpoint(manager, state->pipeline,
                                                     state->current_step) != 0) {
                    return -1;
                }
            }
            
            // Step 6: Redistribute workload
            workload_config_t wl_config = {
                .world_size = manager->config.world_size,
                .local_rank = manager->config.local_rank,
                .batch_size = manager->config.batch_size,
                .micro_batch_size = manager->config.micro_batch_size,
                .use_data_parallel = manager->config.use_data_parallel,
                .use_model_parallel = manager->config.use_model_parallel
            };
            
            if (workload_manager_configure(manager->workload_manager,
                                         &wl_config) != 0) {
                return -1;
            }
            
            // Step 7: Synchronize state
            MPI_Barrier(state->world_comm);
            
            return 0;
        }
    }
    
    // No failure detected
    return 0;
}


