#ifndef DISTRIBUTED_TRAINING_MANAGER_H
#define DISTRIBUTED_TRAINING_MANAGER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types
typedef struct distributed_manager_t distributed_manager_t;
typedef struct workload_manager_t workload_manager_t;
typedef struct communication_optimizer_t communication_optimizer_t;
typedef struct quantum_pipeline_t quantum_pipeline_t;

// Include gradient optimizer header for consistent type definition
#include "quantum_geometric/distributed/gradient_optimizer.h"

// Configuration for distributed training
typedef struct distributed_config_t {
    int world_size;              // Total number of processes
    int local_rank;              // This process's rank
    int num_gpus_per_node;       // GPUs available per node
    size_t batch_size;           // Global batch size
    size_t micro_batch_size;     // Micro-batch size for gradient accumulation
    float learning_rate;         // Initial learning rate
    size_t warmup_steps;         // Steps for learning rate warmup
    size_t max_steps;            // Maximum training steps
    size_t save_interval;        // Checkpoint save frequency
    bool use_model_parallel;     // Enable model parallelism
    bool use_data_parallel;      // Enable data parallelism
    bool use_mixed_precision;    // Enable FP16/BF16 training
    bool use_gradient_checkpointing;  // Memory optimization
    char* checkpoint_dir;        // Directory for checkpoints
} distributed_config_t;

// Training metrics structure
typedef struct training_metrics_t {
    float loss;                  // Current loss value
    float accuracy;              // Current accuracy
    float learning_rate;         // Current learning rate
    double throughput;           // Samples per second
    size_t step;                 // Current step number
    size_t epoch;                // Current epoch number
    double memory_used;          // GPU memory used (bytes)
    double communication_time;   // Time spent in communication (seconds)
    double compute_time;         // Time spent in compute (seconds)
} training_metrics_t;

// Workload configuration
typedef struct workload_config_t {
    int world_size;
    int local_rank;
    size_t batch_size;
    size_t micro_batch_size;
    bool use_data_parallel;
    bool use_model_parallel;
} workload_config_t;

// Distributed manager structure
struct distributed_manager_t {
    distributed_config_t config;
    void* internal_state;                        // Opaque internal state
    workload_manager_t* workload_manager;        // Workload distribution
    gradient_optimizer_t* gradient_optimizer;    // Gradient optimization
    communication_optimizer_t* comm_optimizer;   // Communication optimization
};

// Create and destroy
distributed_manager_t* distributed_manager_create(const distributed_config_t* config);
void distributed_manager_destroy(distributed_manager_t* manager);

// Environment initialization
int distributed_manager_init_environment(distributed_manager_t* manager);

// Training preparation
int distributed_manager_prepare_training(distributed_manager_t* manager,
                                        quantum_pipeline_t* pipeline,
                                        size_t total_samples);

// Training step execution
int distributed_manager_train_step(distributed_manager_t* manager,
                                  quantum_pipeline_t* pipeline,
                                  const void* batch_data,
                                  size_t batch_size,
                                  size_t step,
                                  training_metrics_t* metrics);

// Gradient synchronization
int distributed_manager_sync_gradients(distributed_manager_t* manager,
                                      quantum_pipeline_t* pipeline);

// Checkpoint management
int distributed_manager_save_checkpoint(distributed_manager_t* manager,
                                       quantum_pipeline_t* pipeline,
                                       size_t step);
int distributed_manager_load_checkpoint(distributed_manager_t* manager,
                                       quantum_pipeline_t* pipeline,
                                       size_t step);

// Batch partitioning
int distributed_manager_get_local_batch(distributed_manager_t* manager,
                                       size_t total_samples,
                                       size_t* start_idx,
                                       size_t* end_idx);

// Utility functions
bool distributed_manager_is_primary(const distributed_manager_t* manager);
float distributed_manager_update_learning_rate(distributed_manager_t* manager, size_t step);

// Failure handling and elastic scaling
int distributed_manager_handle_failure(distributed_manager_t* manager, size_t failed_rank);

// Workload manager functions
workload_manager_t* workload_manager_create(void);
void workload_manager_destroy(workload_manager_t* manager);
int workload_manager_configure(workload_manager_t* manager, const workload_config_t* config);

// Gradient optimizer functions
gradient_optimizer_t* gradient_optimizer_create(void);
void gradient_optimizer_destroy(gradient_optimizer_t* optimizer);

// Communication optimizer functions
communication_optimizer_t* communication_optimizer_create(void);
void communication_optimizer_destroy(communication_optimizer_t* optimizer);

// Distributed pipeline functions (separate from public quantum_pipeline.h API)
size_t dist_pipeline_get_parameter_count(quantum_pipeline_t* pipeline);
int dist_pipeline_forward(quantum_pipeline_t* pipeline, const void* data, size_t batch_size);
int dist_pipeline_backward(quantum_pipeline_t* pipeline);
int dist_pipeline_update_parameters(quantum_pipeline_t* pipeline);
int dist_pipeline_get_gradients(quantum_pipeline_t* pipeline, void* buffer, size_t size);
int dist_pipeline_set_gradients(quantum_pipeline_t* pipeline, void* buffer, size_t size);
void dist_pipeline_get_metrics(quantum_pipeline_t* pipeline, training_metrics_t* metrics);
void dist_pipeline_set_learning_rate(quantum_pipeline_t* pipeline, float lr);
int dist_pipeline_save_state(quantum_pipeline_t* pipeline, const char* path);
int dist_pipeline_load_state(quantum_pipeline_t* pipeline, const char* path);
int dist_pipeline_serialize(quantum_pipeline_t* pipeline, void** buffer, size_t* size);
int dist_pipeline_deserialize(quantum_pipeline_t* pipeline, void* buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBUTED_TRAINING_MANAGER_H
