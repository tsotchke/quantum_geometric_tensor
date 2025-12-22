#ifndef TRAINING_ORCHESTRATOR_H
#define TRAINING_ORCHESTRATOR_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Definitions for Distributed Training
// ============================================================================

// Device type enumeration
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_GPU_CUDA,
    DEVICE_GPU_METAL,
    DEVICE_QPU,
    DEVICE_FPGA
} DeviceType;

// Forward declarations
struct TransformerLayer;
struct CommunicationManager;

// Training statistics
typedef struct {
    size_t total_steps;
    size_t current_epoch;
    double loss;
    double accuracy;
    double throughput;              // samples/second
    double communication_time;      // seconds spent in communication
    double computation_time;        // seconds spent in computation
    size_t gradient_sync_count;
    size_t checkpoint_count;
} TrainingStats;

// Training configuration
typedef struct {
    size_t batch_size;
    size_t micro_batch_size;
    size_t num_epochs;
    double learning_rate;
    double weight_decay;
    bool use_gradient_compression;
    bool use_mixed_precision;
    size_t gradient_accumulation_steps;
    size_t checkpoint_interval;
    const char* checkpoint_path;
} TrainingOrchestratorConfig;

// Gradient buffer for distributed training
typedef struct {
    float* data;
    size_t size;
    size_t capacity;
    bool is_compressed;
    double compression_ratio;
} DistributedGradientBuffer;

// Pipeline stage for model parallelism
typedef struct {
    struct TransformerLayer* layer;
    int node_rank;
    int gpu_id;
    size_t micro_batch_size;
    void* input_buffer;
    void* output_buffer;
    void* gradient_buffer;
} PipelineStage;

// Node configuration for distributed setup
typedef struct {
    int rank;
    int world_size;
    int local_rank;
    int num_gpus;
    bool is_master;
    DeviceType* devices;
    size_t num_devices;
} DistributedNodeConfig;

// Training orchestrator handle (opaque)
typedef struct TrainingOrchestrator TrainingOrchestrator;

// ============================================================================
// Function Declarations
// ============================================================================

#ifndef NO_MPI

// Initialization and cleanup
TrainingOrchestrator* init_training_orchestrator(const TrainingOrchestratorConfig* config);
void cleanup_training_orchestrator(TrainingOrchestrator* orchestrator);

// Training control
bool start_training(TrainingOrchestrator* orchestrator, void* model, void* dataset);
bool pause_training(TrainingOrchestrator* orchestrator);
bool resume_training(TrainingOrchestrator* orchestrator);
void stop_training(TrainingOrchestrator* orchestrator);

// Distributed operations
bool synchronize_gradients(TrainingOrchestrator* orchestrator);
bool broadcast_parameters(TrainingOrchestrator* orchestrator, void* params, size_t size);
bool all_reduce_gradients(TrainingOrchestrator* orchestrator, DistributedGradientBuffer* gradients);

// Checkpointing
bool save_checkpoint(TrainingOrchestrator* orchestrator, const char* path);
bool load_checkpoint(TrainingOrchestrator* orchestrator, const char* path);

// Statistics and monitoring
TrainingStats get_training_stats(const TrainingOrchestrator* orchestrator);
void reset_training_stats(TrainingOrchestrator* orchestrator);
double get_current_throughput(const TrainingOrchestrator* orchestrator);

// Pipeline parallelism
bool setup_pipeline_stages(TrainingOrchestrator* orchestrator, size_t num_stages);
bool execute_pipeline_step(TrainingOrchestrator* orchestrator);

// Node management
DistributedNodeConfig get_node_config(const TrainingOrchestrator* orchestrator);
bool is_master_node(const TrainingOrchestrator* orchestrator);
int orchestrator_get_world_size(const TrainingOrchestrator* orchestrator);
int orchestrator_get_rank(const TrainingOrchestrator* orchestrator);

#else

// Stub implementations when MPI is not available
static inline TrainingOrchestrator* init_training_orchestrator(const TrainingOrchestratorConfig* config) {
    (void)config;
    return NULL;
}

static inline void cleanup_training_orchestrator(TrainingOrchestrator* orchestrator) {
    (void)orchestrator;
}

#endif // NO_MPI

#ifdef __cplusplus
}
#endif

#endif // TRAINING_ORCHESTRATOR_H
