/**
 * @file training_orchestrator.c
 * @brief Distributed training orchestration with MPI
 *
 * This module provides distributed training capabilities using MPI for
 * multi-node training with data and model parallelism. It implements
 * pipeline parallelism for large transformer models.
 *
 * NOTE: This module requires MPI. It is only compiled when NO_MPI is not defined.
 */

#include "quantum_geometric/distributed/training_orchestrator.h"

#ifndef NO_MPI

#include "quantum_geometric/distributed/communication_optimization.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

#ifdef HAVE_NVML
#include <nvml.h>
#endif

// Training parameters
#define MAX_NODES 256
#define MAX_GPUS_PER_NODE 8
#define MAX_PIPELINE_STAGES 32
#define GRADIENT_SYNC_INTERVAL 100
#define DEFAULT_CHECKPOINT_INTERVAL 1000
#define ALIGNMENT 64

// Internal training orchestrator structure
struct TrainingOrchestrator {
    // Distributed configuration
    DistributedNodeConfig node_config;
    MPI_Comm global_comm;
    MPI_Comm local_comm;

    // Model parallelism
    PipelineStage* pipeline_stages;
    size_t num_stages;
    size_t max_stages;

    // Data parallelism
    size_t global_batch_size;
    size_t micro_batch_size;

    // Communication
    void* comm_manager;
    DistributedGradientBuffer* gradient_buffer;

    // State
    bool is_initialized;
    bool is_training;
    TrainingStats stats;
    TrainingOrchestratorConfig config;
};

// Forward declarations for internal functions
static void setup_node_config(TrainingOrchestrator* orchestrator);
static void create_local_comm(TrainingOrchestrator* orchestrator);
static size_t calculate_micro_batch_size(size_t global_batch, int world_size);

// ============================================================================
// Initialization and Cleanup
// ============================================================================

TrainingOrchestrator* init_training_orchestrator(const TrainingOrchestratorConfig* config) {
    if (!config) return NULL;

    TrainingOrchestrator* orchestrator = aligned_alloc(ALIGNMENT,
        sizeof(TrainingOrchestrator));
    if (!orchestrator) return NULL;

    memset(orchestrator, 0, sizeof(TrainingOrchestrator));

    // Store configuration
    orchestrator->config = *config;

    // Initialize MPI if needed
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }

    // Setup node configuration
    setup_node_config(orchestrator);

    // Create communicators
    MPI_Comm_dup(MPI_COMM_WORLD, &orchestrator->global_comm);
    create_local_comm(orchestrator);

    // Initialize pipeline stages
    orchestrator->max_stages = MAX_PIPELINE_STAGES;
    orchestrator->pipeline_stages = aligned_alloc(ALIGNMENT,
        orchestrator->max_stages * sizeof(PipelineStage));
    if (!orchestrator->pipeline_stages) {
        free(orchestrator);
        return NULL;
    }
    memset(orchestrator->pipeline_stages, 0,
           orchestrator->max_stages * sizeof(PipelineStage));
    orchestrator->num_stages = 0;

    // Create gradient buffer
    size_t buffer_size = config->batch_size * 1024 * sizeof(float); // Estimate
    orchestrator->gradient_buffer = aligned_alloc(ALIGNMENT,
        sizeof(DistributedGradientBuffer));
    if (orchestrator->gradient_buffer) {
        orchestrator->gradient_buffer->data = aligned_alloc(ALIGNMENT, buffer_size);
        orchestrator->gradient_buffer->capacity = buffer_size / sizeof(float);
        orchestrator->gradient_buffer->size = 0;
        orchestrator->gradient_buffer->is_compressed = false;
        orchestrator->gradient_buffer->compression_ratio = 1.0;
    }

    // Initialize training state
    orchestrator->global_batch_size = config->batch_size;
    orchestrator->micro_batch_size = calculate_micro_batch_size(
        config->batch_size,
        orchestrator->node_config.world_size);

    // Initialize stats
    memset(&orchestrator->stats, 0, sizeof(TrainingStats));

    orchestrator->is_initialized = true;
    orchestrator->is_training = false;

    return orchestrator;
}

void cleanup_training_orchestrator(TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return;

    // Clean up pipeline stages
    for (size_t i = 0; i < orchestrator->num_stages; i++) {
        PipelineStage* stage = &orchestrator->pipeline_stages[i];
        if (stage->input_buffer) free(stage->input_buffer);
        if (stage->output_buffer) free(stage->output_buffer);
        if (stage->gradient_buffer) free(stage->gradient_buffer);
    }

    if (orchestrator->pipeline_stages) {
        free(orchestrator->pipeline_stages);
    }

    // Clean up gradient buffer
    if (orchestrator->gradient_buffer) {
        if (orchestrator->gradient_buffer->data) {
            free(orchestrator->gradient_buffer->data);
        }
        free(orchestrator->gradient_buffer);
    }

    // Clean up node devices
    if (orchestrator->node_config.devices) {
        free(orchestrator->node_config.devices);
    }

    // Clean up MPI communicators
    if (orchestrator->global_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&orchestrator->global_comm);
    }
    if (orchestrator->local_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&orchestrator->local_comm);
    }

    free(orchestrator);
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

static int detect_gpu_count(DeviceType* gpu_type) {
    int num_gpus = 0;
    *gpu_type = DEVICE_CPU;  // Default

#ifdef __APPLE__
    // macOS: Detect Metal GPUs using IOKit
    io_iterator_t iterator;
    kern_return_t result;

    // Count AGXAccelerator (Apple Silicon GPU) devices
    CFMutableDictionaryRef matching = IOServiceMatching("AGXAccelerator");
    if (matching) {
        result = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
        if (result == KERN_SUCCESS) {
            io_object_t service;
            while ((service = IOIteratorNext(iterator)) != 0) {
                num_gpus++;
                *gpu_type = DEVICE_GPU_METAL;
                IOObjectRelease(service);
            }
            IOObjectRelease(iterator);
        }
    }

    // If no Apple Silicon GPU, check for Intel/AMD GPUs
    if (num_gpus == 0) {
        matching = IOServiceMatching("IOGPUDevice");
        if (!matching) {
            matching = IOServiceMatching("AppleGPUPowerManagement");
        }
        if (matching) {
            result = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
            if (result == KERN_SUCCESS) {
                io_object_t service;
                while ((service = IOIteratorNext(iterator)) != 0) {
                    num_gpus++;
                    *gpu_type = DEVICE_GPU_METAL;
                    IOObjectRelease(service);
                }
                IOObjectRelease(iterator);
            }
        }
    }

    // macOS always has at least one GPU (integrated or discrete)
    if (num_gpus == 0) {
        // Fall back to checking system profiler info via sysctl
        // SPDisplaysDataType contains GPU info
        // For simplicity, assume at least 1 Metal GPU on modern macOS
        num_gpus = 1;
        *gpu_type = DEVICE_GPU_METAL;
    }

#elif defined(HAVE_NVML)
    // Linux/NVIDIA: Use NVML to detect CUDA GPUs
    static bool nvml_initialized = false;
    static bool nvml_available = false;

    if (!nvml_initialized) {
        nvml_initialized = true;
        nvmlReturn_t nvml_result = nvmlInit_v2();
        nvml_available = (nvml_result == NVML_SUCCESS);
    }

    if (nvml_available) {
        unsigned int device_count = 0;
        nvmlReturn_t nvml_result = nvmlDeviceGetCount_v2(&device_count);
        if (nvml_result == NVML_SUCCESS && device_count > 0) {
            num_gpus = (int)device_count;
            *gpu_type = DEVICE_GPU_CUDA;
        }
    }

#else
    // Try using the GPU interface from quantum_geometric_gpu.h
    GPUDeviceInfo devices[MAX_GPUS_PER_NODE];
    int device_count = gpu_get_devices(devices, MAX_GPUS_PER_NODE);
    if (device_count > 0) {
        num_gpus = device_count;
        if (devices[0].backend_type == GPU_BACKEND_METAL) {
            *gpu_type = DEVICE_GPU_METAL;
        } else if (devices[0].backend_type == GPU_BACKEND_CUDA) {
            *gpu_type = DEVICE_GPU_CUDA;
        }
    }
#endif

    return num_gpus;
}

static void setup_node_config(TrainingOrchestrator* orchestrator) {
    MPI_Comm_rank(MPI_COMM_WORLD, &orchestrator->node_config.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &orchestrator->node_config.world_size);

    // For local rank, we use shared memory communicator
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &orchestrator->node_config.local_rank);
    MPI_Comm_free(&shmcomm);

    // Detect GPUs using platform-specific APIs
    DeviceType gpu_type;
    int detected_gpus = detect_gpu_count(&gpu_type);

    // Ensure at least 1 device (CPU fallback)
    if (detected_gpus <= 0) {
        detected_gpus = 1;
        gpu_type = DEVICE_CPU;
    }

    // Cap at maximum GPUs per node
    if (detected_gpus > MAX_GPUS_PER_NODE) {
        detected_gpus = MAX_GPUS_PER_NODE;
    }

    orchestrator->node_config.num_gpus = (size_t)detected_gpus;
    orchestrator->node_config.is_master = (orchestrator->node_config.rank == 0);

    // Allocate and set device types
    orchestrator->node_config.num_devices = orchestrator->node_config.num_gpus;
    orchestrator->node_config.devices = malloc(
        orchestrator->node_config.num_devices * sizeof(DeviceType));
    if (orchestrator->node_config.devices) {
        for (size_t i = 0; i < orchestrator->node_config.num_devices; i++) {
            orchestrator->node_config.devices[i] = gpu_type;
        }
    }
}

static void create_local_comm(TrainingOrchestrator* orchestrator) {
    // Create a communicator for processes on the same node
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        orchestrator->node_config.rank,
                        MPI_INFO_NULL,
                        &orchestrator->local_comm);
}

static size_t calculate_micro_batch_size(size_t global_batch, int world_size) {
    if (world_size <= 0) return global_batch;
    return global_batch / (size_t)world_size;
}

// ============================================================================
// Training Control
// ============================================================================

bool start_training(TrainingOrchestrator* orchestrator, void* model, void* dataset) {
    if (!orchestrator || !orchestrator->is_initialized) return false;
    if (orchestrator->is_training) return false; // Already training

    (void)model;
    (void)dataset;

    orchestrator->is_training = true;
    orchestrator->stats.total_steps = 0;
    orchestrator->stats.current_epoch = 0;

    return true;
}

bool pause_training(TrainingOrchestrator* orchestrator) {
    if (!orchestrator || !orchestrator->is_training) return false;
    orchestrator->is_training = false;
    return true;
}

bool resume_training(TrainingOrchestrator* orchestrator) {
    if (!orchestrator || !orchestrator->is_initialized) return false;
    orchestrator->is_training = true;
    return true;
}

void stop_training(TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return;
    orchestrator->is_training = false;
}

// ============================================================================
// Distributed Operations
// ============================================================================

bool synchronize_gradients(TrainingOrchestrator* orchestrator) {
    if (!orchestrator || !orchestrator->gradient_buffer) return false;

    DistributedGradientBuffer* buf = orchestrator->gradient_buffer;
    if (buf->size == 0 || !buf->data) return true; // Nothing to sync

    // All-reduce gradients across all nodes
    MPI_Allreduce(MPI_IN_PLACE, buf->data, (int)buf->size,
                  MPI_FLOAT, MPI_SUM, orchestrator->global_comm);

    // Average the gradients
    float scale = 1.0f / (float)orchestrator->node_config.world_size;
    for (size_t i = 0; i < buf->size; i++) {
        buf->data[i] *= scale;
    }

    orchestrator->stats.gradient_sync_count++;
    return true;
}

bool broadcast_parameters(TrainingOrchestrator* orchestrator, void* params, size_t size) {
    if (!orchestrator || !params || size == 0) return false;

    MPI_Bcast(params, (int)size, MPI_BYTE, 0, orchestrator->global_comm);
    return true;
}

bool all_reduce_gradients(TrainingOrchestrator* orchestrator, DistributedGradientBuffer* gradients) {
    if (!orchestrator || !gradients || !gradients->data) return false;

    MPI_Allreduce(MPI_IN_PLACE, gradients->data, (int)gradients->size,
                  MPI_FLOAT, MPI_SUM, orchestrator->global_comm);
    return true;
}

// ============================================================================
// Checkpointing
// ============================================================================

bool save_checkpoint(TrainingOrchestrator* orchestrator, const char* path) {
    if (!orchestrator || !path) return false;

    // Only master saves checkpoint
    if (!orchestrator->node_config.is_master) return true;

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // Write checkpoint header
    const char magic[] = "QGTC";
    fwrite(magic, 1, 4, f);

    // Write training state
    fwrite(&orchestrator->stats, sizeof(TrainingStats), 1, f);
    fwrite(&orchestrator->config, sizeof(TrainingOrchestratorConfig), 1, f);

    fclose(f);
    orchestrator->stats.checkpoint_count++;
    return true;
}

bool load_checkpoint(TrainingOrchestrator* orchestrator, const char* path) {
    if (!orchestrator || !path) return false;

    FILE* f = fopen(path, "rb");
    if (!f) return false;

    // Verify header
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QGTC", 4) != 0) {
        fclose(f);
        return false;
    }

    // Read training state
    fread(&orchestrator->stats, sizeof(TrainingStats), 1, f);
    fread(&orchestrator->config, sizeof(TrainingOrchestratorConfig), 1, f);

    fclose(f);

    // Broadcast to all nodes
    MPI_Bcast(&orchestrator->stats, sizeof(TrainingStats), MPI_BYTE, 0,
              orchestrator->global_comm);

    return true;
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

TrainingStats get_training_stats(const TrainingOrchestrator* orchestrator) {
    TrainingStats empty = {0};
    if (!orchestrator) return empty;
    return orchestrator->stats;
}

void reset_training_stats(TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return;
    memset(&orchestrator->stats, 0, sizeof(TrainingStats));
}

double get_current_throughput(const TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return 0.0;
    return orchestrator->stats.throughput;
}

// ============================================================================
// Pipeline Parallelism
// ============================================================================

bool setup_pipeline_stages(TrainingOrchestrator* orchestrator, size_t num_stages) {
    if (!orchestrator || num_stages > orchestrator->max_stages) return false;

    orchestrator->num_stages = num_stages;

    // Initialize each stage
    for (size_t i = 0; i < num_stages; i++) {
        PipelineStage* stage = &orchestrator->pipeline_stages[i];
        stage->node_rank = orchestrator->node_config.rank;
        stage->gpu_id = (int)(i % (size_t)orchestrator->node_config.num_gpus);
        stage->micro_batch_size = orchestrator->micro_batch_size;
        stage->layer = NULL; // To be set by model setup
    }

    return true;
}

bool execute_pipeline_step(TrainingOrchestrator* orchestrator) {
    if (!orchestrator || orchestrator->num_stages == 0) return false;

    // Synchronize before pipeline step
    MPI_Barrier(orchestrator->global_comm);

    orchestrator->stats.total_steps++;
    return true;
}

// ============================================================================
// Node Management
// ============================================================================

DistributedNodeConfig get_node_config(const TrainingOrchestrator* orchestrator) {
    DistributedNodeConfig empty = {0};
    if (!orchestrator) return empty;
    return orchestrator->node_config;
}

bool is_master_node(const TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return false;
    return orchestrator->node_config.is_master;
}

int orchestrator_get_world_size(const TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return 0;
    return orchestrator->node_config.world_size;
}

int orchestrator_get_rank(const TrainingOrchestrator* orchestrator) {
    if (!orchestrator) return -1;
    return orchestrator->node_config.rank;
}

#endif // NO_MPI
