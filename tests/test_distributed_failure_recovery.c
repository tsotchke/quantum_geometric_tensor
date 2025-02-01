/**
 * @file test_distributed_failure_recovery.c
 * @brief Test suite for distributed training failure recovery
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "quantum_geometric/distributed/distributed_training_manager.h"
#include "quantum_geometric/core/quantum_geometric_core.h"

// Mock pipeline for testing
typedef struct {
    size_t num_parameters;
    double* parameters;
    double* gradients;
    bool initialized;
} mock_pipeline_t;

static mock_pipeline_t* mock_pipeline_create(size_t num_parameters) {
    mock_pipeline_t* pipeline = malloc(sizeof(mock_pipeline_t));
    if (!pipeline) return NULL;
    
    pipeline->num_parameters = num_parameters;
    pipeline->parameters = malloc(num_parameters * sizeof(double));
    pipeline->gradients = malloc(num_parameters * sizeof(double));
    
    if (!pipeline->parameters || !pipeline->gradients) {
        free(pipeline->parameters);
        free(pipeline->gradients);
        free(pipeline);
        return NULL;
    }
    
    // Initialize with test values
    for (size_t i = 0; i < num_parameters; i++) {
        pipeline->parameters[i] = (double)i;
        pipeline->gradients[i] = 0.0;
    }
    
    pipeline->initialized = true;
    return pipeline;
}

static void mock_pipeline_destroy(mock_pipeline_t* pipeline) {
    if (!pipeline) return;
    free(pipeline->parameters);
    free(pipeline->gradients);
    free(pipeline);
}

// Test failure detection
static void test_failure_detection(void) {
    printf("Testing failure detection...\n");
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create test configuration
    distributed_config_t config = {
        .world_size = size,
        .local_rank = rank,
        .num_gpus_per_node = 2,
        .batch_size = 32,
        .micro_batch_size = 8,
        .use_data_parallel = true,
        .use_model_parallel = false,
        .use_pipeline_parallel = false,
        .use_gradient_checkpointing = true,
        .learning_rate = 0.001f,
        .warmup_steps = 100,
        .max_steps = 1000,
        .save_interval = 50,
        .checkpoint_dir = "/tmp/test_checkpoints"
    };
    
    // Create manager
    distributed_manager_t* manager = distributed_manager_create(&config);
    assert(manager != NULL && "Manager creation failed");
    
    // Initialize environment
    int result = distributed_manager_init_environment(manager);
    assert(result == 0 && "Environment initialization failed");
    
    // Simulate failure of rank 1
    if (size > 1) {
        result = distributed_manager_handle_failure(manager, 1);
        assert(result == 0 && "Failure handling failed");
        
        // Verify world size update
        if (rank != 1) {
            assert(manager->config.world_size == size - 1);
            if (rank > 1) {
                assert(manager->config.local_rank == rank - 1);
            }
        }
    }
    
    distributed_manager_destroy(manager);
    printf("Failure detection test passed on rank %d\n", rank);
}

// Test communicator reconstruction
static void test_communicator_reconstruction(void) {
    printf("Testing communicator reconstruction...\n");
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create test configuration
    distributed_config_t config = {
        .world_size = size,
        .local_rank = rank,
        .num_gpus_per_node = 2,
        .batch_size = 32,
        .micro_batch_size = 8,
        .use_data_parallel = true,
        .use_model_parallel = true,  // Test both types
        .use_pipeline_parallel = false,
        .use_gradient_checkpointing = true,
        .learning_rate = 0.001f,
        .warmup_steps = 100,
        .max_steps = 1000,
        .save_interval = 50,
        .checkpoint_dir = "/tmp/test_checkpoints"
    };
    
    // Create manager
    distributed_manager_t* manager = distributed_manager_create(&config);
    assert(manager != NULL && "Manager creation failed");
    
    // Initialize environment
    int result = distributed_manager_init_environment(manager);
    assert(result == 0 && "Environment initialization failed");
    
    // Simulate failure and verify communicator reconstruction
    if (size > 1) {
        result = distributed_manager_handle_failure(manager, 1);
        assert(result == 0 && "Failure handling failed");
        
        // Verify communicators are valid
        if (rank != 1) {
            MPI_Comm_size(manager->internal_state->world_comm, &size);
            assert(size == config.world_size - 1);
            
            int local_size;
            MPI_Comm_size(manager->internal_state->local_comm, &local_size);
            assert(local_size > 0);
            
            if (config.use_model_parallel) {
                int model_size;
                MPI_Comm_size(manager->internal_state->model_comm, &model_size);
                assert(model_size > 0);
            }
            
            if (config.use_data_parallel) {
                int data_size;
                MPI_Comm_size(manager->internal_state->data_comm, &data_size);
                assert(data_size > 0);
            }
        }
    }
    
    distributed_manager_destroy(manager);
    printf("Communicator reconstruction test passed on rank %d\n", rank);
}

// Test checkpoint recovery
static void test_checkpoint_recovery(void) {
    printf("Testing checkpoint recovery...\n");
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create test configuration
    distributed_config_t config = {
        .world_size = size,
        .local_rank = rank,
        .num_gpus_per_node = 2,
        .batch_size = 32,
        .micro_batch_size = 8,
        .use_data_parallel = true,
        .use_model_parallel = false,
        .use_pipeline_parallel = false,
        .use_gradient_checkpointing = true,
        .learning_rate = 0.001f,
        .warmup_steps = 100,
        .max_steps = 1000,
        .save_interval = 50,
        .checkpoint_dir = "/tmp/test_checkpoints"
    };
    
    // Create manager and mock pipeline
    distributed_manager_t* manager = distributed_manager_create(&config);
    assert(manager != NULL && "Manager creation failed");
    
    mock_pipeline_t* pipeline = mock_pipeline_create(1000);
    assert(pipeline != NULL && "Pipeline creation failed");
    
    // Initialize environment
    int result = distributed_manager_init_environment(manager);
    assert(result == 0 && "Environment initialization failed");
    
    // Prepare training
    result = distributed_manager_prepare_training(manager,
                                               (quantum_pipeline_t*)pipeline,
                                               1000);
    assert(result == 0 && "Training preparation failed");
    
    // Save checkpoint
    result = distributed_manager_save_checkpoint(manager,
                                              (quantum_pipeline_t*)pipeline,
                                              100);
    assert(result == 0 && "Checkpoint saving failed");
    
    // Modify pipeline state
    for (size_t i = 0; i < pipeline->num_parameters; i++) {
        pipeline->parameters[i] = -1.0;
    }
    
    // Simulate failure and recovery
    if (size > 1) {
        result = distributed_manager_handle_failure(manager, 1);
        assert(result == 0 && "Failure handling failed");
        
        // Verify pipeline state is restored
        if (rank != 1) {
            for (size_t i = 0; i < pipeline->num_parameters; i++) {
                assert(fabs(pipeline->parameters[i] - (double)i) < 1e-6);
            }
        }
    }
    
    mock_pipeline_destroy(pipeline);
    distributed_manager_destroy(manager);
    printf("Checkpoint recovery test passed on rank %d\n", rank);
}

// Test workload redistribution
static void test_workload_redistribution(void) {
    printf("Testing workload redistribution...\n");
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create test configuration
    distributed_config_t config = {
        .world_size = size,
        .local_rank = rank,
        .num_gpus_per_node = 2,
        .batch_size = 32,
        .micro_batch_size = 8,
        .use_data_parallel = true,
        .use_model_parallel = false,
        .use_pipeline_parallel = false,
        .use_gradient_checkpointing = true,
        .learning_rate = 0.001f,
        .warmup_steps = 100,
        .max_steps = 1000,
        .save_interval = 50,
        .checkpoint_dir = "/tmp/test_checkpoints"
    };
    
    // Create manager
    distributed_manager_t* manager = distributed_manager_create(&config);
    assert(manager != NULL && "Manager creation failed");
    
    // Initialize environment
    int result = distributed_manager_init_environment(manager);
    assert(result == 0 && "Environment initialization failed");
    
    // Get initial workload distribution
    size_t start_idx, end_idx;
    result = distributed_manager_get_local_batch(manager, 1000,
                                               &start_idx, &end_idx);
    assert(result == 0 && "Initial workload distribution failed");
    size_t initial_samples = end_idx - start_idx;
    
    // Simulate failure and verify workload redistribution
    if (size > 1) {
        result = distributed_manager_handle_failure(manager, 1);
        assert(result == 0 && "Failure handling failed");
        
        if (rank != 1) {
            // Get new workload distribution
            result = distributed_manager_get_local_batch(manager, 1000,
                                                       &start_idx, &end_idx);
            assert(result == 0 && "Workload redistribution failed");
            
            size_t new_samples = end_idx - start_idx;
            
            // Verify workload increased to handle failed process
            if (rank > 1) {
                assert(new_samples > initial_samples);
            }
        }
    }
    
    distributed_manager_destroy(manager);
    printf("Workload redistribution test passed on rank %d\n", rank);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf("Running distributed failure recovery tests...\n\n");
    }
    
    // Run tests
    test_failure_detection();
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_communicator_reconstruction();
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_checkpoint_recovery();
    MPI_Barrier(MPI_COMM_WORLD);
    
    test_workload_redistribution();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nAll tests passed successfully!\n");
    }
    
    MPI_Finalize();
    return 0;
}
