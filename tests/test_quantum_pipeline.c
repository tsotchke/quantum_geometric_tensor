#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_perf.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_error.h"
#include "quantum_geometric/distributed/quantum_distributed_operations.h"
#include "quantum_geometric/hybrid/quantum_classical_orchestrator.h"
#include "quantum_geometric/hardware/quantum_hardware_optimization.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define TEST_DATA_SIZE 100
#define INPUT_DIM 784
#define NUM_CLASSES 10
#define EPSILON 1e-6

static HardwareProfile* create_test_profile(void) {
    HardwareProfile* profile = malloc(sizeof(HardwareProfile));
    if (!profile) {
        return NULL;
    }

    profile->min_confidence_threshold = 0.8;
    profile->learning_rate = 0.1;
    profile->spatial_scale = 2.0;
    profile->pattern_scale_factor = 1.5;
    profile->noise_scale = 0.05;
    profile->phase_calibration = 0.99;
    profile->gate_fidelity = 0.995;
    profile->measurement_fidelity = 0.99;
    profile->fast_feedback_enabled = true;
    profile->hardware_optimized_layout = true;
    profile->supports_parallel_measurement = true;
    profile->max_parallel_operations = 16;
    profile->cache_line_size = 64;
    profile->memory_alignment = 64;
    profile->simd_width = 8;
    profile->num_compute_units = 8;

    return profile;
}

void test_pipeline_create() {
    printf("Testing pipeline creation...\n");
    
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    PipelineConfig config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = false,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(&config);
    assert(pipeline != NULL);
    
    quantum_pipeline_destroy(pipeline);
    free(hw_profile);
    printf("Pipeline creation test passed\n");
}

void test_distributed_pipeline_create() {
    printf("Testing distributed pipeline creation...\n");
    
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    PipelineConfig config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = true,
        .enable_hybrid = false,
        .num_nodes = 4,
        .node_rank = 0,
        .communication_backend = DISTRIBUTED_BACKEND_MPI
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(&config);
    if (pipeline == NULL) {
        printf("GPU pipeline creation skipped - GPU not available\n");
        return;
    }
    
    quantum_pipeline_destroy(pipeline);
    free(hw_profile);
    printf("GPU pipeline creation test passed\n");
}

void test_hybrid_pipeline_create() {
    printf("Testing hybrid quantum-classical pipeline creation...\n");
    
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    PipelineConfig config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = true,
        .quantum_resource_ratio = 0.5,
        .classical_resource_ratio = 0.5,
        .hybrid_optimization_mode = HYBRID_MODE_BALANCED
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(&config);
    if (pipeline == NULL) {
        printf("Hybrid pipeline creation skipped - quantum resources not available\n");
        free(hw_profile);
        return;
    }
    
    quantum_pipeline_destroy(pipeline);
    free(hw_profile);
    printf("Hybrid pipeline creation test passed\n");
}

void test_pipeline_training() {
    printf("Testing pipeline training...\n");
    
    // Create test data
    float* data = malloc(TEST_DATA_SIZE * INPUT_DIM * sizeof(float));
    int* labels = malloc(TEST_DATA_SIZE * sizeof(int));
    
    // Initialize with simple pattern
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            data[i * INPUT_DIM + j] = (float)rand() / RAND_MAX;
        }
        labels[i] = i % NUM_CLASSES;
    }
    
    // Create hardware profile
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    // Test CPU pipeline
    PipelineConfig cpu_config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = false,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false
    };
    
    quantum_pipeline_handle_t cpu_pipeline = quantum_pipeline_create(&cpu_config);
    assert(cpu_pipeline != NULL);
    
    // Train CPU pipeline
    int cpu_result = quantum_pipeline_train(cpu_pipeline, data, labels, TEST_DATA_SIZE);
    assert(cpu_result == 0);
    
    // Test GPU pipeline with hardware optimization
    PipelineConfig gpu_config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false,
        .gpu_optimization_level = GPU_OPTIMIZATION_AGGRESSIVE
    };
    
    quantum_pipeline_handle_t gpu_pipeline = quantum_pipeline_create(&gpu_config);
    if (gpu_pipeline != NULL) {
        // Train GPU pipeline
        int gpu_result = quantum_pipeline_train(gpu_pipeline, data, labels, TEST_DATA_SIZE);
        assert(gpu_result == 0);
        
        // Compare results
        float cpu_results[3], gpu_results[3];
        quantum_pipeline_evaluate(cpu_pipeline, data, labels, TEST_DATA_SIZE, cpu_results);
        quantum_pipeline_evaluate(gpu_pipeline, data, labels, TEST_DATA_SIZE, gpu_results);
        
        // Check accuracy difference is within tolerance
        assert(fabs(cpu_results[0] - gpu_results[0]) < EPSILON);
        
        // GPU should be faster
        assert(gpu_results[1] < cpu_results[1]);
        
        quantum_pipeline_destroy(gpu_pipeline);
    }
    
    quantum_pipeline_destroy(cpu_pipeline);
    free(hw_profile);
    free(data);
    free(labels);
    
    printf("Pipeline training test passed\n");
}

void test_pipeline_error_handling() {
    printf("Testing comprehensive error handling...\n");
    printf("Testing pipeline error handling...\n");
    
    // Test configuration validation
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    // Test null config
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(NULL);
    assert(pipeline == NULL);
    
    // Test invalid dimensions
    PipelineConfig invalid_config = {
        .input_dim = 0,  // Invalid
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = false,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false
    };
    
    pipeline = quantum_pipeline_create(&invalid_config);
    assert(pipeline == NULL);
    
    // Test hardware compatibility errors
    PipelineConfig gpu_config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false,
        .gpu_optimization_level = GPU_OPTIMIZATION_AGGRESSIVE
    };
    
    pipeline = quantum_pipeline_create(&gpu_config);
    if (pipeline != NULL) {
        // Test out of memory error
        size_t huge_size = (size_t)1 << 40;  // 1 TB
        float* huge_data = malloc(huge_size * sizeof(float));
        int* labels = malloc(huge_size * sizeof(int));
        
        if (huge_data && labels) {
            int result = quantum_pipeline_train(pipeline, huge_data, labels, huge_size);
            assert(result != 0);  // Should fail
            free(huge_data);
            free(labels);
        }
        
        quantum_pipeline_destroy(pipeline);
    }
    
    // Test distributed errors
    PipelineConfig distributed_config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = true,
        .enable_hybrid = false,
        .num_nodes = 0,  // Invalid
        .node_rank = 0,
        .communication_backend = DISTRIBUTED_BACKEND_MPI
    };
    
    pipeline = quantum_pipeline_create(&distributed_config);
    assert(pipeline == NULL);  // Should fail due to invalid node count

    // Test hybrid mode errors
    PipelineConfig hybrid_config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = true,
        .quantum_resource_ratio = 2.0,  // Invalid
        .classical_resource_ratio = 0.5,
        .hybrid_optimization_mode = HYBRID_MODE_BALANCED
    };
    
    pipeline = quantum_pipeline_create(&hybrid_config);
    assert(pipeline == NULL);  // Should fail due to invalid resource ratio

    free(hw_profile);
    printf("Comprehensive error handling test passed\n");
}

void test_pipeline_performance_monitoring() {
    printf("Testing pipeline performance monitoring...\n");
    
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    PipelineConfig config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = false,
        .enable_hybrid = false,
        .gpu_optimization_level = GPU_OPTIMIZATION_AGGRESSIVE,
        .enable_profiling = true,
        .profile_memory = true,
        .profile_compute = true
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(&config);
    if (pipeline == NULL) {
        printf("Performance test skipped - GPU not available\n");
        return;
    }
    
    // Create test data
    float* data = malloc(TEST_DATA_SIZE * INPUT_DIM * sizeof(float));
    int* labels = malloc(TEST_DATA_SIZE * sizeof(int));
    
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            data[i * INPUT_DIM + j] = (float)rand() / RAND_MAX;
        }
        labels[i] = i % NUM_CLASSES;
    }
    
    // Train and measure performance
    float results[3];
    quantum_pipeline_train(pipeline, data, labels, TEST_DATA_SIZE);
    quantum_pipeline_evaluate(pipeline, data, labels, TEST_DATA_SIZE, results);
    
    // Verify performance metrics
    assert(results[0] >= 0.0f && results[0] <= 1.0f);  // accuracy
    assert(results[1] > 0.0f);  // execution time
    assert(results[2] > 0.0f);  // memory usage
    
    // Check comprehensive performance metrics
    PerformanceMetrics metrics = quantum_pipeline_get_metrics(pipeline);
    
    // Verify hardware utilization
    assert(metrics.gpu_utilization > 0.0f);
    assert(metrics.memory_utilization > 0.0f);
    assert(metrics.tensor_core_utilization > 0.0f);
    
    // Verify throughput metrics
    assert(metrics.operations_per_second > 0.0);
    assert(metrics.memory_bandwidth > 0.0);
    
    // Verify efficiency metrics
    assert(metrics.computational_efficiency > 0.0f);
    assert(metrics.memory_efficiency > 0.0f);
    
    // Verify hardware-specific metrics
    if (metrics.tensor_cores_available) {
        assert(metrics.tensor_core_occupancy > 0.0f);
        assert(metrics.active_warps > 0);
        assert(metrics.achieved_occupancy > 0.0f);
    }
    
    // Verify distributed metrics if enabled
    if (config.enable_distributed) {
        assert(metrics.communication_overhead >= 0.0f);
        assert(metrics.load_balance_factor > 0.0f);
    }
    
    // Verify hybrid metrics if enabled
    if (config.enable_hybrid) {
        assert(metrics.quantum_classical_ratio > 0.0f);
        assert(metrics.hybrid_efficiency > 0.0f);
    }
    
    quantum_pipeline_destroy(pipeline);
    free(hw_profile);
    free(data);
    free(labels);
    
    printf("Pipeline performance test passed\n");
}

void test_pipeline_recovery() {
    printf("Testing pipeline error recovery...\n");
    
    HardwareProfile* hw_profile = create_test_profile();
    assert(hw_profile != NULL);

    PipelineConfig config = {
        .input_dim = INPUT_DIM,
        .latent_dim = 32,
        .num_clusters = 16,
        .num_classes = NUM_CLASSES,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .use_gpu = true,
        .hw_profile = hw_profile,
        .enable_distributed = true,
        .enable_hybrid = true,
        .enable_fault_tolerance = true,
        .checkpoint_interval = 100,
        .max_recovery_attempts = 3
    };
    
    quantum_pipeline_handle_t pipeline = quantum_pipeline_create(&config);
    if (!pipeline) {
        printf("Recovery test skipped - required features not available\n");
        free(hw_profile);
        return;
    }

    // Test recovery from simulated failures
    ErrorInjectionConfig error_config = {
        .gpu_failure_rate = 0.1,
        .network_failure_rate = 0.1,
        .quantum_error_rate = 0.1
    };
    
    quantum_pipeline_inject_errors(pipeline, &error_config);
    
    // Run training with error injection
    float* data = malloc(TEST_DATA_SIZE * INPUT_DIM * sizeof(float));
    int* labels = malloc(TEST_DATA_SIZE * sizeof(int));
    
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            data[i * INPUT_DIM + j] = (float)rand() / RAND_MAX;
        }
        labels[i] = i % NUM_CLASSES;
    }
    
    int result = quantum_pipeline_train(pipeline, data, labels, TEST_DATA_SIZE);
    assert(result == 0);  // Should complete despite injected errors
    
    // Verify recovery metrics
    RecoveryMetrics recovery_metrics = quantum_pipeline_get_recovery_metrics(pipeline);
    assert(recovery_metrics.num_recoveries > 0);
    assert(recovery_metrics.avg_recovery_time > 0.0);
    assert(recovery_metrics.recovery_success_rate > 0.0f);
    
    free(data);
    free(labels);
    quantum_pipeline_destroy(pipeline);
    free(hw_profile);
    printf("Pipeline recovery test passed\n");
}

int main() {
    printf("Running comprehensive quantum pipeline tests...\n\n");
    
    test_pipeline_create();
    test_distributed_pipeline_create();
    test_hybrid_pipeline_create();
    test_pipeline_training();
    test_pipeline_error_handling();
    test_pipeline_performance_monitoring();
    test_pipeline_recovery();
    
    printf("\nAll tests passed!\n");
    return 0;
}
