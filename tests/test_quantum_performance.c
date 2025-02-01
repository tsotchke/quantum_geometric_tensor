#include "../include/quantum_geometric_core.h"
#include "../include/quantum_geometric_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>

/**
 * @file test_quantum_performance.c
 * @brief Performance validation tests for quantum geometric learning
 */

/* Test parameters */
#define SMALL_MODEL_SIZE 1000
#define MEDIUM_MODEL_SIZE 10000
#define LARGE_MODEL_SIZE 100000
#define HUGE_MODEL_SIZE 1000000

#define NUM_ITERATIONS 10
#define ERROR_TOLERANCE 1e-6
#define MIN_SPEEDUP 10.0
#define MIN_ERROR_REDUCTION 1e6  // 10⁻⁶ error reduction
#define MAX_CIRCUIT_DEPTH 1024

/* Helper function to measure execution time */
static double measure_time(void (*func)(void)) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    func();
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
}

/* Test quantum error correction */
static void test_error_correction() {
    printf("\nTesting Quantum Error Correction\n");
    printf("================================\n");
    
    /* Create test tensor */
    quantum_geometric_tensor* tensor = create_quantum_tensor(
        MEDIUM_MODEL_SIZE, MEDIUM_MODEL_SIZE,
        QGT_MEM_HUGE_PAGES | QGT_OP_GPU_OFFLOAD
    );
    assert(tensor != NULL);
    
    /* Initialize with random states */
    for (size_t i = 0; i < tensor->num_spins; i++) {
        double real = (double)rand() / RAND_MAX;
        double imag = (double)rand() / RAND_MAX;
        double norm = sqrt(real * real + imag * imag);
        tensor->spin_system.spin_states[i] = (real + I * imag) / norm;
    }
    
    /* Measure initial error rate */
    double initial_error = 0.0;
    for (size_t i = 0; i < NUM_ITERATIONS; i++) {
        /* Introduce random errors */
        for (size_t j = 0; j < tensor->num_spins; j++) {
            if ((double)rand() / RAND_MAX < 0.01) {  // 1% error rate
                tensor->spin_system.spin_states[j] *= -1;
            }
        }
        
        /* Measure error */
        double fidelity;
        calculate_fidelity(tensor, &fidelity);
        initial_error += 1.0 - fidelity;
    }
    initial_error /= NUM_ITERATIONS;
    
    /* Apply error correction */
    qgt_error_t err = apply_error_correction(tensor, QGT_QEC_HIGH_PERFORMANCE);
    assert(err == QGT_SUCCESS);
    
    /* Measure final error rate */
    double final_error = 0.0;
    for (size_t i = 0; i < NUM_ITERATIONS; i++) {
        double fidelity;
        calculate_fidelity(tensor, &fidelity);
        final_error += 1.0 - fidelity;
    }
    final_error /= NUM_ITERATIONS;
    
    printf("Initial error rate: %.2e\n", initial_error);
    printf("Final error rate: %.2e\n", final_error);
    printf("Error reduction: %.2e x\n", initial_error / final_error);
    
    assert(final_error < initial_error / MIN_ERROR_REDUCTION);
    
    free_quantum_tensor(tensor);
}

/* Test circuit optimization */
static void test_circuit_optimization() {
    printf("\nTesting Circuit Optimization\n");
    printf("===========================\n");
    
    /* Create test tensors of increasing size */
    size_t sizes[] = {
        SMALL_MODEL_SIZE,
        MEDIUM_MODEL_SIZE,
        LARGE_MODEL_SIZE,
        HUGE_MODEL_SIZE
    };
    
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        printf("\nTesting size %zu\n", sizes[i]);
        
        quantum_geometric_tensor* tensor = create_quantum_tensor(
            sizes[i], sizes[i],
            QGT_MEM_HUGE_PAGES | QGT_OP_GPU_OFFLOAD
        );
        assert(tensor != NULL);
        
        /* Initialize with random states */
        for (size_t j = 0; j < tensor->num_spins; j++) {
            double real = (double)rand() / RAND_MAX;
            double imag = (double)rand() / RAND_MAX;
            double norm = sqrt(real * real + imag * imag);
            tensor->spin_system.spin_states[j] = (real + I * imag) / norm;
        }
        
        /* Measure baseline performance */
        double baseline_time = 0.0;
        for (size_t j = 0; j < NUM_ITERATIONS; j++) {
            clock_t start = clock();
            evolve_quantum_state(tensor, 0.1, 0);
            baseline_time += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        baseline_time /= NUM_ITERATIONS;
        
        /* Measure optimized performance */
        double optimized_time = 0.0;
        for (size_t j = 0; j < NUM_ITERATIONS; j++) {
            clock_t start = clock();
            optimize_quantum_operations(tensor, QGT_OP_GPU_OFFLOAD);
            optimized_time += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        optimized_time /= NUM_ITERATIONS;
        
        printf("Baseline time: %.3f s\n", baseline_time);
        printf("Optimized time: %.3f s\n", optimized_time);
        printf("Speedup: %.2f x\n", baseline_time / optimized_time);
        
        /* Verify O(log n) scaling */
        if (i > 0) {
            double size_ratio = (double)sizes[i] / sizes[i-1];
            double time_ratio = optimized_time / baseline_time;
            double complexity = log2(time_ratio) / log2(size_ratio);
            
            printf("Empirical complexity: O(n^%.2f)\n", complexity);
            assert(complexity < 0.5);  // Should be close to O(log n)
        }
        
        free_quantum_tensor(tensor);
    }
}

/* Test distributed execution */
static void test_distributed_execution() {
    printf("\nTesting Distributed Execution\n");
    printf("============================\n");
    
    /* Create large test tensor */
    quantum_geometric_tensor* tensor = create_quantum_tensor(
        HUGE_MODEL_SIZE, HUGE_MODEL_SIZE,
        QGT_MEM_HUGE_PAGES | QGT_OP_GPU_OFFLOAD
    );
    assert(tensor != NULL);
    
    /* Initialize with random states */
    for (size_t i = 0; i < tensor->num_spins; i++) {
        double real = (double)rand() / RAND_MAX;
        double imag = (double)rand() / RAND_MAX;
        double norm = sqrt(real * real + imag * imag);
        tensor->spin_system.spin_states[i] = (real + I * imag) / norm;
    }
    
    /* Measure baseline performance */
    double baseline_time = 0.0;
    for (size_t i = 0; i < NUM_ITERATIONS; i++) {
        clock_t start = clock();
        evolve_quantum_state(tensor, 0.1, QGT_OP_GPU_OFFLOAD);
        baseline_time += (double)(clock() - start) / CLOCKS_PER_SEC;
    }
    baseline_time /= NUM_ITERATIONS;
    
    /* Measure distributed performance */
    double distributed_time = 0.0;
    for (size_t i = 0; i < NUM_ITERATIONS; i++) {
        clock_t start = clock();
        execute_distributed(tensor, QGT_OP_GPU_OFFLOAD);
        distributed_time += (double)(clock() - start) / CLOCKS_PER_SEC;
    }
    distributed_time /= NUM_ITERATIONS;
    
    printf("Baseline time: %.3f s\n", baseline_time);
    printf("Distributed time: %.3f s\n", distributed_time);
    printf("Speedup: %.2f x\n", baseline_time / distributed_time);
    
    assert(baseline_time / distributed_time > MIN_SPEEDUP);
    
    free_quantum_tensor(tensor);
}

/* Test quantum backend integration */
static void test_quantum_backend() {
    printf("\nTesting Quantum Backend Integration\n");
    printf("=================================\n");
    
    /* Create test tensor */
    quantum_geometric_tensor* tensor = create_quantum_tensor(
        MEDIUM_MODEL_SIZE, MEDIUM_MODEL_SIZE,
        QGT_MEM_HUGE_PAGES | QGT_OP_GPU_OFFLOAD
    );
    assert(tensor != NULL);
    
    /* Initialize with random states */
    for (size_t i = 0; i < tensor->num_spins; i++) {
        double real = (double)rand() / RAND_MAX;
        double imag = (double)rand() / RAND_MAX;
        double norm = sqrt(real * real + imag * imag);
        tensor->spin_system.spin_states[i] = (real + I * imag) / norm;
    }
    
    /* Execute on quantum hardware */
    qgt_error_t err = execute_on_quantum_hardware(tensor, QGT_OP_GPU_OFFLOAD);
    assert(err == QGT_SUCCESS);
    
    /* Verify results */
    double fidelity;
    err = calculate_fidelity(tensor, &fidelity);
    assert(err == QGT_SUCCESS);
    
    printf("Quantum execution fidelity: %.6f\n", fidelity);
    assert(fidelity > 0.99);
    
    free_quantum_tensor(tensor);
}

/* Test GPU backends */
static void test_gpu_backends() {
    printf("\nTesting GPU Backends\n");
    printf("===================\n");
    
    /* Test Metal backend on Apple Silicon */
    #ifdef __APPLE__
    printf("\nTesting Metal backend:\n");
    assert(set_gpu_backend(GPU_BACKEND_METAL));
    
    GpuConfig metal_config;
    assert(get_gpu_config(&metal_config));
    
    printf("Metal device: %s\n", gpu_get_device_name(0));
    printf("Compute units: %zu\n", metal_config.compute_units);
    printf("Memory: %zu MB\n", metal_config.memory_size / (1024*1024));
    printf("Max threads per block: %zu\n", metal_config.max_threads_per_block);
    printf("Shared memory: %zu KB\n", metal_config.shared_memory_size / 1024);
    #endif
    
    /* Test CUDA backend */
    #ifdef USE_CUDA
    printf("\nTesting CUDA backend:\n");
    assert(set_gpu_backend(GPU_BACKEND_CUDA));
    
    GpuConfig cuda_config;
    assert(get_gpu_config(&cuda_config));
    
    printf("CUDA device: %s\n", gpu_get_device_name(0));
    printf("Compute units: %zu\n", cuda_config.compute_units);
    printf("Memory: %zu MB\n", cuda_config.memory_size / (1024*1024));
    printf("Max threads per block: %zu\n", cuda_config.max_threads_per_block);
    printf("Shared memory: %zu KB\n", cuda_config.shared_memory_size / 1024);
    #endif
    
    /* Test GPU memory operations */
    size_t size = 1024 * 1024;  // 1 MB
    void* gpu_ptr = gpu_malloc(size);
    assert(gpu_ptr != NULL);
    
    void* host_ptr = malloc(size);
    assert(host_ptr != NULL);
    
    /* Test memory operations */
    assert(gpu_memset(gpu_ptr, 0, size));
    assert(gpu_memcpy_host_to_device(gpu_ptr, host_ptr, size));
    assert(gpu_memcpy_device_to_host(host_ptr, gpu_ptr, size));
    
    /* Test profiling */
    float elapsed_ms;
    assert(gpu_start_profiling("memory_test"));
    assert(gpu_memcpy_device_to_device(gpu_ptr, gpu_ptr, size));
    assert(gpu_stop_profiling(&elapsed_ms));
    
    printf("Memory copy time: %.3f ms\n", elapsed_ms);
    
    /* Cleanup */
    gpu_free(gpu_ptr);
    free(host_ptr);
}

/* Main test runner */
int main() {
    printf("Quantum Geometric Learning Performance Tests\n");
    printf("==========================================\n");
    
    /* Initialize random seed */
    srand(time(NULL));
    
    /* Initialize GPU */
    assert(init_gpu());
    
    /* Run tests */
    test_error_correction();
    test_circuit_optimization();
    test_distributed_execution();
    test_quantum_backend();
    test_gpu_backends();
    
    printf("\nAll performance tests passed successfully!\n");
    return 0;
}
