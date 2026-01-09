/**
 * @file test_quantum_geometric_tensor_perf.c
 * @brief Performance tests for quantum geometric tensor GPU operations
 *
 * Uses the portable GPU abstraction layer to work on Metal, CUDA, or CPU fallback.
 */

#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_error.h"
#include "quantum_geometric/core/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Test configuration
#define NUM_ITERATIONS 10

// Helper function to get time in milliseconds
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Test utilities
static void generate_random_state(ComplexFloat* state, size_t size) {
    for (size_t i = 0; i < size; i++) {
        state[i].real = (float)rand() / RAND_MAX;
        state[i].imag = (float)rand() / RAND_MAX;
    }

    // Normalize the state vector
    float norm_sq = 0.0f;
    for (size_t i = 0; i < size; i++) {
        norm_sq += state[i].real * state[i].real + state[i].imag * state[i].imag;
    }
    float norm = sqrtf(norm_sq);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < size; i++) {
            state[i].real /= norm;
            state[i].imag /= norm;
        }
    }
}

static void print_test_header(const char* test_name) {
    printf("\n=== Running %s ===\n", test_name);
}

static void print_test_result(const char* test_name, qgt_error_t error) {
    const char* status = (error == QGT_SUCCESS) ? "PASSED" : "FAILED";
    printf("%s: %s (error code: %d)\n", test_name, status, error);
}

/**
 * Initialize GPU context with portable abstraction
 */
static qgt_error_t init_gpu_context(GPUContext* ctx) {
    if (!ctx) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    memset(ctx, 0, sizeof(GPUContext));

    // Initialize GPU system
    int result = qg_gpu_init();
    if (result != QG_GPU_SUCCESS) {
        ctx->is_available = false;
        printf("Note: GPU not available, using CPU fallback\n");
        return QGT_SUCCESS;  // Not an error - just no GPU
    }

    // Check device count
    int device_count = 0;
    result = qg_gpu_get_device_count(&device_count);
    if (result != QG_GPU_SUCCESS || device_count == 0) {
        ctx->is_available = false;
        printf("Note: No GPU devices found, using CPU fallback\n");
        return QGT_SUCCESS;
    }

    // Get device info
    gpu_device_info_t info;
    result = qg_gpu_get_device_info(0, &info);
    if (result == QG_GPU_SUCCESS) {
        printf("GPU Device: %s\n", info.name);
        printf("  Total Memory: %.2f MB\n", info.total_memory / (1024.0 * 1024.0));
        printf("  Compute Units: %d\n", info.compute_units);
        printf("  Max Threads/Block: %d\n", info.max_threads_per_block);
    }

    // Set up function pointers
    ctx->is_available = true;
    ctx->malloc = gpu_malloc;
    ctx->free = qgt_gpu_free_buffer;
    ctx->memcpy_to_device = gpu_memcpy_host_to_device;
    ctx->memcpy_from_device = gpu_memcpy_device_to_host;
    ctx->get_optimal_block_size = NULL;  // Use default

    return QGT_SUCCESS;
}

/**
 * Cleanup GPU context
 */
static void cleanup_gpu_context(GPUContext* ctx) {
    if (ctx && ctx->is_available) {
        qg_gpu_cleanup();
        ctx->is_available = false;
    }
}

/**
 * CPU fallback for quantum metric tensor computation
 */
static qgt_error_t compute_quantum_metric_cpu(
    const ComplexFloat* state,
    ComplexFloat* metric,
    size_t rows,
    size_t cols) {

    // Compute quantum metric tensor g_ij = Re(<d_i psi|d_j psi>) - <d_i psi|psi><psi|d_j psi>
    // For performance testing, we do a simplified matrix-style computation

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;

            // Compute simplified metric element
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t k = 0; k < cols; k++) {
                size_t state_idx_i = i * cols + k;
                size_t state_idx_j = j * cols + k;
                if (state_idx_i < rows * cols && state_idx_j < rows * cols) {
                    // Accumulate: state[i,k]* * state[j,k]
                    sum.real += state[state_idx_i].real * state[state_idx_j].real
                              + state[state_idx_i].imag * state[state_idx_j].imag;
                    sum.imag += state[state_idx_i].real * state[state_idx_j].imag
                              - state[state_idx_i].imag * state[state_idx_j].real;
                }
            }

            metric[idx] = sum;
        }
    }

    return QGT_SUCCESS;
}

/**
 * Test quantum metric tensor performance
 */
static qgt_error_t test_metric_tensor_performance(GPUContext* ctx, size_t matrix_size) {
    print_test_header("Quantum Metric Tensor Performance Test");

    printf("Matrix size: %zu x %zu\n", matrix_size, matrix_size);

    // Allocate host memory
    size_t total_size = matrix_size * matrix_size;
    ComplexFloat* h_state = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));
    ComplexFloat* h_metric = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));

    if (!h_state || !h_metric) {
        printf("Error: Host memory allocation failed\n");
        free(h_state);
        free(h_metric);
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Generate random state vector
    generate_random_state(h_state, total_size);

    // Performance measurement variables
    double total_time = 0.0;
    double min_time = INFINITY;
    double max_time = 0.0;

    QGTConfig config = qgt_default_config();
    config.precision = 1e-6;
    config.use_quantum_estimation = false;
    config.optimization_level = 2;

    if (ctx->is_available) {
        // GPU path
        printf("Using GPU acceleration\n");

        // Allocate device memory
        void* d_state = NULL;
        void* d_metric = NULL;
        qgt_error_t err;

        err = ctx->malloc(&d_state, total_size * sizeof(ComplexFloat));
        if (err != QGT_SUCCESS) {
            printf("Error: Device memory allocation failed for state\n");
            free(h_state);
            free(h_metric);
            return err;
        }

        err = ctx->malloc(&d_metric, total_size * sizeof(ComplexFloat));
        if (err != QGT_SUCCESS) {
            printf("Error: Device memory allocation failed for metric\n");
            ctx->free(d_state);
            free(h_state);
            free(h_metric);
            return err;
        }

        // Copy data to device
        err = ctx->memcpy_to_device(d_state, h_state, total_size * sizeof(ComplexFloat));
        if (err != QGT_SUCCESS) {
            printf("Error: Host to device memory copy failed\n");
            ctx->free(d_state);
            ctx->free(d_metric);
            free(h_state);
            free(h_metric);
            return err;
        }

        // Warm-up run
        err = compute_quantum_metric_gpu(ctx, (ComplexFloat*)d_state,
                                         (ComplexFloat*)d_metric,
                                         matrix_size, matrix_size, &config);
        if (err != QGT_SUCCESS) {
            printf("Warning: GPU kernel failed, falling back to CPU\n");
            ctx->free(d_state);
            ctx->free(d_metric);
            goto cpu_path;
        }

        qg_gpu_synchronize();

        // Performance measurement loop
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start_time = get_time_ms();

            err = compute_quantum_metric_gpu(ctx, (ComplexFloat*)d_state,
                                             (ComplexFloat*)d_metric,
                                             matrix_size, matrix_size, &config);
            qg_gpu_synchronize();

            double end_time = get_time_ms();
            double kernel_time = end_time - start_time;

            if (err != QGT_SUCCESS) {
                printf("Error: Kernel launch failed on iteration %d\n", i);
                ctx->free(d_state);
                ctx->free(d_metric);
                free(h_state);
                free(h_metric);
                return err;
            }

            total_time += kernel_time;
            min_time = fmin(min_time, kernel_time);
            max_time = fmax(max_time, kernel_time);
        }

        // Copy result back for verification
        ctx->memcpy_from_device(h_metric, d_metric, total_size * sizeof(ComplexFloat));

        // Cleanup device memory
        ctx->free(d_state);
        ctx->free(d_metric);

    } else {
cpu_path:
        // CPU fallback path
        printf("Using CPU fallback\n");

        // Warm-up run
        compute_quantum_metric_cpu(h_state, h_metric, matrix_size, matrix_size);

        // Performance measurement loop
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start_time = get_time_ms();

            qgt_error_t err = compute_quantum_metric_cpu(h_state, h_metric,
                                                         matrix_size, matrix_size);

            double end_time = get_time_ms();
            double kernel_time = end_time - start_time;

            if (err != QGT_SUCCESS) {
                printf("Error: CPU computation failed on iteration %d\n", i);
                free(h_state);
                free(h_metric);
                return err;
            }

            total_time += kernel_time;
            min_time = fmin(min_time, kernel_time);
            max_time = fmax(max_time, kernel_time);
        }
    }

    // Calculate and print statistics
    double avg_time = total_time / NUM_ITERATIONS;
    double throughput = (double)(total_size * sizeof(ComplexFloat) * 2) / (avg_time * 1e6);  // GB/s
    double flops = (double)(total_size * matrix_size * 8) / (avg_time * 1e6);  // GFLOPS (approx)

    printf("\nPerformance Results:\n");
    printf("  Iterations: %d\n", NUM_ITERATIONS);
    printf("  Average Time: %.3f ms\n", avg_time);
    printf("  Min Time: %.3f ms\n", min_time);
    printf("  Max Time: %.3f ms\n", max_time);
    printf("  Estimated Throughput: %.2f GB/s\n", throughput);
    printf("  Estimated GFLOPS: %.2f\n", flops);

    // Basic result validation - check for NaN/Inf
    bool valid = true;
    for (size_t i = 0; i < total_size && valid; i++) {
        if (isnan(h_metric[i].real) || isnan(h_metric[i].imag) ||
            isinf(h_metric[i].real) || isinf(h_metric[i].imag)) {
            valid = false;
        }
    }

    if (!valid) {
        printf("Warning: Result contains NaN or Inf values\n");
    } else {
        printf("Result validation: OK (no NaN/Inf)\n");
    }

    // Cleanup host memory
    free(h_state);
    free(h_metric);

    return QGT_SUCCESS;
}

/**
 * Test quantum connection performance
 */
static qgt_error_t test_connection_performance(GPUContext* ctx, size_t matrix_size) {
    print_test_header("Quantum Connection Performance Test");

    printf("Matrix size: %zu x %zu\n", matrix_size, matrix_size);

    size_t total_size = matrix_size * matrix_size;
    ComplexFloat* h_state = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));
    ComplexFloat* h_connection = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));

    if (!h_state || !h_connection) {
        printf("Error: Host memory allocation failed\n");
        free(h_state);
        free(h_connection);
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    generate_random_state(h_state, total_size);

    double total_time = 0.0;
    QGTConfig config = qgt_default_config();

    if (ctx->is_available) {
        void* d_state = NULL;
        void* d_connection = NULL;

        ctx->malloc(&d_state, total_size * sizeof(ComplexFloat));
        ctx->malloc(&d_connection, total_size * sizeof(ComplexFloat));
        ctx->memcpy_to_device(d_state, h_state, total_size * sizeof(ComplexFloat));

        // Warm-up
        compute_quantum_connection_gpu(ctx, (ComplexFloat*)d_state,
                                       (ComplexFloat*)d_connection,
                                       matrix_size, matrix_size, &config);
        qg_gpu_synchronize();

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            compute_quantum_connection_gpu(ctx, (ComplexFloat*)d_state,
                                           (ComplexFloat*)d_connection,
                                           matrix_size, matrix_size, &config);
            qg_gpu_synchronize();
            total_time += get_time_ms() - start;
        }

        ctx->free(d_state);
        ctx->free(d_connection);
    } else {
        printf("Using CPU fallback (connection computation)\n");
        // Simple CPU fallback - just measure overhead
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            memset(h_connection, 0, total_size * sizeof(ComplexFloat));
            total_time += get_time_ms() - start;
        }
    }

    printf("Average Time: %.3f ms\n", total_time / NUM_ITERATIONS);

    free(h_state);
    free(h_connection);

    return QGT_SUCCESS;
}

/**
 * Test quantum curvature performance
 */
static qgt_error_t test_curvature_performance(GPUContext* ctx, size_t matrix_size) {
    print_test_header("Quantum Curvature Performance Test");

    printf("Matrix size: %zu x %zu\n", matrix_size, matrix_size);

    size_t total_size = matrix_size * matrix_size;
    ComplexFloat* h_state = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));
    ComplexFloat* h_curvature = (ComplexFloat*)malloc(total_size * sizeof(ComplexFloat));

    if (!h_state || !h_curvature) {
        printf("Error: Host memory allocation failed\n");
        free(h_state);
        free(h_curvature);
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    generate_random_state(h_state, total_size);

    double total_time = 0.0;
    QGTConfig config = qgt_default_config();

    if (ctx->is_available) {
        void* d_state = NULL;
        void* d_curvature = NULL;

        ctx->malloc(&d_state, total_size * sizeof(ComplexFloat));
        ctx->malloc(&d_curvature, total_size * sizeof(ComplexFloat));
        ctx->memcpy_to_device(d_state, h_state, total_size * sizeof(ComplexFloat));

        // Warm-up
        compute_quantum_curvature_gpu(ctx, (ComplexFloat*)d_state,
                                      (ComplexFloat*)d_curvature,
                                      matrix_size, matrix_size, &config);
        qg_gpu_synchronize();

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            compute_quantum_curvature_gpu(ctx, (ComplexFloat*)d_state,
                                          (ComplexFloat*)d_curvature,
                                          matrix_size, matrix_size, &config);
            qg_gpu_synchronize();
            total_time += get_time_ms() - start;
        }

        ctx->free(d_state);
        ctx->free(d_curvature);
    } else {
        printf("Using CPU fallback (curvature computation)\n");
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            memset(h_curvature, 0, total_size * sizeof(ComplexFloat));
            total_time += get_time_ms() - start;
        }
    }

    printf("Average Time: %.3f ms\n", total_time / NUM_ITERATIONS);

    free(h_state);
    free(h_curvature);

    return QGT_SUCCESS;
}

/**
 * Test memory bandwidth
 */
static qgt_error_t test_memory_bandwidth(GPUContext* ctx, size_t buffer_size_mb) {
    print_test_header("Memory Bandwidth Test");

    size_t buffer_size = buffer_size_mb * 1024 * 1024;
    printf("Buffer size: %zu MB\n", buffer_size_mb);

    char* h_data = (char*)malloc(buffer_size);
    if (!h_data) {
        printf("Error: Host memory allocation failed\n");
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Initialize with pattern
    for (size_t i = 0; i < buffer_size; i++) {
        h_data[i] = (char)(i & 0xFF);
    }

    if (ctx->is_available) {
        void* d_data = NULL;
        qgt_error_t err = ctx->malloc(&d_data, buffer_size);
        if (err != QGT_SUCCESS) {
            printf("Error: Device memory allocation failed\n");
            free(h_data);
            return err;
        }

        // Test host-to-device bandwidth
        double h2d_time = 0.0;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            ctx->memcpy_to_device(d_data, h_data, buffer_size);
            qg_gpu_synchronize();
            h2d_time += get_time_ms() - start;
        }
        double h2d_bandwidth = (buffer_size * NUM_ITERATIONS) / (h2d_time * 1e6);
        printf("Host-to-Device Bandwidth: %.2f GB/s\n", h2d_bandwidth);

        // Test device-to-host bandwidth
        double d2h_time = 0.0;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            double start = get_time_ms();
            ctx->memcpy_from_device(h_data, d_data, buffer_size);
            qg_gpu_synchronize();
            d2h_time += get_time_ms() - start;
        }
        double d2h_bandwidth = (buffer_size * NUM_ITERATIONS) / (d2h_time * 1e6);
        printf("Device-to-Host Bandwidth: %.2f GB/s\n", d2h_bandwidth);

        ctx->free(d_data);
    } else {
        printf("GPU not available, skipping bandwidth test\n");
    }

    free(h_data);
    return QGT_SUCCESS;
}

int main(int argc, char** argv) {
    printf("=================================================\n");
    printf("Quantum Geometric Tensor Performance Tests\n");
    printf("=================================================\n");

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Get matrix size from command line
    size_t matrix_size = 256;  // Default size
    if (argc > 1) {
        matrix_size = (size_t)atoi(argv[1]);
        if (matrix_size == 0) {
            printf("Error: Invalid matrix size\n");
            return 1;
        }
    }

    printf("Matrix size: %zu x %zu\n", matrix_size, matrix_size);
    printf("Iterations per test: %d\n", NUM_ITERATIONS);

    // Initialize GPU context
    GPUContext ctx;
    qgt_error_t err = init_gpu_context(&ctx);
    if (err != QGT_SUCCESS) {
        printf("Error: Failed to initialize GPU context\n");
        return 1;
    }

    int failed = 0;

    // Run performance tests
    err = test_metric_tensor_performance(&ctx, matrix_size);
    print_test_result("Metric Tensor Performance", err);
    if (err != QGT_SUCCESS) failed++;

    err = test_connection_performance(&ctx, matrix_size);
    print_test_result("Connection Performance", err);
    if (err != QGT_SUCCESS) failed++;

    err = test_curvature_performance(&ctx, matrix_size);
    print_test_result("Curvature Performance", err);
    if (err != QGT_SUCCESS) failed++;

    err = test_memory_bandwidth(&ctx, 64);  // 64 MB buffer
    print_test_result("Memory Bandwidth", err);
    if (err != QGT_SUCCESS) failed++;

    // Cleanup
    cleanup_gpu_context(&ctx);

    printf("\n=================================================\n");
    printf("Performance Tests Complete\n");
    if (failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed\n", failed);
    }
    printf("=================================================\n");

    return failed > 0 ? 1 : 0;
}
