#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/core/quantum_geometric_tensor_cpu.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Test utilities
static void init_test_state(QuantumAmplitude* state, size_t dim) {
    // Initialize with normalized quantum state
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float real = (float)rand() / RAND_MAX - 0.5f;
        float imag = (float)rand() / RAND_MAX - 0.5f;
        state[i].amplitude = (ComplexFloat){real, imag};
        norm += real * real + imag * imag;
    }
    
    // Normalize
    norm = sqrtf(norm);
    for (size_t i = 0; i < dim; i++) {
        state[i].amplitude.real /= norm;
        state[i].amplitude.imag /= norm;
    }
}

// Test cases
static void test_quantum_metric() {
    printf("Testing quantum metric computation...\n");
    
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* metric_gpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* metric_cpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    
    // Initialize test state
    init_test_state(state, dim);
    
    // Initialize GPU context
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    // Compute metric using GPU
    QGTConfig config = qgt_default_config();
    QGTError err = compute_quantum_metric_gpu(ctx, state, metric_gpu,
                                            dim, dim, &config);
    assert(err == QGT_SUCCESS && "GPU metric computation failed");
    
    // Compute metric using CPU for validation
    compute_quantum_metric_cpu(state, metric_cpu, dim, dim);
    
    // Compare results
    const float tolerance = 1e-6f;
    for (size_t i = 0; i < dim * dim; i++) {
        float diff_real = fabsf(metric_gpu[i].amplitude.real - metric_cpu[i].amplitude.real);
        float diff_imag = fabsf(metric_gpu[i].amplitude.imag - metric_cpu[i].amplitude.imag);
        assert(diff_real < tolerance && diff_imag < tolerance && 
               "GPU and CPU results differ");
    }
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(metric_gpu);
    free(metric_cpu);
    
    printf("Quantum metric test passed\n");
}

static void test_quantum_connection() {
    printf("Testing quantum connection computation...\n");
    
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* connection_gpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* connection_cpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    
    // Initialize test state
    init_test_state(state, dim);
    
    // Initialize GPU context
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    // Compute connection using GPU
    QGTConfig config = qgt_default_config();
    QGTError err = compute_quantum_connection_gpu(ctx, state, connection_gpu,
                                                dim, dim, &config);
    assert(err == QGT_SUCCESS && "GPU connection computation failed");
    
    // Compute connection using CPU for validation
    compute_quantum_connection_cpu(state, connection_cpu, dim, dim);
    
    // Compare results
    const float tolerance = 1e-6f;
    for (size_t i = 0; i < dim * dim; i++) {
        float diff_real = fabsf(connection_gpu[i].amplitude.real - connection_cpu[i].amplitude.real);
        float diff_imag = fabsf(connection_gpu[i].amplitude.imag - connection_cpu[i].amplitude.imag);
        assert(diff_real < tolerance && diff_imag < tolerance && 
               "GPU and CPU results differ");
    }
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(connection_gpu);
    free(connection_cpu);
    
    printf("Quantum connection test passed\n");
}

static void test_quantum_curvature() {
    printf("Testing quantum curvature computation...\n");
    
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* curvature_gpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* curvature_cpu = malloc(dim * dim * sizeof(QuantumAmplitude));
    
    // Initialize test state
    init_test_state(state, dim);
    
    // Initialize GPU context
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    // Compute curvature using GPU
    QGTConfig config = qgt_default_config();
    QGTError err = compute_quantum_curvature_gpu(ctx, state, curvature_gpu,
                                               dim, dim, &config);
    assert(err == QGT_SUCCESS && "GPU curvature computation failed");
    
    // Compute curvature using CPU for validation
    compute_quantum_curvature_cpu(state, curvature_cpu, dim, dim);
    
    // Compare results
    const float tolerance = 1e-6f;
    for (size_t i = 0; i < dim * dim; i++) {
        float diff_real = fabsf(curvature_gpu[i].amplitude.real - curvature_cpu[i].amplitude.real);
        float diff_imag = fabsf(curvature_gpu[i].amplitude.imag - curvature_cpu[i].amplitude.imag);
        assert(diff_real < tolerance && diff_imag < tolerance && 
               "GPU and CPU results differ");
    }
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(curvature_gpu);
    free(curvature_cpu);
    
    printf("Quantum curvature test passed\n");
}

static void test_parallel_transport() {
    printf("Testing parallel transport computation...\n");
    
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* transported_gpu = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* transported_cpu = malloc(dim * sizeof(QuantumAmplitude));
    
    // Initialize test state
    init_test_state(state, dim);
    
    // Initialize GPU context
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    // Compute parallel transport using GPU
    QGTConfig config = qgt_default_config();
    QGTError err = compute_parallel_transport_gpu(ctx, state, transported_gpu,
                                                dim, 1.0f, 0.0f, &config);
    assert(err == QGT_SUCCESS && "GPU parallel transport computation failed");
    
    // Compute parallel transport using CPU for validation
    compute_parallel_transport_cpu(state, transported_cpu, dim, 1.0f, 0.0f);
    
    // Compare results
    const float tolerance = 1e-6f;
    for (size_t i = 0; i < dim; i++) {
        float diff_real = fabsf(transported_gpu[i].amplitude.real - transported_cpu[i].amplitude.real);
        float diff_imag = fabsf(transported_gpu[i].amplitude.imag - transported_cpu[i].amplitude.imag);
        assert(diff_real < tolerance && diff_imag < tolerance && 
               "GPU and CPU results differ");
    }
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(transported_gpu);
    free(transported_cpu);
    
    printf("Parallel transport test passed\n");
}

static void test_geometric_phase() {
    printf("Testing geometric phase computation...\n");
    
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    float* phase_gpu = malloc(dim * sizeof(float));
    float* phase_cpu = malloc(dim * sizeof(float));
    
    // Initialize test state
    init_test_state(state, dim);
    
    // Initialize GPU context
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    // Compute geometric phase using GPU
    QGTConfig config = qgt_default_config();
    QGTError err = compute_geometric_phase_gpu(ctx, state, phase_gpu,
                                             dim, &config);
    assert(err == QGT_SUCCESS && "GPU geometric phase computation failed");
    
    // Compute geometric phase using CPU for validation
    compute_geometric_phase_cpu(state, phase_cpu, dim);
    
    // Compare results
    const float tolerance = 1e-6f;
    for (size_t i = 0; i < dim; i++) {
        float diff = fabsf(phase_gpu[i] - phase_cpu[i]);
        assert(diff < tolerance && "GPU and CPU results differ");
    }
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(phase_gpu);
    free(phase_cpu);
    
    printf("Geometric phase test passed\n");
}

// Error handling tests
static void test_error_handling() {
    printf("Testing error handling...\n");
    
    // Test invalid arguments
    GPUContext* ctx = quantum_gpu_init();
    assert(ctx != NULL && "Failed to initialize GPU context");
    
    QGTConfig config = qgt_default_config();
    QGTError err = compute_quantum_metric_gpu(ctx, NULL, NULL, 0, 0, &config);
    assert(err == QGT_ERROR_INVALID_ARGUMENT && "Invalid argument check failed");
    
    // Test invalid quantum state
    const size_t dim = 16;
    QuantumAmplitude* state = malloc(dim * sizeof(QuantumAmplitude));
    QuantumAmplitude* metric = malloc(dim * dim * sizeof(QuantumAmplitude));
    
    // Initialize with invalid state (not normalized)
    for (size_t i = 0; i < dim; i++) {
        state[i].amplitude = (ComplexFloat){1.0f, 1.0f};
    }
    
    err = compute_quantum_metric_gpu(ctx, state, metric, dim, dim, &config);
    assert(err == QGT_ERROR_INVALID_STATE && "Invalid state check failed");
    
    // Cleanup
    quantum_gpu_cleanup(ctx);
    free(state);
    free(metric);
    
    printf("Error handling test passed\n");
}

// Main test runner
int main() {
    printf("Running quantum geometric tensor GPU tests...\n");
    
    // Run tests
    test_quantum_metric();
    test_quantum_connection();
    test_quantum_curvature();
    test_parallel_transport();
    test_geometric_phase();
    test_error_handling();
    
    printf("All tests passed!\n");
    return 0;
}
