#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Test parameters
#define TEST_LATTICE_SIZE 32
#define TEST_NUM_COMPONENTS 4
#define TEST_NUM_GENERATORS 3
#define TEST_ITERATIONS 100

// Timer function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize test field
static QuantumField* init_test_field() {
    FieldConfig config = {
        .lattice_size = TEST_LATTICE_SIZE,
        .num_components = TEST_NUM_COMPONENTS,
        .num_generators = TEST_NUM_GENERATORS,
        .mass = 1.0,
        .coupling = 0.1,
        .field_strength = 1.0,
        .gauge_group = true
    };
    
    GeometricConfig geom = {
        .metric = calloc(SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .connection = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double)),
        .curvature = calloc(SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS * SPACETIME_DIMS, sizeof(double))
    };
    
    // Set Minkowski metric
    for (size_t i = 0; i < SPACETIME_DIMS; i++) {
        geom.metric[i * SPACETIME_DIMS + i] = (i == 0) ? -1.0 : 1.0;
    }
    
    QuantumField* field = init_quantum_field(&config, &geom);
    
    free(geom.metric);
    free(geom.connection);
    free(geom.curvature);
    
    return field;
}

// Test GPU initialization
void test_gpu_init() {
    printf("Testing GPU initialization...\n");
    
    bool has_gpu = has_gpu_acceleration();
    printf("GPU acceleration available: %s\n", has_gpu ? "yes" : "no");
    
    if (has_gpu) {
        printf("GPU device: %s\n", get_gpu_device_name());
        printf("GPU backend: %s\n", 
            get_gpu_backend_type() == GPU_BACKEND_CUDA ? "CUDA" :
            get_gpu_backend_type() == GPU_BACKEND_METAL ? "Metal" : "Unknown");
    }
    
    printf("GPU initialization test passed\n\n");
}

// Test rotation operation
void test_rotation() {
    printf("Testing rotation operation...\n");
    
    QuantumField* field = init_test_field();
    
    // Test parameters
    size_t qubit = 0;
    double theta = M_PI / 4;
    double phi = M_PI / 3;
    
    // Measure CPU time
    double cpu_start = get_time();
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        apply_rotation_cpu(field, qubit, theta, phi);
    }
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    
    // Reset field
    cleanup_quantum_field(field);
    field = init_test_field();
    
    // Measure GPU time
    double gpu_start = get_time();
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        int result = apply_rotation_gpu(field, qubit, theta, phi);
        assert(result == 0);
    }
    double gpu_end = get_time();
    double gpu_time = gpu_end - gpu_start;
    
    printf("Rotation performance:\n");
    printf("  CPU time: %.6f seconds\n", cpu_time);
    printf("  GPU time: %.6f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    
    cleanup_quantum_field(field);
    printf("Rotation test passed\n\n");
}

// Test energy calculation
void test_energy() {
    printf("Testing energy calculation...\n");
    
    QuantumField* field = init_test_field();
    
    // Measure CPU time
    double cpu_start = get_time();
    double cpu_energy = 0.0;
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        cpu_energy = calculate_field_energy_cpu(field);
    }
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    
    // Measure GPU time
    double gpu_start = get_time();
    double gpu_energy = 0.0;
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        gpu_energy = calculate_field_energy_gpu(field);
    }
    double gpu_end = get_time();
    double gpu_time = gpu_end - gpu_start;
    
    // Verify results match
    assert(fabs(cpu_energy - gpu_energy) < 1e-6);
    
    printf("Energy calculation performance:\n");
    printf("  CPU time: %.6f seconds\n", cpu_time);
    printf("  GPU time: %.6f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    
    cleanup_quantum_field(field);
    printf("Energy calculation test passed\n\n");
}

// Test field equations
void test_equations() {
    printf("Testing field equations...\n");
    
    QuantumField* field = init_test_field();
    
    // Create equations tensor
    size_t eq_dims[5] = {
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_LATTICE_SIZE,
        TEST_NUM_COMPONENTS
    };
    Tensor* cpu_equations = init_tensor(eq_dims, 5);
    Tensor* gpu_equations = init_tensor(eq_dims, 5);
    
    // Measure CPU time
    double cpu_start = get_time();
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        calculate_field_equations_cpu(field, cpu_equations);
    }
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    
    // Measure GPU time
    double gpu_start = get_time();
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        int result = calculate_field_equations_gpu(field, gpu_equations);
        assert(result == 0);
    }
    double gpu_end = get_time();
    double gpu_time = gpu_end - gpu_start;
    
    // Verify results match
    for (size_t i = 0; i < cpu_equations->size; i++) {
        assert(cabs(cpu_equations->data[i] - gpu_equations->data[i]) < 1e-6);
    }
    
    printf("Field equations performance:\n");
    printf("  CPU time: %.6f seconds\n", cpu_time);
    printf("  GPU time: %.6f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    
    cleanup_tensor(cpu_equations);
    cleanup_tensor(gpu_equations);
    cleanup_quantum_field(field);
    printf("Field equations test passed\n\n");
}

// Test GPU monitoring
void test_monitoring() {
    printf("Testing GPU monitoring...\n");
    
    // Memory usage
    size_t mem_usage = get_gpu_memory_usage();
    printf("GPU memory usage: %zu bytes\n", mem_usage);
    
    // Utilization
    int utilization = get_gpu_utilization();
    if (utilization >= 0) {
        printf("GPU utilization: %d%%\n", utilization);
    }
    
    // Temperature
    int temp = get_gpu_temperature();
    if (temp >= 0) {
        printf("GPU temperature: %d°C\n", temp);
    }
    
    // Power usage
    float power = get_gpu_power_usage();
    if (power >= 0) {
        printf("GPU power usage: %.1f watts\n", power);
    }
    
    printf("GPU monitoring test passed\n\n");
}

int main() {
    printf("Running quantum field GPU tests...\n\n");
    
    test_gpu_init();
    test_rotation();
    test_energy();
    test_equations();
    test_monitoring();
    
    cleanup_gpu_backend();
    printf("All tests passed!\n");
    return 0;
}
