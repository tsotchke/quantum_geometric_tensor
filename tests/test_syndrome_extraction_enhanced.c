/**
 * @file test_syndrome_extraction_enhanced.c
 * @brief Tests for enhanced syndrome extraction with Z-stabilizer optimizations
 */

#include "quantum_geometric/physics/syndrome_extraction.h"
#include "quantum_geometric/physics/z_stabilizer_operations.h"
#include "quantum_geometric/hardware/quantum_hardware_optimization.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test helper functions
static void test_z_stabilizer_integration(void);
static void test_enhanced_correlation_tracking(void);
static void test_hardware_optimizations(void);
static void test_phase_stability(void);
static void test_spatial_coherence(void);
static void test_error_cases(void);
static void test_performance_requirements(void);
static void test_parallel_processing(void);

// Helper functions
static SyndromeConfig create_test_config(void);
static quantum_state* create_test_state(void);
static void verify_phase_metrics(const SyndromeState* state);
static void verify_spatial_metrics(const SyndromeState* state);
static void verify_hardware_metrics(const SyndromeState* state);

int main(void) {
    printf("Running enhanced syndrome extraction tests...\n");

    // Run all tests
    test_z_stabilizer_integration();
    test_enhanced_correlation_tracking();
    test_hardware_optimizations();
    test_phase_stability();
    test_spatial_coherence();
    test_error_cases();
    test_performance_requirements();
    test_parallel_processing();

    printf("All enhanced syndrome extraction tests passed!\n");
    return 0;
}

static void test_z_stabilizer_integration(void) {
    printf("Testing Z-stabilizer integration...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();
    ErrorSyndrome syndrome = {0};

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);
    assert(state.cache != NULL);
    assert(state.cache->z_state != NULL);

    success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
    assert(success);

    assert(state.phase_stability > 0.95);
    assert(state.hardware_efficiency > 0.95);
    assert(state.cache->z_state->phase_tracking_enabled);
    assert(state.cache->z_state->error_correction_active);

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Z-stabilizer integration tests passed\n");
}

static void test_enhanced_correlation_tracking(void) {
    printf("Testing enhanced correlation tracking...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);

    for (int i = 0; i < 10; i++) {
        ErrorSyndrome syndrome = {0};
        success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
        assert(success);
        
        // Verify temporal correlations
        assert(state.cache->temporal_correlations != NULL);
        assert(state.temporal_stability > 0.9);
    }

    // Verify spatial correlations
    for (size_t i = 0; i < config.lattice_width * config.lattice_height; i++) {
        for (size_t j = i + 1; j < config.lattice_width * config.lattice_height; j++) {
            size_t x1 = i % config.lattice_width;
            size_t y1 = i / config.lattice_width;
            size_t x2 = j % config.lattice_width;
            size_t y2 = j / config.lattice_width;
            double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
            
            size_t idx = i * config.lattice_width * config.lattice_height + j;
            assert(state.cache->spatial_correlations[idx] <= exp(-distance/2.0));
        }
    }

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Enhanced correlation tracking tests passed\n");
}

static void test_hardware_optimizations(void) {
    printf("Testing hardware optimizations...\n");

    SyndromeConfig config = create_test_config();
    config.dynamic_phase_correction = true;
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);

    ErrorSyndrome syndrome = {0};
    success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
    assert(success);

    verify_hardware_metrics(&state);

    double initial_phase = state.phase_stability;
    for (int i = 0; i < 5; i++) {
        success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
        assert(success);
    }
    assert(state.phase_stability >= initial_phase);

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Hardware optimization tests passed\n");
}

static void test_parallel_processing(void) {
    printf("Testing parallel processing...\n");

    SyndromeConfig config = create_test_config();
    config.enable_parallel = true;
    config.parallel_group_size = 16;
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);

    // Verify parallel group setup
    assert(state.num_parallel_groups > 0);
    assert(state.parallel_group_size == 16);
    assert(state.parallel_enabled);

    // Test parallel execution
    ErrorSyndrome syndrome = {0};
    clock_t start = clock();
    success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
    clock_t end = clock();
    assert(success);

    double parallel_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1e6;

    // Test serial execution for comparison
    config.enable_parallel = false;
    SyndromeState serial_state = {0};
    success = init_syndrome_extraction_enhanced(&serial_state, &config);
    assert(success);

    start = clock();
    success = extract_error_syndrome_enhanced(&serial_state, qstate, &syndrome);
    end = clock();
    assert(success);

    double serial_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1e6;

    // Verify parallel speedup
    assert(parallel_time < serial_time / 2.0);

    cleanup_syndrome_extraction(&state);
    cleanup_syndrome_extraction(&serial_state);
    free(qstate);

    printf("Parallel processing tests passed\n");
}

static void test_phase_stability(void) {
    printf("Testing phase stability metrics...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);

    for (int i = 0; i < 10; i++) {
        ErrorSyndrome syndrome = {0};
        success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
        assert(success);
        verify_phase_metrics(&state);
    }

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Phase stability tests passed\n");
}

static void test_spatial_coherence(void) {
    printf("Testing spatial coherence analysis...\n");

    SyndromeConfig config = create_test_config();
    config.track_spatial_correlations = true;
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    bool success = init_syndrome_extraction_enhanced(&state, &config);
    assert(success);

    for (int i = 0; i < 10; i++) {
        ErrorSyndrome syndrome = {0};
        success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
        assert(success);
        verify_spatial_metrics(&state);
    }

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Spatial coherence tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    SyndromeConfig config = create_test_config();
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    assert(!init_syndrome_extraction_enhanced(NULL, &config));
    assert(!init_syndrome_extraction_enhanced(&state, NULL));

    config.lattice_width = 0;
    assert(!init_syndrome_extraction_enhanced(&state, &config));
    config = create_test_config();

    ErrorSyndrome syndrome = {0};
    assert(!extract_error_syndrome_enhanced(NULL, qstate, &syndrome));
    assert(!extract_error_syndrome_enhanced(&state, NULL, &syndrome));
    assert(!extract_error_syndrome_enhanced(&state, qstate, NULL));

    free(qstate);
    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    SyndromeConfig config = create_test_config();
    config.lattice_width = 100;
    config.lattice_height = 100;
    config.enable_parallel = true;
    config.parallel_group_size = 16;
    
    SyndromeState state = {0};
    quantum_state* qstate = create_test_state();

    clock_t start = clock();
    bool success = init_syndrome_extraction_enhanced(&state, &config);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1e6;
    
    assert(success);
    assert(init_time < 10.0);  // <10μs target

    ErrorSyndrome syndrome = {0};
    start = clock();
    success = extract_error_syndrome_enhanced(&state, qstate, &syndrome);
    end = clock();
    double measurement_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1e6;
    
    assert(success);
    assert(measurement_time < 10.0);  // <10μs target

    cleanup_syndrome_extraction(&state);
    free(qstate);

    printf("Performance requirement tests passed\n");
}

static SyndromeConfig create_test_config(void) {
    SyndromeConfig config = {
        .lattice_width = 16,
        .lattice_height = 16,
        .num_threads = 8,
        .error_threshold = 0.1,
        .auto_correction = true,
        .enable_z_optimization = true,
        .use_phase_tracking = true,
        .track_spatial_correlations = true,
        .confidence_threshold = 0.9,
        .history_capacity = 1000,
        .dynamic_phase_correction = true,
        .phase_calibration = 1.0,
        .z_gate_fidelity = 0.99,
        .measurement_fidelity = 0.98,
        .enable_parallel = true,
        .parallel_group_size = 16
    };
    return config;
}

static quantum_state* create_test_state(void) {
    quantum_state* state = malloc(sizeof(quantum_state));
    if (!state) {
        return NULL;
    }

    state->width = 16;
    state->height = 16;
    state->depth = 1;
    state->num_qubits = state->width * state->height;

    size_t aligned_size = ((state->num_qubits * sizeof(qubit_state) + 63) / 64) * 64;
    state->qubits = aligned_alloc(64, aligned_size);
    if (!state->qubits) {
        free(state);
        return NULL;
    }

    for (size_t i = 0; i < state->num_qubits; i++) {
        state->qubits[i].amplitude_real = 1.0 / sqrt(2.0);
        state->qubits[i].amplitude_imag = 1.0 / sqrt(2.0);
        state->qubits[i].phase = 0.0;
        state->qubits[i].error_rate = 0.001;
    }

    state->layout = HARDWARE_OPTIMIZED_LAYOUT;
    state->cache_line_aligned = true;
    state->supports_simd = true;
    state->supports_gpu_acceleration = true;

    state->hardware_config.num_threads = 8;
    state->hardware_config.cache_size = get_l3_cache_size();
    state->hardware_config.simd_width = get_simd_width();
    state->hardware_config.gpu_available = check_gpu_availability();
    state->hardware_config.tensor_cores_available = check_tensor_cores();

    optimize_memory_layout(state);
    setup_hardware_prefetch(state);
    initialize_error_buffers(state);

    return state;
}

static void verify_phase_metrics(const SyndromeState* state) {
    assert(state->phase_stability >= 0.0 && state->phase_stability <= 1.0);
    assert(state->confidence_level >= 0.0 && state->confidence_level <= 1.0);
    assert(state->temporal_stability >= 0.0 && state->temporal_stability <= 1.0);
    
    for (size_t i = 0; i < state->config.lattice_width * state->config.lattice_height; i++) {
        assert(fabs(state->cache->phase_correlations[i]) <= 1.0);
    }
}

static void verify_spatial_metrics(const SyndromeState* state) {
    assert(state->spatial_coherence >= 0.0 && state->spatial_coherence <= 1.0);
    
    size_t size = state->config.lattice_width * state->config.lattice_height;
    for (size_t i = 0; i < size; i++) {
        for (size_t j = i + 1; j < size; j++) {
            size_t idx = i * size + j;
            assert(state->cache->spatial_correlations[idx] >= 0.0);
            assert(state->cache->spatial_correlations[idx] <= 1.0);
        }
    }
}

static void verify_hardware_metrics(const SyndromeState* state) {
    assert(state->hardware_efficiency > 0.95);
    assert(state->cache_hit_rate > 0.90);
    assert(state->simd_utilization > 0.85);
    assert(state->gpu_utilization > 0.80);
    assert(state->memory_bandwidth_utilization > 0.75);
    assert(state->parallel_efficiency > 0.90);
}
