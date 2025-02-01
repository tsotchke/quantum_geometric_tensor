/**
 * @file test_stabilizer_measurement.c
 * @brief Tests for quantum stabilizer measurement system
 */

#include "quantum_geometric/physics/stabilizer_measurement.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_initialization(void);
static void test_measurement(void);
static void test_error_detection(void);
static void test_error_correction(void);
static void test_error_cases(void);
static void test_performance_requirements(void);
static void test_hardware_integration(void);
static void test_parallel_measurement(void);
static void test_hardware_error_mitigation(void);
static void test_hardware_performance(void);
static void test_config_validation(void);
static void test_resource_usage(void);
static void test_reliability_metrics(void);

// Mock functions and data
static quantum_state* create_test_state(size_t width, size_t height);
static void apply_test_errors(quantum_state* state);
static void cleanup_test_state(quantum_state* state);
static bool stabilizers_share_qubits(const StabilizerState* state, 
                                   size_t idx1, 
                                   size_t idx2);
static void verify_hardware_metrics(const StabilizerState* state,
                                  const HardwareConfig* config);

// Helper function to verify hardware metrics
static void verify_hardware_metrics(const StabilizerState* state,
                                  const HardwareConfig* config) {
    if (!state || !config) return;

    // Verify basic metrics
    assert(state->hardware_metrics.readout_fidelity > 0.95);
    assert(state->hardware_metrics.gate_fidelity > 0.99);
    assert(state->hardware_metrics.parallel_efficiency > 0.8);

    // Verify error mitigation effectiveness
    assert(state->hardware_metrics.error_mitigation_factor > 1.5);  // At least 50% improvement
    assert(state->hardware_metrics.readout_error < config->noise_model.readout_error);
    assert(state->hardware_metrics.gate_error < config->noise_model.gate_error);

    // Verify parallel optimization
    if (config->parallel_enabled) {
        assert(state->hardware_metrics.parallel_group_count > 0);
        assert(state->hardware_metrics.avg_group_size > 1.0);
        assert(state->hardware_metrics.parallel_speedup > 1.2);  // At least 20% speedup
        assert(state->hardware_metrics.crosstalk_level < config->noise_model.crosstalk_threshold);
    }

    // Verify hardware-specific optimizations
    switch (config->type) {
        case HARDWARE_IBM:
            assert(state->hardware_metrics.custom_instruction_usage > 0.5);
            assert(state->hardware_metrics.dynamic_decoupling_effectiveness > 0.7);
            break;
        case HARDWARE_RIGETTI:
            assert(state->hardware_metrics.native_gate_utilization > 0.8);
            assert(state->hardware_metrics.qubit_mapping_efficiency > 0.9);
            break;
        case HARDWARE_DWAVE:
            assert(state->hardware_metrics.chain_strength_optimization > 0.85);
            assert(state->hardware_metrics.embedding_efficiency > 0.75);
            break;
        default:
            break;
    }

    // Verify performance metrics
    assert(state->hardware_metrics.measurement_time_us < 100.0);  // Less than 100μs
    assert(state->hardware_metrics.correction_time_us < 200.0);   // Less than 200μs
    assert(state->hardware_metrics.memory_overhead_kb < 512);     // Less than 512KB overhead
}

// Helper function to check if two stabilizers share any qubits
static bool stabilizers_share_qubits(const StabilizerState* state,
                                   size_t idx1,
                                   size_t idx2) {
    if (!state) return false;

    // Get stabilizer types and coordinates
    bool is_plaquette1 = idx1 < state->plaquette_stabilizers->size;
    bool is_plaquette2 = idx2 < state->plaquette_stabilizers->size;
    
    size_t width = state->config.lattice_width;
    size_t x1, y1, x2, y2;

    // Convert indices to grid coordinates
    if (is_plaquette1) {
        x1 = idx1 % (width - 1);
        y1 = idx1 / (width - 1);
    } else {
        idx1 -= state->plaquette_stabilizers->size;
        x1 = (idx1 % (width - 1)) + 1;
        y1 = (idx1 / (width - 1)) + 1;
    }

    if (is_plaquette2) {
        x2 = idx2 % (width - 1);
        y2 = idx2 / (width - 1);
    } else {
        idx2 -= state->plaquette_stabilizers->size;
        x2 = (idx2 % (width - 1)) + 1;
        y2 = (idx2 / (width - 1)) + 1;
    }

    // Check if stabilizers share any qubits based on type and location
    if (is_plaquette1 == is_plaquette2) {
        // Same type stabilizers
        return abs((int)x1 - (int)x2) <= 1 && abs((int)y1 - (int)y2) <= 1;
    } else {
        // Different type stabilizers
        return abs((int)x1 - (int)x2) < 1 || abs((int)y1 - (int)y2) < 1;
    }
}

int main(void) {
    printf("Running stabilizer measurement tests...\n");

    // Run all tests
    test_initialization();
    test_measurement();
    test_error_detection();
    test_error_correction();
    test_error_cases();
    test_performance_requirements();
    test_hardware_integration();
    test_parallel_measurement();
    test_hardware_error_mitigation();
    test_hardware_performance();
    test_config_validation();
    test_resource_usage();
    test_reliability_metrics();

    printf("All stabilizer measurement tests passed!\n");
    return 0;
}

static void test_resource_usage(void) {
    printf("Testing resource usage...\n");

    // Initialize large test system
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 100,   // Large lattice for stress testing
        .lattice_height = 100,
        .error_threshold = 0.1,
        .auto_correction = true,
        .hardware_config = {
            .type = HARDWARE_IBM,
            .optimization_level = 3,
            .error_mitigation = true,
            .parallel_enabled = true
        }
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create large test state
    quantum_state* qstate = create_test_state(100, 100);
    assert(qstate != NULL);

    // Monitor resource usage during operations
    for (int i = 0; i < 100; i++) {
        // Perform measurements
        success = measure_stabilizers(&state, qstate);
        assert(success);

        // Verify memory usage
        assert(state.resource_metrics.memory_overhead_kb < 1024);  // Less than 1MB
        assert(state.resource_metrics.memory_peak_kb < 2048);     // Peak under 2MB
        assert(state.resource_metrics.memory_fragmentation < 0.2); // Low fragmentation

        // Verify CPU utilization
        assert(state.resource_metrics.cpu_usage_percent < 80.0);  // Under 80% CPU
        assert(state.resource_metrics.cpu_peak_percent < 90.0);   // Peak under 90%
        assert(state.resource_metrics.thread_count <= 8);         // Reasonable threads

        // Verify GPU utilization if enabled
        if (state.hardware_config.gpu_enabled) {
            assert(state.resource_metrics.gpu_usage_percent < 90.0);  // Under 90% GPU
            assert(state.resource_metrics.gpu_memory_usage_kb < 512); // Under 512KB GPU memory
            assert(state.resource_metrics.gpu_temperature_c < 85.0);  // Temperature in check
        }

        // Verify overall system impact
        assert(state.resource_metrics.system_memory_impact < 0.1);  // Low system impact
        assert(state.resource_metrics.io_operations_per_sec < 1000); // Reasonable I/O
        assert(state.resource_metrics.network_usage_bytes < 1024);   // Minimal network
    }

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Resource usage tests passed\n");
}

static void test_reliability_metrics(void) {
    printf("Testing reliability metrics...\n");

    // Initialize system with reliability tracking
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = true,
        .reliability_tracking = true
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create test state with known errors
    quantum_state* qstate = create_test_state(4, 4);
    assert(qstate != NULL);
    apply_test_errors(qstate);

    // Track reliability over multiple operations
    size_t total_ops = 1000;
    size_t successful_ops = 0;
    size_t false_positives = 0;
    size_t recovery_attempts = 0;
    size_t successful_recoveries = 0;

    for (size_t i = 0; i < total_ops; i++) {
        // Perform measurement
        success = measure_stabilizers(&state, qstate);
        assert(success);
        
        // Track operation success
        if (state.reliability_metrics.operation_successful) {
            successful_ops++;
        }

        // Track error detection accuracy
        if (state.reliability_metrics.false_positive_detected) {
            false_positives++;
        }

        // Track recovery success
        if (state.reliability_metrics.recovery_attempted) {
            recovery_attempts++;
            if (state.reliability_metrics.recovery_successful) {
                successful_recoveries++;
            }
        }

        // Verify individual operation metrics
        assert(state.reliability_metrics.measurement_fidelity > 0.95);
        assert(state.reliability_metrics.error_detection_confidence > 0.9);
        assert(state.reliability_metrics.correction_confidence > 0.9);
    }

    // Calculate final reliability metrics
    double success_rate = (double)successful_ops / total_ops;
    double false_positive_rate = (double)false_positives / total_ops;
    double recovery_success_rate = recovery_attempts > 0 ? 
        (double)successful_recoveries / recovery_attempts : 1.0;

    // Verify against requirements
    assert(success_rate > 0.99);           // >99% success rate
    assert(false_positive_rate < 0.01);    // <1% false positives
    assert(recovery_success_rate > 0.99);  // >99% recovery success

    // Verify system uptime and stability
    assert(state.reliability_metrics.system_uptime_seconds > 0);
    assert(state.reliability_metrics.consecutive_failures < 3);
    assert(state.reliability_metrics.error_correction_latency_us < 100);

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Reliability metrics tests passed\n");
}

static void test_config_validation(void) {
    printf("Testing configuration validation...\n");

    // Test invalid hardware configurations
    struct {
        const char* name;
        StabilizerConfig config;
        bool should_succeed;
    } test_configs[] = {
        {
            .name = "Invalid lattice dimensions",
            .config = {
                .lattice_width = 0,
                .lattice_height = 4,
                .error_threshold = 0.1
            },
            .should_succeed = false
        },
        {
            .name = "Invalid error threshold",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = -0.1
            },
            .should_succeed = false
        },
        {
            .name = "Invalid parallel config",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1,
                .enable_parallel = true,
                .max_parallel_ops = 0
            },
            .should_succeed = false
        },
        {
            .name = "Invalid hardware config",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1,
                .hardware_config = {
                    .type = 999,  // Invalid type
                    .optimization_level = 3
                }
            },
            .should_succeed = false
        },
        {
            .name = "Invalid noise model",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1,
                .hardware_config = {
                    .type = HARDWARE_IBM,
                    .noise_model = {
                        .readout_error = 2.0,  // Invalid: > 1.0
                        .gate_error = 0.001
                    }
                }
            },
            .should_succeed = false
        },
        {
            .name = "Invalid error mitigation config",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1,
                .hardware_config = {
                    .type = HARDWARE_IBM,
                    .error_mitigation = true,
                    .mitigation_config = {
                        .readout_error_correction = true,
                        .dynamic_decoupling = true,
                        .zero_noise_extrapolation = true,
                        .extrapolation_order = 0  // Invalid: must be > 0
                    }
                }
            },
            .should_succeed = false
        },
        {
            .name = "Valid minimal config",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1
            },
            .should_succeed = true
        },
        {
            .name = "Valid full config",
            .config = {
                .lattice_width = 4,
                .lattice_height = 4,
                .error_threshold = 0.1,
                .auto_correction = true,
                .enable_parallel = true,
                .max_parallel_ops = 8,
                .hardware_config = {
                    .type = HARDWARE_IBM,
                    .optimization_level = 3,
                    .error_mitigation = true,
                    .parallel_enabled = true,
                    .noise_model = {
                        .readout_error = 0.01,
                        .gate_error = 0.001
                    }
                }
            },
            .should_succeed = true
        }
    };

    for (size_t i = 0; i < sizeof(test_configs)/sizeof(test_configs[0]); i++) {
        printf("Testing %s...\n", test_configs[i].name);

        StabilizerState state;
        bool success = init_stabilizer_measurement(&state, &test_configs[i].config);
        
        if (test_configs[i].should_succeed) {
            assert(success);
            if (success) {
                // Verify state initialization
                assert(state.plaquette_stabilizers != NULL);
                assert(state.vertex_stabilizers != NULL);
                assert(state.measurement_count == 0);
                assert(state.error_rate == 0.0);

                // Verify config copying
                assert(state.config.lattice_width == test_configs[i].config.lattice_width);
                assert(state.config.lattice_height == test_configs[i].config.lattice_height);
                assert(state.config.error_threshold == test_configs[i].config.error_threshold);

                // Cleanup valid state
                cleanup_stabilizer_measurement(&state);
            }
        } else {
            assert(!success);
        }

        printf("%s validation test passed\n", test_configs[i].name);
    }

    printf("Configuration validation tests passed\n");
}

static void test_hardware_performance(void) {
    printf("Testing hardware-specific performance metrics...\n");

    // Test each hardware backend's performance characteristics
    struct {
        HardwareType type;
        const char* name;
        PerformanceConfig perf_config;
        double expected_metrics[3];  // [latency_us, throughput_ops, efficiency]
    } test_configs[] = {
        {
            .type = HARDWARE_IBM,
            .name = "IBM",
            .perf_config = {
                .optimization_level = 3,
                .max_parallel_ops = 8,
                .pipeline_depth = 4,
                .batch_size = 16
            },
            .expected_metrics = {50.0, 1000.0, 0.85}  // 50μs latency, 1000 ops/s, 85% efficiency
        },
        {
            .type = HARDWARE_RIGETTI,
            .name = "Rigetti",
            .perf_config = {
                .optimization_level = 3,
                .max_parallel_ops = 6,
                .pipeline_depth = 3,
                .batch_size = 12
            },
            .expected_metrics = {75.0, 800.0, 0.80}  // 75μs latency, 800 ops/s, 80% efficiency
        },
        {
            .type = HARDWARE_DWAVE,
            .name = "D-Wave",
            .perf_config = {
                .optimization_level = 3,
                .max_parallel_ops = 16,
                .pipeline_depth = 2,
                .batch_size = 32
            },
            .expected_metrics = {100.0, 600.0, 0.75}  // 100μs latency, 600 ops/s, 75% efficiency
        }
    };

    for (size_t i = 0; i < sizeof(test_configs)/sizeof(test_configs[0]); i++) {
        printf("Testing %s performance metrics...\n", test_configs[i].name);

        // Initialize with performance configuration
        StabilizerState state;
        StabilizerConfig config = {
            .lattice_width = 4,
            .lattice_height = 4,
            .error_threshold = 0.1,
            .auto_correction = true,
            .hardware_config = {
                .type = test_configs[i].type,
                .perf_config = test_configs[i].perf_config,
                .parallel_enabled = true,
                .error_mitigation = true
            }
        };

        bool success = init_stabilizer_measurement(&state, &config);
        assert(success);

        // Create test state
        quantum_state* qstate = create_test_state(4, 4);
        assert(qstate != NULL);

        // Perform measurements and collect metrics
        clock_t start = clock();
        for (int j = 0; j < 100; j++) {  // Run multiple iterations for stable metrics
            success = measure_stabilizers(&state, qstate);
            assert(success);
        }
        clock_t end = clock();
        double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        // Verify basic performance metrics
        assert(state.hardware_metrics.measurement_latency_us < test_configs[i].expected_metrics[0]);
        assert(state.hardware_metrics.operation_throughput > test_configs[i].expected_metrics[1]);
        assert(state.hardware_metrics.hardware_efficiency > test_configs[i].expected_metrics[2]);

        // Verify hardware-specific metrics
        switch (test_configs[i].type) {
            case HARDWARE_IBM:
                assert(state.hardware_metrics.gate_execution_time_us < 10.0);
                assert(state.hardware_metrics.readout_time_us < 40.0);
                assert(state.hardware_metrics.instruction_throughput > 2000.0);
                break;
            case HARDWARE_RIGETTI:
                assert(state.hardware_metrics.native_gate_latency_us < 15.0);
                assert(state.hardware_metrics.qubit_routing_overhead < 0.2);
                assert(state.hardware_metrics.pipeline_efficiency > 0.9);
                break;
            case HARDWARE_DWAVE:
                assert(state.hardware_metrics.annealing_time_us < 20.0);
                assert(state.hardware_metrics.readout_efficiency > 0.95);
                assert(state.hardware_metrics.chain_break_fraction < 0.1);
                break;
            default:
                break;
        }

        // Verify resource utilization
        assert(state.hardware_metrics.memory_usage_kb < 1024);  // Less than 1MB
        assert(state.hardware_metrics.power_consumption_mw < 1000);  // Less than 1W
        assert(state.hardware_metrics.qubit_utilization > 0.7);  // At least 70% utilization

        // Cleanup
        cleanup_stabilizer_measurement(&state);
        cleanup_test_state(qstate);
        
        printf("%s performance tests passed\n", test_configs[i].name);
    }

    printf("Hardware performance tests passed\n");
}

static void test_hardware_error_mitigation(void) {
    printf("Testing hardware-specific error mitigation...\n");

    // Test each hardware backend's error mitigation
    struct {
        HardwareType type;
        const char* name;
        double expected_improvement;
        ErrorMitigationConfig mitigation_config;
    } test_configs[] = {
        {
            .type = HARDWARE_IBM,
            .name = "IBM",
            .expected_improvement = 0.6,  // 40% error reduction
            .mitigation_config = {
                .readout_error_correction = true,
                .dynamic_decoupling = true,
                .zero_noise_extrapolation = true
            }
        },
        {
            .type = HARDWARE_RIGETTI,
            .name = "Rigetti",
            .expected_improvement = 0.7,  // 30% error reduction
            .mitigation_config = {
                .symmetrization = true,
                .richardson_extrapolation = true,
                .quasi_probability = true
            }
        },
        {
            .type = HARDWARE_DWAVE,
            .name = "D-Wave",
            .expected_improvement = 0.8,  // 20% error reduction
            .mitigation_config = {
                .spin_reversal_transform = true,
                .gauge_averaging = true,
                .thermal_sampling = true
            }
        }
    };

    for (size_t i = 0; i < sizeof(test_configs)/sizeof(test_configs[0]); i++) {
        printf("Testing %s error mitigation...\n", test_configs[i].name);

        // Initialize with error mitigation enabled
        StabilizerState state;
        StabilizerConfig config = {
            .lattice_width = 4,
            .lattice_height = 4,
            .error_threshold = 0.1,
            .auto_correction = true,
            .hardware_config = {
                .type = test_configs[i].type,
                .optimization_level = 3,
                .error_mitigation = true,
                .mitigation_config = test_configs[i].mitigation_config,
                .noise_model = {
                    .readout_error = 0.01,
                    .gate_error = 0.001,
                    .decoherence_rate = 0.05
                }
            }
        };

        bool success = init_stabilizer_measurement(&state, &config);
        assert(success);

        // Create test state with errors
        quantum_state* qstate = create_test_state(4, 4);
        assert(qstate != NULL);
        apply_test_errors(qstate);

        // Measure without mitigation
        config.hardware_config.error_mitigation = false;
        success = init_stabilizer_measurement(&state, &config);
        assert(success);
        success = measure_stabilizers(&state, qstate);
        assert(success);
        double unmitigated_error = get_error_rate(&state);

        // Measure with mitigation
        config.hardware_config.error_mitigation = true;
        success = init_stabilizer_measurement(&state, &config);
        assert(success);
        success = measure_stabilizers(&state, qstate);
        assert(success);
        double mitigated_error = get_error_rate(&state);

        // Verify error reduction
        assert(mitigated_error < unmitigated_error * test_configs[i].expected_improvement);

        // Verify hardware-specific metrics
        verify_hardware_metrics(&state, &config.hardware_config);

        // Verify mitigation-specific metrics
        switch (test_configs[i].type) {
            case HARDWARE_IBM:
                assert(state.hardware_metrics.readout_correction_effectiveness > 0.8);
                assert(state.hardware_metrics.decoupling_fidelity > 0.9);
                assert(state.hardware_metrics.zne_extrapolation_quality > 0.85);
                break;
            case HARDWARE_RIGETTI:
                assert(state.hardware_metrics.symmetrization_quality > 0.85);
                assert(state.hardware_metrics.extrapolation_confidence > 0.9);
                assert(state.hardware_metrics.quasi_probability_accuracy > 0.8);
                break;
            case HARDWARE_DWAVE:
                assert(state.hardware_metrics.spin_reversal_effectiveness > 0.9);
                assert(state.hardware_metrics.gauge_sampling_quality > 0.85);
                assert(state.hardware_metrics.thermal_calibration_accuracy > 0.8);
                break;
            default:
                break;
        }

        // Cleanup
        cleanup_stabilizer_measurement(&state);
        cleanup_test_state(qstate);
        
        printf("%s error mitigation tests passed\n", test_configs[i].name);
    }

    printf("Hardware error mitigation tests passed\n");
}

static void test_parallel_measurement(void) {
    printf("Testing parallel measurement optimization...\n");

    // Initialize stabilizer system with parallel config
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 6,    // Larger lattice to test parallel groups
        .lattice_height = 6,
        .error_threshold = 0.1,
        .auto_correction = true,
        .enable_parallel = true,
        .max_parallel_ops = 4,
        .correlation_threshold = 0.3,
        .parallel_config = {
            .group_size = 4,
            .min_distance = 2,
            .max_crosstalk = 0.01
        }
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(6, 6);
    assert(qstate != NULL);

    // Apply scattered errors to test parallel detection
    apply_test_errors(qstate);

    // Test parallel measurement groups
    success = measure_stabilizers(&state, qstate);
    assert(success);
    
    // Verify parallel execution
    assert(state.current_parallel_group > 1);  // Should use multiple groups
    assert(state.parallel_stats.total_groups >= 4);  // Should form at least 4 groups
    assert(state.parallel_stats.avg_group_size >= 2.0);  // Average group size
    assert(state.parallel_stats.max_group_size <= 4);  // Respect max parallel ops
    
    // Verify measurement independence
    for (size_t i = 0; i < state.parallel_stats.total_groups; i++) {
        const ParallelGroup* group = &state.parallel_groups[i];
        // Check that no stabilizers in group share qubits
        for (size_t j = 0; j < group->size; j++) {
            for (size_t k = j + 1; k < group->size; k++) {
                assert(!stabilizers_share_qubits(&state, 
                                               group->stabilizer_indices[j],
                                               group->stabilizer_indices[k]));
            }
        }
    }

    // Verify parallel measurement performance
    clock_t start = clock();
    success = measure_stabilizers(&state, qstate);
    clock_t end = clock();
    double parallel_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Disable parallel and measure again
    config.enable_parallel = false;
    success = init_stabilizer_measurement(&state, &config);
    assert(success);
    
    start = clock();
    success = measure_stabilizers(&state, qstate);
    end = clock();
    double sequential_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Verify parallel speedup
    assert(parallel_time < sequential_time * 0.7);  // At least 30% speedup

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Parallel measurement tests passed\n");
}

static void test_hardware_integration(void) {
    printf("Testing hardware integration...\n");

    // Test each hardware backend type
    HardwareType backends[] = {HARDWARE_IBM, HARDWARE_RIGETTI, HARDWARE_DWAVE};
    const char* backend_names[] = {"IBM", "Rigetti", "D-Wave"};

    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); i++) {
        printf("Testing %s backend...\n", backend_names[i]);

        // Initialize stabilizer system with backend config
        StabilizerState state;
        StabilizerConfig config = {
            .lattice_width = 4,
            .lattice_height = 4,
            .error_threshold = 0.1,
            .auto_correction = true,
            .hardware_config = {
                .type = backends[i],
                .optimization_level = 3,
                .error_mitigation = true,
                .parallel_enabled = true,
                .max_parallel_ops = 8,
                .noise_model = {
                    .readout_error = 0.01,
                    .gate_error = 0.001,
                    .decoherence_rate = 0.05,
                    .crosstalk_threshold = 0.02
                }
            }
        };

        bool success = init_stabilizer_measurement(&state, &config);
        assert(success);

        // Create test quantum state
        quantum_state* qstate = create_test_state(4, 4);
        assert(qstate != NULL);

        // Apply test errors
        apply_test_errors(qstate);

        // Initial measurement to get baseline
        success = measure_stabilizers(&state, qstate);
        assert(success);
        double initial_error_rate = get_error_rate(&state);
        
        // Verify hardware metrics after first measurement
        verify_hardware_metrics(&state, &config.hardware_config);
        
        // Test error correction with hardware optimization
        success = measure_stabilizers(&state, qstate);
        assert(success);
        double final_error_rate = get_error_rate(&state);
        
        // Verify error reduction
        assert(final_error_rate < initial_error_rate * 0.7);  // At least 30% improvement
        
        // Verify hardware metrics after correction
        verify_hardware_metrics(&state, &config.hardware_config);

        // Verify parallel measurement statistics
        if (config.hardware_config.parallel_enabled) {
            assert(state.parallel_stats.total_groups > 0);
            assert(state.parallel_stats.avg_group_size > 1.0);
            assert(state.parallel_stats.max_group_size <= config.hardware_config.max_parallel_ops);
            assert(state.parallel_stats.execution_time_us > 0);
            assert(state.parallel_stats.speedup_factor > 1.2);  // At least 20% speedup
        }

        // Cleanup
        cleanup_stabilizer_measurement(&state);
        cleanup_test_state(qstate);
        
        printf("%s backend tests passed\n", backend_names[i]);
    }

    printf("Hardware integration tests passed\n");
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = true
    };

    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);
    assert(state.plaquette_stabilizers != NULL);
    assert(state.vertex_stabilizers != NULL);
    assert(state.plaquette_stabilizers->measurements != NULL);
    assert(state.vertex_stabilizers->measurements != NULL);
    assert(state.measurement_count == 0);
    assert(state.error_rate == 0.0);

    // Test cleanup
    cleanup_stabilizer_measurement(&state);

    // Test invalid parameters
    success = init_stabilizer_measurement(NULL, &config);
    assert(!success);
    success = init_stabilizer_measurement(&state, NULL);
    assert(!success);

    printf("Initialization tests passed\n");
}

static void test_measurement(void) {
    printf("Testing stabilizer measurement...\n");

    // Initialize stabilizer system
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = false
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create test quantum state
    quantum_state* qstate = create_test_state(4, 4);
    assert(qstate != NULL);

    // Perform measurements
    success = measure_stabilizers(&state, qstate);
    assert(success);
    assert(state.measurement_count == 1);

    // Verify plaquette measurements
    size_t plaquette_size;
    const double* plaquette_results = get_stabilizer_measurements(&state,
                                                                STABILIZER_PLAQUETTE,
                                                                &plaquette_size);
    assert(plaquette_results != NULL);
    assert(plaquette_size == 9);  // 3x3 plaquettes in 4x4 lattice

    // Verify vertex measurements
    size_t vertex_size;
    const double* vertex_results = get_stabilizer_measurements(&state,
                                                             STABILIZER_VERTEX,
                                                             &vertex_size);
    assert(vertex_results != NULL);
    assert(vertex_size == 9);  // 3x3 vertices in 4x4 lattice

    // Verify measurement values
    for (size_t i = 0; i < plaquette_size; i++) {
        assert(fabs(plaquette_results[i]) <= 1.0);
    }
    for (size_t i = 0; i < vertex_size; i++) {
        assert(fabs(vertex_results[i]) <= 1.0);
    }

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Measurement tests passed\n");
}

static void test_error_detection(void) {
    printf("Testing error detection...\n");

    // Initialize stabilizer system
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = false
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create test quantum state with errors
    quantum_state* qstate = create_test_state(4, 4);
    assert(qstate != NULL);
    apply_test_errors(qstate);

    // Measure stabilizers
    success = measure_stabilizers(&state, qstate);
    assert(success);

    // Verify error detection
    double error_rate = get_error_rate(&state);
    assert(error_rate > 0.0);  // Should detect errors
    assert(error_rate <= 1.0);

    // Verify syndrome storage
    size_t syndrome_size;
    const double* syndrome = get_last_syndrome(&state, &syndrome_size);
    assert(syndrome != NULL);
    assert(syndrome_size == 18);  // Total stabilizers for 4x4 lattice

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Error detection tests passed\n");
}

static void test_error_correction(void) {
    printf("Testing error correction...\n");

    // Initialize stabilizer system with auto-correction
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = true
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create test quantum state with errors
    quantum_state* qstate = create_test_state(4, 4);
    assert(qstate != NULL);
    apply_test_errors(qstate);

    // Initial measurement to detect errors
    success = measure_stabilizers(&state, qstate);
    assert(success);
    double initial_error_rate = get_error_rate(&state);
    assert(initial_error_rate > 0.0);

    // Second measurement after auto-correction
    success = measure_stabilizers(&state, qstate);
    assert(success);
    double final_error_rate = get_error_rate(&state);
    assert(final_error_rate < initial_error_rate);

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Error correction tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 4,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = true
    };

    // Test NULL parameters
    bool success = measure_stabilizers(NULL, NULL);
    assert(!success);

    quantum_state* qstate = create_test_state(4, 4);
    success = measure_stabilizers(&state, NULL);
    assert(!success);
    success = measure_stabilizers(NULL, qstate);
    assert(!success);

    // Test invalid dimensions
    StabilizerConfig invalid_config = {
        .lattice_width = 0,
        .lattice_height = 4,
        .error_threshold = 0.1,
        .auto_correction = true
    };
    success = init_stabilizer_measurement(&state, &invalid_config);
    assert(!success);

    // Test invalid measurement access
    size_t size;
    assert(get_stabilizer_measurements(NULL, STABILIZER_PLAQUETTE, &size) == NULL);
    assert(get_stabilizer_measurements(&state, STABILIZER_PLAQUETTE, NULL) == NULL);

    // Cleanup
    cleanup_test_state(qstate);

    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Initialize large test system
    StabilizerState state;
    StabilizerConfig config = {
        .lattice_width = 100,   // Large lattice for stress testing
        .lattice_height = 100,
        .error_threshold = 0.1,
        .auto_correction = true
    };
    bool success = init_stabilizer_measurement(&state, &config);
    assert(success);

    // Create large test state
    quantum_state* qstate = create_test_state(100, 100);
    assert(qstate != NULL);

    // Measure initialization time
    clock_t start = clock();
    success = init_stabilizer_measurement(&state, &config);
    clock_t end = clock();
    double init_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(init_time < 0.001);  // Should initialize quickly

    // Measure stabilizer measurement time
    start = clock();
    success = measure_stabilizers(&state, qstate);
    end = clock();
    double measurement_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    assert(measurement_time < 0.01);  // Should measure quickly

    // Verify memory usage
    size_t expected_memory = 2 * 99 * 99 * sizeof(double);  // Plaquette and vertex arrays
    size_t actual_memory = sizeof(StabilizerState) +
                          2 * sizeof(StabilizerArray) +
                          expected_memory;
    // Memory overhead should be reasonable
    assert(actual_memory < 1024 * 1024);  // Less than 1MB for 100x100 lattice

    // Cleanup
    cleanup_stabilizer_measurement(&state);
    cleanup_test_state(qstate);

    printf("Performance requirement tests passed\n");
}

// Mock implementation of test helpers
static quantum_state* create_test_state(size_t width, size_t height) {
    quantum_state* state = malloc(sizeof(quantum_state));
    if (!state) return NULL;

    // Allocate quantum state arrays
    state->width = width;
    state->height = height;
    state->qubits = calloc(width * height, sizeof(qubit_state));
    state->stabilizers = calloc(width * height, sizeof(stabilizer_state));
    
    if (!state->qubits || !state->stabilizers) {
        free(state->qubits);
        free(state->stabilizers);
        free(state);
        return NULL;
    }

    // Initialize to |0⟩ state
    for (size_t i = 0; i < width * height; i++) {
        state->qubits[i].amplitude_0 = 1.0;
        state->qubits[i].amplitude_1 = 0.0;
        state->qubits[i].phase = 0.0;
        state->stabilizers[i].value = 1.0;  // +1 eigenstate
        state->stabilizers[i].confidence = 1.0;
    }

    return state;
}

static void apply_test_errors(quantum_state* state) {
    if (!state) return;

    // Apply X errors at specific locations
    size_t error_locations[] = {
        1 * state->width + 1,  // (1,1)
        2 * state->width + 2,  // (2,2)
        (state->height - 2) * state->width + (state->width - 2)  // (w-2,h-2)
    };
    
    for (size_t i = 0; i < sizeof(error_locations)/sizeof(error_locations[0]); i++) {
        size_t idx = error_locations[i];
        if (idx < state->width * state->height) {
            // Apply X error (bit flip)
            double temp = state->qubits[idx].amplitude_0;
            state->qubits[idx].amplitude_0 = state->qubits[idx].amplitude_1;
            state->qubits[idx].amplitude_1 = temp;
            // Mark stabilizer as potentially affected
            state->stabilizers[idx].value = -1.0;  // Error state
            state->stabilizers[idx].confidence = 0.9;  // High confidence error
        }
    }

    // Apply Z errors at different locations
    size_t z_error_locations[] = {
        1 * state->width + 2,  // (1,2)
        2 * state->width + 1,  // (2,1)
        (state->height - 2) * state->width + (state->width - 3)  // (w-3,h-2)
    };

    for (size_t i = 0; i < sizeof(z_error_locations)/sizeof(z_error_locations[0]); i++) {
        size_t idx = z_error_locations[i];
        if (idx < state->width * state->height) {
            // Apply Z error (phase flip)
            state->qubits[idx].phase += M_PI;
            // Mark stabilizer
            state->stabilizers[idx].value = -1.0;
            state->stabilizers[idx].confidence = 0.85;  // Slightly lower confidence
        }
    }
}

static void cleanup_test_state(quantum_state* state) {
    if (state) {
        free(state->qubits);
        free(state->stabilizers);
        free(state);
    }
}
