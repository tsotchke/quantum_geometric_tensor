/**
 * @file test_error_correction_monitor.c
 * @brief Tests for error correction monitoring system
 */

#include "quantum_geometric/monitoring/error_correction_monitor.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test helper functions
static CorrectionState* create_test_correction_state(double success_rate,
                                                   size_t total_corrections,
                                                   size_t total_successes);
static void cleanup_test_correction_state(CorrectionState* state);
static bool compare_metrics(const CorrectionMetrics* a,
                          const CorrectionMetrics* b,
                          double epsilon);
static void simulate_correction_cycle(MonitorState* state,
                                    CorrectionState* correction_state,
                                    size_t num_cycles);

// Test cases
static void test_initialization(void);
static void test_metrics_recording(void);
static void test_alert_generation(void);
static void test_report_generation(void);
static void test_health_checking(void);
static void test_error_handling(void);
static void test_performance_trends(void);
static void test_resource_utilization(void);
static void test_error_patterns(void);
static void test_real_time_monitoring(void);
static void test_pipeline_integration(void);

int main(void) {
    printf("Running error correction monitor tests...\n\n");

    test_initialization();
    test_metrics_recording();
    test_alert_generation();
    test_report_generation();
    test_health_checking();
    test_error_handling();
    test_performance_trends();
    test_resource_utilization();
    test_error_patterns();
    test_real_time_monitoring();
    test_pipeline_integration();

    printf("\nAll error correction monitor tests passed!\n");
    return 0;
}

[Previous test implementations remain unchanged...]

static void test_performance_trends(void) {
    printf("Testing performance trend analysis...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Simulate declining performance trend
    CorrectionState* correction_state = create_test_correction_state(0.95, 0, 0);
    for (size_t i = 0; i < 50; i++) {
        correction_state->success_rate = 0.95 - (double)i/200;  // Gradual decline
        correction_state->total_corrections = 100 + i*10;
        correction_state->total_successes = (size_t)((100 + i*10) * 
                                                    correction_state->success_rate);
        
        bool success = record_correction_metrics(&state, correction_state, 0.001);
        assert(success);
    }

    // Analyze trends
    PerformanceTrend trend;
    bool success = analyze_performance_trend(&state, &trend);
    assert(success);
    assert(trend.direction == TREND_DECLINING);
    assert(trend.rate < 0);
    assert(trend.confidence > 0.9);

    // Test trend detection thresholds
    assert(detect_performance_degradation(&state));
    assert(!detect_performance_improvement(&state));

    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Performance trend analysis test passed\n");
}

static void test_resource_utilization(void) {
    printf("Testing resource utilization tracking...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true,
        .track_resources = true
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Simulate resource usage
    CorrectionState* correction_state = create_test_correction_state(0.95, 100, 95);
    ResourceMetrics metrics = {
        .cpu_usage = 0.75,
        .memory_usage = 0.60,
        .gpu_usage = 0.80,
        .network_bandwidth = 0.40
    };

    for (size_t i = 0; i < 10; i++) {
        metrics.cpu_usage += 0.02;
        metrics.memory_usage += 0.03;
        bool success = record_resource_metrics(&state, &metrics);
        assert(success);
    }

    // Verify resource tracking
    ResourceStats stats;
    bool success = get_resource_statistics(&state, &stats);
    assert(success);
    assert(stats.peak_cpu_usage > 0.9);
    assert(stats.peak_memory_usage > 0.8);
    assert(stats.avg_cpu_usage > 0.8);
    assert(stats.avg_memory_usage > 0.7);

    // Test resource alerts
    assert(check_resource_thresholds(&state));

    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Resource utilization tracking test passed\n");
}

static void test_error_patterns(void) {
    printf("Testing error pattern detection...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true,
        .pattern_detection = true
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Simulate repeating error pattern
    CorrectionState* correction_state = create_test_correction_state(0.95, 100, 95);
    ErrorPattern pattern = {
        .locations = {1, 2, 5, 6},
        .types = {ERROR_X, ERROR_X, ERROR_Z, ERROR_Z},
        .size = 4,
        .frequency = 0.0,
        .confidence = 0.0
    };

    for (size_t i = 0; i < 20; i++) {
        record_error_pattern(&state, &pattern);
    }

    // Test pattern detection
    size_t num_patterns;
    ErrorPattern* detected = detect_error_patterns(&state, &num_patterns);
    assert(detected != NULL);
    assert(num_patterns > 0);
    assert(detected[0].frequency > 0.8);
    assert(detected[0].confidence > 0.9);

    // Verify pattern matching
    assert(match_error_pattern(&state, &pattern));

    free(detected);
    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Error pattern detection test passed\n");
}

static void test_real_time_monitoring(void) {
    printf("Testing real-time monitoring...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true,
        .update_interval_ms = 100
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Start real-time monitoring
    bool success = start_real_time_monitoring(&state);
    assert(success);

    // Simulate correction cycles
    CorrectionState* correction_state = create_test_correction_state(0.95, 100, 95);
    simulate_correction_cycle(&state, correction_state, 10);

    // Verify monitoring updates
    MonitoringStats stats;
    success = get_monitoring_stats(&state, &stats);
    assert(success);
    assert(stats.update_count > 0);
    assert(stats.last_update_time > 0);
    assert(stats.avg_update_interval > 0);

    // Stop monitoring
    success = stop_real_time_monitoring(&state);
    assert(success);

    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Real-time monitoring test passed\n");
}

static void test_pipeline_integration(void) {
    printf("Testing pipeline integration...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true
    };

    MonitorState monitor_state;
    init_correction_monitor(&monitor_state, &config);

    // Create test quantum state and error syndrome
    quantum_state_t* qstate = malloc(sizeof(quantum_state_t));
    qstate->num_qubits = 16;
    qstate->amplitudes = calloc(32, sizeof(double complex));
    
    SyndromeConfig syndrome_config = {
        .detection_threshold = 0.1,
        .weight_scale_factor = 1.0,
        .max_matching_iterations = 100,
        .use_boundary_matching = true
    };

    MatchingGraph* graph;
    bool success = init_matching_graph(16, 32, &graph);
    assert(success);

    // Run correction pipeline with monitoring
    for (size_t i = 0; i < 10; i++) {
        // Inject random errors
        for (size_t j = 0; j < 3; j++) {
            size_t loc = rand() % 16;
            error_type_t type = rand() % 3;
            inject_error(qstate, loc, type);
        }

        // Extract and correct errors with monitoring
        size_t num_syndromes;
        success = extract_error_syndromes(qstate, &syndrome_config, graph, &num_syndromes);
        assert(success);

        success = find_minimum_weight_matching(graph, &syndrome_config);
        assert(success);

        CorrectionState correction_state = {
            .success_rate = graph->correction_success_rate,
            .total_corrections = graph->total_corrections,
            .total_successes = graph->successful_corrections
        };

        success = record_correction_metrics(&monitor_state, &correction_state, 0.001);
        assert(success);
    }

    // Verify pipeline metrics
    PipelineStats stats;
    success = get_pipeline_statistics(&monitor_state, &stats);
    assert(success);
    assert(stats.total_cycles > 0);
    assert(stats.success_rate > 0);
    assert(stats.avg_cycle_time > 0);

    cleanup_matching_graph(graph);
    free(qstate->amplitudes);
    free(qstate);
    cleanup_correction_monitor(&monitor_state);
    printf("Pipeline integration test passed\n");
}

static void simulate_correction_cycle(MonitorState* state,
                                   CorrectionState* correction_state,
                                   size_t num_cycles) {
    struct timespec sleep_time = {0, 50000000};  // 50ms

    for (size_t i = 0; i < num_cycles; i++) {
        correction_state->total_corrections++;
        if ((double)rand() / RAND_MAX < correction_state->success_rate) {
            correction_state->total_successes++;
        }
        record_correction_metrics(state, correction_state, 0.001);
        nanosleep(&sleep_time, NULL);
    }
}

[Previous helper functions remain unchanged...]
