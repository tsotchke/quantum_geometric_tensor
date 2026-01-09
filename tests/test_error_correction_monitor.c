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
                                                   size_t total_successes) {
    CorrectionState* state = malloc(sizeof(CorrectionState));
    state->success_rate = success_rate;
    state->total_corrections = total_corrections;
    state->total_successes = total_successes;
    return state;
}

static void cleanup_test_correction_state(CorrectionState* state) {
    free(state);
}

static bool compare_metrics(const CorrectionMetrics* a,
                          const CorrectionMetrics* b,
                          double epsilon) {
    return fabs(a->success_rate - b->success_rate) < epsilon &&
           a->total_corrections == b->total_corrections &&
           a->total_successes == b->total_successes;
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

// Test cases
static void test_initialization(void) {
    printf("Testing monitor initialization...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true,
        .track_resources = false,
        .pattern_detection = false,
        .update_interval_ms = 1000
    };

    MonitorState state;
    bool success = init_correction_monitor(&state, &config);
    assert(success && "Monitor initialization failed");
    assert(state.config.history_length == 100);
    assert(state.config.alert_threshold == 0.9);

    cleanup_correction_monitor(&state);
    printf("Monitor initialization test passed\n");
}

static void test_metrics_recording(void) {
    printf("Testing metrics recording...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = false
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    CorrectionState* correction_state = create_test_correction_state(0.95, 100, 95);
    bool success = record_correction_metrics(&state, correction_state, 0.001);
    assert(success);

    // Verify metrics were recorded
    CorrectionMetrics metrics;
    success = get_current_metrics(&state, &metrics);
    assert(success);
    assert(fabs(metrics.success_rate - 0.95) < 1e-6);

    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Metrics recording test passed\n");
}

static void test_alert_generation(void) {
    printf("Testing alert generation...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.95,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = true
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Record metrics below threshold
    CorrectionState* correction_state = create_test_correction_state(0.90, 100, 90);
    record_correction_metrics(&state, correction_state, 0.001);

    // Check if alert was generated
    size_t num_alerts;
    Alert* alerts = get_pending_alerts(&state, &num_alerts);
    assert(num_alerts > 0);
    assert(alerts[0].level == ALERT_WARNING || alerts[0].level == ALERT_ERROR);

    free(alerts);
    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Alert generation test passed\n");
}

static void test_report_generation(void) {
    printf("Testing report generation...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = false
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Record some metrics
    for (size_t i = 0; i < 10; i++) {
        CorrectionState* correction_state = create_test_correction_state(0.95 - i*0.01, 100 + i*10, 95 + i*9);
        record_correction_metrics(&state, correction_state, 0.001);
        cleanup_test_correction_state(correction_state);
    }

    // Generate report
    char* report = generate_monitoring_report(&state);
    assert(report != NULL);
    assert(strlen(report) > 0);

    free(report);
    cleanup_correction_monitor(&state);
    printf("Report generation test passed\n");
}

static void test_health_checking(void) {
    printf("Testing health checking...\n");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = false
    };

    MonitorState state;
    init_correction_monitor(&state, &config);

    // Record good metrics
    CorrectionState* correction_state = create_test_correction_state(0.98, 1000, 980);
    record_correction_metrics(&state, correction_state, 0.001);

    // Check health
    HealthStatus health = check_system_health(&state);
    assert(health == HEALTH_GOOD || health == HEALTH_EXCELLENT);

    // Record poor metrics
    correction_state->success_rate = 0.70;
    correction_state->total_corrections = 1100;
    correction_state->total_successes = 770;
    record_correction_metrics(&state, correction_state, 0.001);

    health = check_system_health(&state);
    assert(health == HEALTH_POOR || health == HEALTH_CRITICAL);

    cleanup_test_correction_state(correction_state);
    cleanup_correction_monitor(&state);
    printf("Health checking test passed\n");
}

static void test_error_handling(void) {
    printf("Testing error handling...\n");

    MonitorState state;
    bool success = init_correction_monitor(&state, NULL);
    assert(!success && "Should fail with NULL config");

    MonitorConfig config = {
        .history_length = 100,
        .alert_threshold = 0.9,
        .log_to_file = false,
        .log_path = NULL,
        .real_time_alerts = false
    };

    init_correction_monitor(&state, &config);
    success = record_correction_metrics(&state, NULL, 0.001);
    assert(!success && "Should fail with NULL correction state");

    cleanup_correction_monitor(&state);
    printf("Error handling test passed\n");
}

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
    assert(monitor_detect_performance_degradation(&state));
    assert(!monitor_detect_performance_improvement(&state));

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
        (void)record_resource_metrics(&state, &metrics);
    }

    // Verify resource tracking
    ResourceStats stats;
    (void)get_resource_statistics(&state, &stats);
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
    ErrorPattern* detected = monitor_detect_error_patterns(&state, &num_patterns);
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
    (void)start_real_time_monitoring(&state);

    // Simulate correction cycles
    CorrectionState* correction_state = create_test_correction_state(0.95, 100, 95);
    simulate_correction_cycle(&state, correction_state, 10);

    // Verify monitoring updates
    MonitoringStats stats;
    bool success = get_monitoring_stats(&state, &stats);
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
    qstate->coordinates = calloc(65536, sizeof(ComplexFloat));  // 2^16

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
        // Extract and correct errors with monitoring
        size_t num_syndromes = extract_error_syndromes(qstate, &syndrome_config, graph);

        success = find_minimum_weight_matching(graph, &syndrome_config);
        assert(success);

        // Create correction state with available data
        CorrectionState correction_state = {
            .success_rate = 0.95,
            .total_corrections = i + 1,
            .total_successes = i + 1
        };

        success = record_correction_metrics(&monitor_state, &correction_state, 0.001);
        assert(success);
        (void)num_syndromes;  // Suppress unused warning
    }

    // Verify pipeline metrics
    PipelineStats stats;
    success = get_pipeline_statistics(&monitor_state, &stats);
    assert(success);
    assert(stats.total_cycles > 0);
    assert(stats.success_rate > 0);
    assert(stats.avg_cycle_time > 0);

    cleanup_matching_graph(graph);
    free(qstate->coordinates);
    free(qstate);
    cleanup_correction_monitor(&monitor_state);
    printf("Pipeline integration test passed\n");
}

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
