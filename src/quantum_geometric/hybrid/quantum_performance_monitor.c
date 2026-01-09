/**
 * @file quantum_performance_monitor.c
 * @brief Performance monitoring for hybrid quantum-classical operations
 *
 * This module tracks system performance metrics, detects bottlenecks,
 * and recommends optimization strategies for quantum operations.
 */

#include "quantum_geometric/hybrid/quantum_performance_monitor.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Performance thresholds
#define MIN_PERFORMANCE_RATIO 0.8
#define MAX_OVERHEAD_RATIO 0.2
#define MIN_EFFICIENCY_RATIO 0.7
#define MAX_LATENCY_MS 10.0

// Monitoring parameters
#define METRIC_HISTORY_SIZE 1000
#define UPDATE_INTERVAL_MS 100
#define SAMPLE_WINDOW_SIZE 100

// ============================================================================
// Internal structures
// ============================================================================

// Internal performance metrics (matches header's PerformanceSnapshot)
typedef struct {
    double total_runtime;
    double compute_time;
    double network_time;
    double memory_time;
    double memory_usage;
    double network_usage;
    double cpu_usage;
    double gpu_usage;
    double throughput;
    double latency;
    double efficiency;
    double overhead;
} InternalMetrics;

typedef struct {
    double memory_utilization;
    double network_utilization;
    double compute_utilization;
    bool memory_bound;
    bool network_bound;
    bool compute_bound;
    double memory_headroom;
    double network_headroom;
    double compute_headroom;
} SystemAnalysis;

struct MetricsTracker {
    InternalMetrics* history;
    size_t history_size;
    size_t current_index;
    size_t sample_count;
    size_t update_interval_ms;
    double last_update_time;
};

struct SystemAnalyzer {
    double analyze_interval;
    double threshold;
    SystemAnalysis current;
    SystemAnalysis baseline;
    bool has_baseline;
};

struct OptimizationEngine {
    OptimizationStrategy strategy;
    bool aggressive;
    double last_improvement;
    size_t optimization_count;
};

struct PerformanceState {
    InternalMetrics* metric_history;
    size_t history_size;
    size_t current_index;
    SystemAnalysis* current_analysis;
    SystemAnalysis* baseline_analysis;
    bool needs_optimization;
    double optimization_threshold;
    OptimizationStrategy strategy;
};

// ============================================================================
// Forward declarations
// ============================================================================

static InternalMetrics* get_current_metrics(PerformanceMonitor* monitor);
static void update_timing_metrics(InternalMetrics* metrics, const QuantumOperation* op);
static void update_resource_metrics(InternalMetrics* metrics, const QuantumOperation* op);
static void update_perf_metrics(InternalMetrics* metrics, const QuantumOperation* op);
static void add_to_history(PerformanceState* state, const InternalMetrics* metrics);
static void analyze_performance(PerformanceMonitor* monitor, const InternalMetrics* metrics);
static void analyze_resource_utilization(SystemAnalysis* analysis, const InternalMetrics* metrics);
static void detect_bottlenecks(SystemAnalysis* analysis, const InternalMetrics* metrics, const SystemAnalysis* baseline);
static void identify_optimizations(SystemAnalysis* analysis, const InternalMetrics* metrics, OptimizationEngine* optimizer);
static void update_optimization_state(PerformanceState* state, const SystemAnalysis* analysis);
static double get_current_time_ms(void);

// ============================================================================
// System resource queries
// ============================================================================

double get_total_memory(void) {
#ifdef __APPLE__
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t physical_memory;
    size_t length = sizeof(physical_memory);
    if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
        return (double)physical_memory;
    }
#endif
    // Default to 8GB if unable to determine
    return 8.0 * 1024.0 * 1024.0 * 1024.0;
}

double get_network_bandwidth(void) {
    // Return typical network bandwidth in bytes/sec (1 Gbps)
    return 1000.0 * 1024.0 * 1024.0 / 8.0;
}

static double get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// ============================================================================
// Metrics tracker implementation
// ============================================================================

MetricsTracker* init_metrics_tracker(size_t history_size, size_t update_interval) {
    MetricsTracker* tracker = calloc(1, sizeof(MetricsTracker));
    if (!tracker) return NULL;

    tracker->history_size = history_size > 0 ? history_size : METRIC_HISTORY_SIZE;
    tracker->history = calloc(tracker->history_size, sizeof(InternalMetrics));
    if (!tracker->history) {
        free(tracker);
        return NULL;
    }

    tracker->current_index = 0;
    tracker->sample_count = 0;
    tracker->update_interval_ms = update_interval > 0 ? update_interval : UPDATE_INTERVAL_MS;
    tracker->last_update_time = get_current_time_ms();

    return tracker;
}

void cleanup_metrics_tracker(MetricsTracker* tracker) {
    if (!tracker) return;
    free(tracker->history);
    free(tracker);
}

// ============================================================================
// System analyzer implementation
// ============================================================================

SystemAnalyzer* init_system_analyzer(double interval, double threshold) {
    SystemAnalyzer* analyzer = calloc(1, sizeof(SystemAnalyzer));
    if (!analyzer) return NULL;

    analyzer->analyze_interval = interval > 0 ? interval : 1000.0;  // 1 second default
    analyzer->threshold = threshold > 0 ? threshold : 0.8;
    analyzer->has_baseline = false;
    memset(&analyzer->current, 0, sizeof(SystemAnalysis));
    memset(&analyzer->baseline, 0, sizeof(SystemAnalysis));

    return analyzer;
}

void cleanup_system_analyzer(SystemAnalyzer* analyzer) {
    free(analyzer);
}

// ============================================================================
// Optimization engine implementation
// ============================================================================

OptimizationEngine* init_optimization_engine(OptimizationStrategy strategy, bool aggressive) {
    OptimizationEngine* engine = calloc(1, sizeof(OptimizationEngine));
    if (!engine) return NULL;

    engine->strategy = strategy;
    engine->aggressive = aggressive;
    engine->last_improvement = 0.0;
    engine->optimization_count = 0;

    return engine;
}

void cleanup_optimization_engine(OptimizationEngine* engine) {
    free(engine);
}

// ============================================================================
// Performance state implementation
// ============================================================================

PerformanceState* init_performance_state(void) {
    PerformanceState* state = calloc(1, sizeof(PerformanceState));
    if (!state) return NULL;

    state->history_size = METRIC_HISTORY_SIZE;
    state->metric_history = calloc(state->history_size, sizeof(InternalMetrics));
    state->current_analysis = calloc(1, sizeof(SystemAnalysis));
    state->baseline_analysis = calloc(1, sizeof(SystemAnalysis));

    if (!state->metric_history || !state->current_analysis || !state->baseline_analysis) {
        free(state->metric_history);
        free(state->current_analysis);
        free(state->baseline_analysis);
        free(state);
        return NULL;
    }

    state->current_index = 0;
    state->needs_optimization = false;
    state->optimization_threshold = 0.0;
    state->strategy = OPTIMIZE_NONE;

    return state;
}

void cleanup_performance_state(PerformanceState* state) {
    if (!state) return;
    free(state->metric_history);
    free(state->current_analysis);
    free(state->baseline_analysis);
    free(state);
}

// ============================================================================
// Performance monitor implementation
// ============================================================================

PerformanceMonitor* qpm_init(const MonitorConfig* config) {
    if (!config) return NULL;

    PerformanceMonitor* monitor = calloc(1, sizeof(PerformanceMonitor));
    if (!monitor) return NULL;

    monitor->metrics = init_metrics_tracker(METRIC_HISTORY_SIZE, UPDATE_INTERVAL_MS);
    monitor->analyzer = init_system_analyzer(config->analyze_interval, config->threshold);
    monitor->optimizer = init_optimization_engine(config->strategy, config->aggressive);
    monitor->state = init_performance_state();

    if (!monitor->metrics || !monitor->analyzer || !monitor->optimizer || !monitor->state) {
        qpm_cleanup(monitor);
        return NULL;
    }

    return monitor;
}

void qpm_cleanup(PerformanceMonitor* monitor) {
    if (!monitor) return;

    cleanup_metrics_tracker(monitor->metrics);
    cleanup_system_analyzer(monitor->analyzer);
    cleanup_optimization_engine(monitor->optimizer);
    cleanup_performance_state(monitor->state);
    free(monitor);
}

int qpm_update(PerformanceMonitor* monitor, const QuantumOperation* op) {
    if (!monitor || !op) return -1;

    // Get current metrics
    InternalMetrics* metrics = get_current_metrics(monitor);
    if (!metrics) return -1;

    // Update timing metrics
    update_timing_metrics(metrics, op);

    // Update resource metrics
    update_resource_metrics(metrics, op);

    // Update performance metrics
    update_perf_metrics(metrics, op);

    // Add to history
    add_to_history(monitor->state, metrics);

    // Analyze system state
    analyze_performance(monitor, metrics);

    return 0;
}

PerformanceSnapshot qpm_get_snapshot(const PerformanceMonitor* monitor) {
    PerformanceSnapshot snapshot = {0};

    if (!monitor || !monitor->state) return snapshot;

    // Get most recent metrics from history
    size_t idx = (monitor->state->current_index > 0)
                 ? monitor->state->current_index - 1
                 : monitor->state->history_size - 1;

    const InternalMetrics* m = &monitor->state->metric_history[idx];

    snapshot.total_runtime = m->total_runtime;
    snapshot.compute_time = m->compute_time;
    snapshot.network_time = m->network_time;
    snapshot.memory_time = m->memory_time;
    snapshot.throughput = m->throughput;
    snapshot.latency = m->latency;
    snapshot.efficiency = m->efficiency;
    snapshot.overhead = m->overhead;

    return snapshot;
}

OptimizationStrategy qpm_get_strategy(const PerformanceMonitor* monitor) {
    if (!monitor || !monitor->state) return OPTIMIZE_NONE;
    return monitor->state->strategy;
}

bool qpm_needs_optimization(const PerformanceMonitor* monitor) {
    if (!monitor || !monitor->state) return false;
    return monitor->state->needs_optimization;
}

// ============================================================================
// Static helper implementations
// ============================================================================

static InternalMetrics* get_current_metrics(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->metrics) return NULL;

    size_t idx = monitor->metrics->current_index;
    return &monitor->metrics->history[idx];
}

// Helper to estimate qubit count from an operation
static size_t estimate_operation_qubits(const QuantumOperation* op) {
    if (!op) return 1;

    switch (op->type) {
        case OPERATION_GATE:
            // For 2-qubit gates, max of target and control
            {
                uint32_t max_qubit = op->op.gate.target;
                if (op->op.gate.control > max_qubit) max_qubit = op->op.gate.control;
                return (size_t)(max_qubit + 1);  // Qubits are 0-indexed
            }
        case OPERATION_MEASURE:
            return (size_t)(op->op.measure.qubit + 1);
        case OPERATION_RESET:
            return (size_t)(op->op.reset.qubit + 1);
        case OPERATION_BARRIER:
            return op->op.barrier.num_qubits > 0 ? op->op.barrier.num_qubits : 1;
        case OPERATION_ANNEAL:
            return 10;  // Default for annealing operations
        default:
            return 1;
    }
}

static void update_timing_metrics(InternalMetrics* metrics, const QuantumOperation* op) {
    if (!metrics || !op) return;

    // Measure operation timing
    double start = get_current_time_ms();

    // Estimate qubit count for complexity calculation
    size_t num_qubits = estimate_operation_qubits(op);
    double op_complexity = (double)num_qubits * (double)num_qubits;

    metrics->compute_time = op_complexity * 0.01;  // Base compute time
    metrics->memory_time = op_complexity * 0.002;  // Memory access time
    metrics->network_time = 0.0;  // No network for local ops

    double end = get_current_time_ms();
    metrics->total_runtime = end - start + metrics->compute_time + metrics->memory_time;
}

static void update_resource_metrics(InternalMetrics* metrics, const QuantumOperation* op) {
    if (!metrics || !op) return;

    // Estimate qubit count
    size_t num_qubits = estimate_operation_qubits(op);

    // Estimate memory usage: 2^n complex doubles for state vector
    // Use floating point for large qubit counts to avoid overflow
    double state_size = pow(2.0, (double)num_qubits) * sizeof(double) * 2;
    metrics->memory_usage = state_size;

    // Network usage (0 for local operations)
    metrics->network_usage = 0.0;

    // Estimate CPU usage based on operation complexity
    double normalized_qubits = (double)num_qubits / 20.0;
#ifdef _OPENMP
    metrics->cpu_usage = 0.5 + 0.5 * (normalized_qubits > 1.0 ? 1.0 : normalized_qubits);
#else
    metrics->cpu_usage = 0.3 + 0.3 * (normalized_qubits > 1.0 ? 1.0 : normalized_qubits);
#endif

    // GPU usage (0 if not using GPU)
    metrics->gpu_usage = 0.0;
}

static void update_perf_metrics(InternalMetrics* metrics, const QuantumOperation* op) {
    if (!metrics || !op) return;

    // Calculate throughput (operations per second)
    if (metrics->total_runtime > 0) {
        metrics->throughput = 1000.0 / metrics->total_runtime;
    }

    // Calculate latency (milliseconds per operation)
    metrics->latency = metrics->total_runtime;

    // Calculate efficiency (useful work / total work)
    double useful_work = metrics->compute_time;
    double total_work = metrics->total_runtime;
    metrics->efficiency = (total_work > 0) ? (useful_work / total_work) : 0.0;

    // Calculate overhead
    metrics->overhead = 1.0 - metrics->efficiency;
}

static void add_to_history(PerformanceState* state, const InternalMetrics* metrics) {
    if (!state || !metrics) return;

    // Copy metrics to history
    memcpy(&state->metric_history[state->current_index], metrics, sizeof(InternalMetrics));

    // Advance index
    state->current_index = (state->current_index + 1) % state->history_size;
}

static void analyze_performance(PerformanceMonitor* monitor, const InternalMetrics* metrics) {
    if (!monitor || !metrics || !monitor->state) return;

    SystemAnalysis* analysis = monitor->state->current_analysis;

    // Analyze resource utilization
    analyze_resource_utilization(analysis, metrics);

    // Detect bottlenecks
    detect_bottlenecks(analysis, metrics, monitor->state->baseline_analysis);

    // Identify optimization opportunities
    identify_optimizations(analysis, metrics, monitor->optimizer);

    // Update optimization state
    update_optimization_state(monitor->state, analysis);
}

static void analyze_resource_utilization(SystemAnalysis* analysis, const InternalMetrics* metrics) {
    if (!analysis || !metrics) return;

    // Calculate memory utilization
    double total_mem = get_total_memory();
    analysis->memory_utilization = (total_mem > 0) ? (metrics->memory_usage / total_mem) : 0.0;

    // Calculate network utilization
    double net_bw = get_network_bandwidth();
    analysis->network_utilization = (net_bw > 0) ? (metrics->network_usage / net_bw) : 0.0;

    // Calculate compute utilization
    double cpu = metrics->cpu_usage;
    double gpu = metrics->gpu_usage;
    analysis->compute_utilization = (cpu > gpu) ? cpu : gpu;
}

static void detect_bottlenecks(SystemAnalysis* analysis, const InternalMetrics* metrics, const SystemAnalysis* baseline) {
    if (!analysis || !metrics) return;

    // Check memory bottleneck
    analysis->memory_bound =
        (analysis->memory_utilization > 0.9) ||
        (metrics->memory_time > metrics->compute_time);

    // Check network bottleneck
    analysis->network_bound =
        (analysis->network_utilization > 0.9) ||
        (metrics->network_time > metrics->compute_time);

    // Check compute bottleneck
    analysis->compute_bound =
        (analysis->compute_utilization > 0.9) &&
        (!analysis->memory_bound) &&
        (!analysis->network_bound);

    // Calculate headroom
    if (baseline) {
        analysis->memory_headroom = baseline->memory_utilization - analysis->memory_utilization;
        analysis->network_headroom = baseline->network_utilization - analysis->network_utilization;
        analysis->compute_headroom = baseline->compute_utilization - analysis->compute_utilization;
    } else {
        analysis->memory_headroom = 1.0 - analysis->memory_utilization;
        analysis->network_headroom = 1.0 - analysis->network_utilization;
        analysis->compute_headroom = 1.0 - analysis->compute_utilization;
    }
}

static void identify_optimizations(SystemAnalysis* analysis, const InternalMetrics* metrics, OptimizationEngine* optimizer) {
    if (!analysis || !metrics || !optimizer) return;

    bool needs_optimization = false;

    if (metrics->efficiency < MIN_EFFICIENCY_RATIO) {
        needs_optimization = true;
    }

    if (metrics->overhead > MAX_OVERHEAD_RATIO) {
        needs_optimization = true;
    }

    if (metrics->latency > MAX_LATENCY_MS) {
        needs_optimization = true;
    }

    // Update optimization strategy
    if (needs_optimization) {
        if (analysis->memory_bound) {
            optimizer->strategy = OPTIMIZE_MEMORY;
        } else if (analysis->network_bound) {
            optimizer->strategy = OPTIMIZE_NETWORK;
        } else if (analysis->compute_bound) {
            optimizer->strategy = OPTIMIZE_COMPUTE;
        } else {
            optimizer->strategy = OPTIMIZE_ALL;
        }
        optimizer->optimization_count++;
    }
}

static void update_optimization_state(PerformanceState* state, const SystemAnalysis* analysis) {
    if (!state || !analysis) return;

    // Check if optimization needed
    state->needs_optimization =
        analysis->memory_bound ||
        analysis->network_bound ||
        analysis->compute_bound;

    // Update optimization threshold (minimum headroom)
    if (state->needs_optimization) {
        double min_headroom = analysis->memory_headroom;
        if (analysis->network_headroom < min_headroom) min_headroom = analysis->network_headroom;
        if (analysis->compute_headroom < min_headroom) min_headroom = analysis->compute_headroom;
        state->optimization_threshold = min_headroom;
    }

    // Update optimization strategy
    if (state->needs_optimization) {
        if (analysis->memory_bound) {
            state->strategy = OPTIMIZE_MEMORY;
        } else if (analysis->network_bound) {
            state->strategy = OPTIMIZE_NETWORK;
        } else if (analysis->compute_bound) {
            state->strategy = OPTIMIZE_COMPUTE;
        }
    } else {
        state->strategy = OPTIMIZE_NONE;
    }
}

// ============================================================================
// Legacy API compatibility (matches original function names)
// ============================================================================

PerformanceMonitor* init_performance_monitor_qpm(const MonitorConfig* config) {
    return qpm_init(config);
}

int update_performance_metrics_qpm(PerformanceMonitor* monitor, const QuantumOperation* op) {
    return qpm_update(monitor, op);
}

void cleanup_performance_monitor_qpm(PerformanceMonitor* monitor) {
    qpm_cleanup(monitor);
}
