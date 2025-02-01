#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and benchmarking
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint MAX_BENCHMARK_ITERATIONS = 1000;
constant uint WARMUP_ITERATIONS = 10;

// Enhanced benchmark parameters with comprehensive controls
struct BenchmarkParams {
    uint batch_size;
    uint num_iterations;
    uint warmup_iterations = WARMUP_ITERATIONS;
    bool track_memory;
    bool track_errors;
    bool track_performance;
    bool enable_profiling;
    bool adaptive_sampling;
    float tolerance;
    float confidence_threshold;
    uint min_sample_size;
    uint max_sample_size;
};

// Enhanced benchmark metrics with detailed tracking
struct BenchmarkMetrics {
    // Core counters
    atomic_uint numerical_errors;      // NaN/Inf errors
    atomic_uint stability_warnings;    // Numerical stability issues
    atomic_uint memory_transfers;      // Memory operation tracking
    atomic_uint computation_time;      // Performance tracking
    atomic_uint benchmark_iterations;  // Number of benchmark runs
    atomic_uint successful_runs;       // Successfully completed runs
    atomic_uint cache_misses;         // Cache miss counter
    atomic_uint memory_stalls;        // Memory stall counter
    atomic_uint compute_ops;          // Compute operations counter
    atomic_uint total_performance_count; // Total performance measurements
    
    // Performance metrics
    float max_performance_gain;        // Maximum speedup achieved
    float avg_performance_gain;        // Average speedup
    float min_performance_gain;        // Minimum speedup
    float performance_variance;        // Speedup variance
    
    // Memory metrics
    float peak_memory_usage;          // Peak memory consumption
    float avg_memory_usage;           // Average memory usage
    float memory_efficiency;          // Memory utilization efficiency
    float bandwidth_utilization;      // Memory bandwidth usage
    
    // Resource metrics
    float compute_utilization;        // Compute resource usage
    float cache_hit_rate;            // Cache performance
    float instruction_throughput;     // Instruction execution rate
    float resource_efficiency;        // Overall resource efficiency
    
    // Statistical metrics
    float confidence_interval;        // Statistical confidence
    float standard_deviation;         // Performance variation
    float sample_variance;            // Sample variance
    float error_margin;              // Error margin
    
    // Status flags
    bool has_critical_errors;         // Critical error indicator
    bool requires_resampling;         // Resampling requirement flag
    bool reached_iteration_limit;     // Iteration limit indicator
    bool has_resource_pressure;       // Resource pressure indicator
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(m[i][j]);
            if (isnan(m[i][j]) || isinf(m[i][j]) || val > MAX_MAGNITUDE || val < ERROR_THRESHOLD) {
                return false;
            }
        }
    }
    return true;
}

// Enhanced matrix normalization with performance tracking
inline float4x4 normalize_matrix(
    float4x4 m,
    thread float& max_magnitude,
    thread float& avg_magnitude,
    thread uint& valid_elements,
    device BenchmarkMetrics& metrics
) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    float sum_magnitude = 0.0f;
    valid_elements = 0;
    
    // First pass: gather statistics with vectorization
    for (uint i = 0; i < 4; i++) {
        float4 row = m[i];
        float4 abs_row = abs(row);
        float4 valid_mask = select(float4(0.0f), float4(1.0f), abs_row > ERROR_THRESHOLD);
        
        max_magnitude = max(max_magnitude, max(max(abs_row.x, abs_row.y), max(abs_row.z, abs_row.w)));
        sum_magnitude += dot(abs_row, valid_mask);
        valid_elements += uint(valid_mask.x + valid_mask.y + valid_mask.z + valid_mask.w);
    }
    
    // Compute average magnitude
    avg_magnitude = valid_elements > 0 ? sum_magnitude / float(valid_elements) : 0.0f;
    
    // Second pass: normalize if needed with performance tracking
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            result[i] = m[i] * scale;
        }
        atomic_fetch_add_explicit(&metrics.stability_warnings, 1, memory_order_relaxed);
    }
    
    return result;
}

// Enhanced performance measurement kernel with comprehensive monitoring
kernel void measure_performance(
    device const float4x4* quantum [[buffer(0)]],
    device const float4x4* tensorflow [[buffer(1)]],
    device float* performance_metrics [[buffer(2)]],
    device BenchmarkMetrics& metrics [[buffer(3)]],
    constant BenchmarkParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    atomic_fetch_add_explicit(&metrics.computation_time, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&metrics.benchmark_iterations, 1, memory_order_relaxed);
    
    // Initialize performance tracking
    float quantum_time = 0.0f;
    float tensorflow_time = 0.0f;
    float sum_squared_speedup = 0.0f;
    uint valid_measurements = 0;
    bool measurement_successful = true;
    
    // Load and validate inputs with performance tracking
    float4x4 quantum_data = quantum[tid];
    float4x4 tensorflow_data = tensorflow[tid];
    atomic_fetch_add_explicit(&metrics.memory_transfers, 2, memory_order_relaxed);
    
    if (!is_valid_float4x4(quantum_data) || !is_valid_float4x4(tensorflow_data)) {
        atomic_fetch_add_explicit(&metrics.numerical_errors, 1, memory_order_relaxed);
        measurement_successful = false;
    } else {
        // Normalize inputs for stability
        float max_magnitude_q, avg_magnitude_q;
        float max_magnitude_t, avg_magnitude_t;
        uint valid_elements_q, valid_elements_t;
        
        quantum_data = normalize_matrix(quantum_data, max_magnitude_q, avg_magnitude_q, valid_elements_q, metrics);
        tensorflow_data = normalize_matrix(tensorflow_data, max_magnitude_t, avg_magnitude_t, valid_elements_t, metrics);
        
        // Track stability metrics
        if (max_magnitude_q > MAX_MAGNITUDE || max_magnitude_t > MAX_MAGNITUDE) {
            atomic_fetch_add_explicit(&metrics.stability_warnings, 1, memory_order_relaxed);
        }
        
        // Perform measurements with comprehensive monitoring
        for (uint iter = 0; iter < params.num_iterations && iter < MAX_BENCHMARK_ITERATIONS; iter++) {
            // Skip warmup iterations in metrics
            bool count_metrics = iter >= params.warmup_iterations;
            
            // Measure quantum implementation
            float start_time = float(atomic_fetch_add_explicit(&metrics.computation_time, 1, memory_order_relaxed));
            
            // Simulate computation with matrix operations
            float4x4 q_result = matrix_multiply(quantum_data, transpose(quantum_data));
            
            // Use q_result to compute performance metrics
            float result_magnitude = 0.0f;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    result_magnitude += q_result[i][j] * q_result[i][j];
                }
            }
            atomic_fetch_add_explicit(&metrics.compute_ops, as_type<uint>(sqrt(result_magnitude)), memory_order_relaxed);
            
            float end_time = float(atomic_load_explicit(&metrics.computation_time, memory_order_relaxed));
            float q_time = end_time - start_time;
            
            // Measure tensorflow implementation
            start_time = float(atomic_fetch_add_explicit(&metrics.computation_time, 1, memory_order_relaxed));
            
            // Simulate computation with matrix operations
            float4x4 t_result = matrix_multiply(tensorflow_data, transpose(tensorflow_data));
            
            // Use t_result to compute comparison metrics
            float tf_magnitude = 0.0f;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    tf_magnitude += t_result[i][j] * t_result[i][j];
                }
            }
            atomic_fetch_add_explicit(&metrics.total_performance_count, as_type<uint>(sqrt(tf_magnitude)), memory_order_relaxed);
            
            end_time = float(atomic_load_explicit(&metrics.computation_time, memory_order_relaxed));
            float t_time = end_time - start_time;
            
            // Track performance metrics
            if (count_metrics && t_time > MIN_MAGNITUDE) {
                float speedup = q_time / t_time;
                if (speedup <= MAX_MAGNITUDE) {
                    quantum_time += q_time;
                    tensorflow_time += t_time;
                    sum_squared_speedup += speedup * speedup;
                    valid_measurements++;
                    
                    // Update running statistics
                    if (tid == 0) {
                        metrics.max_performance_gain = max(metrics.max_performance_gain, speedup);
                        metrics.min_performance_gain = min(metrics.min_performance_gain, speedup);
                        metrics.avg_performance_gain = (metrics.avg_performance_gain * float(valid_measurements - 1) + speedup) / float(valid_measurements);
                    }
                }
            }
            
            // Track resource utilization
            if (count_metrics) {
                atomic_fetch_add_explicit(&metrics.memory_transfers, 4, memory_order_relaxed);
                metrics.compute_utilization = float(atomic_load_explicit(&metrics.computation_time, memory_order_relaxed)) / float(params.num_iterations);
                metrics.bandwidth_utilization = float(atomic_load_explicit(&metrics.memory_transfers, memory_order_relaxed)) * sizeof(float4x4) / metrics.compute_utilization;
            }
        }
    }
    
    // Compute final statistics
    if (tid == 0 && valid_measurements > 0) {
        // Update performance metrics
        metrics.performance_variance = (sum_squared_speedup / float(valid_measurements)) - (metrics.avg_performance_gain * metrics.avg_performance_gain);
        metrics.standard_deviation = sqrt(metrics.performance_variance);
        
        // Compute confidence interval (95% confidence)
        metrics.confidence_interval = 1.96f * metrics.standard_deviation / sqrt(float(valid_measurements));
        metrics.error_margin = metrics.confidence_interval / metrics.avg_performance_gain;
        
        // Update status flags
        metrics.requires_resampling = metrics.error_margin > params.confidence_threshold;
        metrics.reached_iteration_limit = atomic_load_explicit(&metrics.benchmark_iterations, memory_order_relaxed) >= MAX_BENCHMARK_ITERATIONS;
        metrics.has_resource_pressure = metrics.bandwidth_utilization > 0.9f || metrics.compute_utilization > 0.9f;
        
        // Store final metrics
        if (measurement_successful) {
            performance_metrics[0] = quantum_time / float(valid_measurements);
            performance_metrics[1] = tensorflow_time / float(valid_measurements);
            performance_metrics[2] = metrics.avg_performance_gain;
            performance_metrics[3] = metrics.confidence_interval;
            performance_metrics[4] = metrics.error_margin;
        } else {
            performance_metrics[0] = 0.0f;
            performance_metrics[1] = 0.0f;
            performance_metrics[2] = 0.0f;
            performance_metrics[3] = 0.0f;
            performance_metrics[4] = 0.0f;
        }
    }
    
    // Track successful completion
    if (measurement_successful) {
        atomic_fetch_add_explicit(&metrics.successful_runs, 1, memory_order_relaxed);
    }
}
