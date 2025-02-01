#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Production configuration parameters
struct ProductionConfig {
    uint batch_size;
    uint num_threads;
    float error_threshold;
    bool enable_optimizations;
};

// Production metrics
struct ProductionMetrics {
    float throughput;
    float latency;
    uint error_count;
    bool is_stable;
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            if (isnan(m[i][j]) || isinf(m[i][j]) || abs(m[i][j]) > MAX_MAGNITUDE) {
                return false;
            }
        }
    }
    return true;
}

inline float4x4 normalize_matrix(float4x4 m, thread float& max_magnitude) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    
    // First pass: find maximum magnitude
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(m[i][j]);
            max_magnitude = max(max_magnitude, mag);
        }
    }
    
    // Second pass: normalize if needed
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                result[i][j] *= scale;
            }
        }
    }
    
    return result;
}

// Kernel for production-ready tensor operations with enhanced stability
kernel void production_tensor_compute(
    device const GeometricTensor* input [[buffer(0)]],
    device GeometricTensor* output [[buffer(1)]],
    device ProductionMetrics* metrics [[buffer(2)]],
    constant ProductionConfig& config [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Load and validate input tensor
    GeometricTensor tensor = input[tid];
    if (!is_valid_float4x4(tensor.metric) || 
        !is_valid_float4x4(tensor.connection) || 
        !is_valid_float4x4(tensor.curvature)) {
        if (tid == 0) {
            metrics->error_count = UINT_MAX;
            metrics->is_stable = false;
            metrics->throughput = 0.0f;
            metrics->latency = INFINITY;
        }
        return;
    }
    
    // Initialize stability tracking
    float max_error = 0.0f;
    uint errors = 0;
    bool computation_valid = true;
    
    // Use ERROR_THRESHOLD as minimum allowed error threshold
    float effective_threshold = max(config.error_threshold, ERROR_THRESHOLD);
    
    // Normalize input tensors for stability
    float max_magnitude;
    tensor.metric = normalize_matrix(tensor.metric, max_magnitude);
    tensor.connection = normalize_matrix(tensor.connection, max_magnitude);
    tensor.curvature = normalize_matrix(tensor.curvature, max_magnitude);
    
    // Apply optimized computations with stability checks
    if (config.enable_optimizations) {
        // Optimize metric tensor computations with validation
        float4x4 metric_product = matrix_multiply(tensor.metric, tensor.metric);
        if (!is_valid_float4x4(metric_product)) {
            computation_valid = false;
        } else {
            tensor.metric = normalize_matrix(metric_product, max_magnitude);
        }
        
        if (computation_valid) {
            // Optimize connection computations with stability checks
            for (uint i = 0; i < 4 && computation_valid; i++) {
                for (uint j = 0; j < 4; j++) {
                    float val = tensor.connection[i][j];
                    if (isnan(val) || isinf(val)) {
                        tensor.connection[i][j] = 0.0f;
                        errors++;
                    } else {
                        float abs_val = abs(val);
                        max_error = max(max_error, abs_val);
                        if (abs_val > effective_threshold) {
                            tensor.connection[i][j] = clamp(val, 
                                -effective_threshold, 
                                effective_threshold);
                            errors++;
                        }
                    }
                }
            }
            
            // Optimize curvature computations with validation
            if (computation_valid) {
                float4x4 curvature_product = matrix_multiply(
                    tensor.connection, 
                    transpose(tensor.connection)
                );
                
                if (!is_valid_float4x4(curvature_product)) {
                    computation_valid = false;
                } else {
                    tensor.curvature = normalize_matrix(curvature_product, max_magnitude);
                }
            }
        }
    }
    
    // Store results with stability checks
    if (!computation_valid) {
        output[tid] = GeometricTensor(); // Zero-initialized tensor
        if (tid == 0) {
            metrics->error_count = UINT_MAX;
            metrics->is_stable = false;
            metrics->throughput = 0.0f;
            metrics->latency = INFINITY;
        }
    } else {
        output[tid] = tensor;
        if (tid == 0) {
            metrics->error_count = errors;
            metrics->is_stable = max_error <= effective_threshold;
            metrics->throughput = float(config.batch_size) / max(float(config.num_threads), MIN_MAGNITUDE);
            metrics->latency = float(config.num_threads) / max(float(config.batch_size), MIN_MAGNITUDE);
        }
    }
}

// Kernel for production performance monitoring with enhanced stability
kernel void monitor_production_performance(
    device const float4x4* workload [[buffer(0)]],
    device ProductionMetrics* metrics [[buffer(1)]],
    constant ProductionConfig& config [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Load and validate workload
    float4x4 data = workload[tid];
    if (!is_valid_float4x4(data)) {
        if (tid == 0) {
            metrics->error_count = UINT_MAX;
            metrics->is_stable = false;
            metrics->throughput = 0.0f;
            metrics->latency = INFINITY;
        }
        return;
    }
    
    // Initialize monitoring metrics
    float max_val = 0.0f;
    uint error_count = 0;
    
    // Use ERROR_THRESHOLD as minimum allowed error threshold
    float effective_threshold = max(config.error_threshold, ERROR_THRESHOLD);
    
    // Normalize workload for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    
    // Monitor numerical stability with enhanced validation
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = data[i][j];
            if (isnan(val) || isinf(val)) {
                error_count++;
                continue;
            }
            float abs_val = abs(val);
            max_val = max(max_val, abs_val);
            if (abs_val > effective_threshold) {
                error_count++;
            }
        }
    }
    
    // Update metrics with stability checks (thread 0 only)
    if (tid == 0) {
        metrics->error_count = error_count;
        metrics->is_stable = max_val <= effective_threshold;
        metrics->throughput = float(config.batch_size) / max(float(config.num_threads), MIN_MAGNITUDE);
        metrics->latency = float(config.num_threads) / max(float(config.batch_size), MIN_MAGNITUDE);
    }
}

// Kernel for production error handling with enhanced stability
kernel void handle_production_errors(
    device GeometricTensor* data [[buffer(0)]],
    device ProductionMetrics* metrics [[buffer(1)]],
    constant ProductionConfig& config [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Load and validate tensor
    GeometricTensor tensor = data[tid];
    uint errors = 0;
    bool needs_correction = false;
    float max_error = 0.0f;
    
    // Use ERROR_THRESHOLD as minimum allowed error threshold
    float effective_threshold = max(config.error_threshold, ERROR_THRESHOLD);
    
    // Initialize stability tracking
    bool tensor_valid = true;
    float max_magnitude;
    
    // Check and correct metric tensor with stability
    if (!is_valid_float4x4(tensor.metric)) {
        tensor.metric = float4x4(0.0f);
        errors += 16;
        needs_correction = true;
        tensor_valid = false;
    } else {
        tensor.metric = normalize_matrix(tensor.metric, max_magnitude);
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = tensor.metric[i][j];
                float abs_val = abs(val);
                max_error = max(max_error, abs_val);
                if (abs_val > effective_threshold) {
                    tensor.metric[i][j] = clamp(val, 
                        -effective_threshold, 
                        effective_threshold);
                    errors++;
                    needs_correction = true;
                }
            }
        }
    }
    
    // Check and correct connection coefficients with stability
    if (!is_valid_float4x4(tensor.connection)) {
        tensor.connection = float4x4(0.0f);
        errors += 16;
        needs_correction = true;
        tensor_valid = false;
    } else {
        tensor.connection = normalize_matrix(tensor.connection, max_magnitude);
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = tensor.connection[i][j];
                float abs_val = abs(val);
                max_error = max(max_error, abs_val);
                if (abs_val > effective_threshold) {
                    tensor.connection[i][j] = clamp(val, 
                        -effective_threshold, 
                        effective_threshold);
                    errors++;
                    needs_correction = true;
                }
            }
        }
    }
    
    // Check and correct curvature tensor with stability
    if (!is_valid_float4x4(tensor.curvature)) {
        tensor.curvature = float4x4(0.0f);
        errors += 16;
        needs_correction = true;
        tensor_valid = false;
    } else {
        tensor.curvature = normalize_matrix(tensor.curvature, max_magnitude);
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = tensor.curvature[i][j];
                float abs_val = abs(val);
                max_error = max(max_error, abs_val);
                if (abs_val > effective_threshold) {
                    tensor.curvature[i][j] = clamp(val, 
                        -effective_threshold, 
                        effective_threshold);
                    errors++;
                    needs_correction = true;
                }
            }
        }
    }
    
    // Store corrected tensor with stability checks
    if (needs_correction) {
        data[tid] = tensor;
    }
    
    // Update metrics with stability checks (thread 0 only)
    if (tid == 0) {
        metrics->error_count = errors;
        metrics->is_stable = tensor_valid && (max_error <= effective_threshold);
        metrics->throughput = tensor_valid ? 
            (float(config.batch_size) / max(float(config.num_threads), MIN_MAGNITUDE)) : 0.0f;
        metrics->latency = tensor_valid ? 
            (float(config.num_threads) / max(float(config.batch_size), MIN_MAGNITUDE)) : INFINITY;
    }
}
