#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;

// Pipeline configuration
struct PipelineConfig {
    uint batch_size;
    uint num_stages;
    bool enable_checkpointing;
    float dropout_rate;
};

// Pipeline stage parameters
struct StageParams {
    uint stage_id;
    uint input_dim;
    uint output_dim;
    bool is_training;
};

// Performance metrics for monitoring
struct PipelineMetrics {
    device atomic_uint* computation_time;
    device atomic_uint* memory_transfers;
    device atomic_uint* numerical_errors;
    device atomic_uint* stability_warnings;
    device atomic_uint* max_observed_magnitude;  // Will store float as uint bits
    device atomic_uint* min_observed_magnitude;  // Will store float as uint bits
    bool stage_completed;
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            if (isnan(m[i][j]) || isinf(m[i][j])) {
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

// Forward pass kernel with enhanced stability and monitoring
kernel void forward_pipeline_stage(
    device const float4x4* input [[buffer(0)]],
    device float4x4* output [[buffer(1)]],
    device const GeometricTensor* geometry [[buffer(2)]],
    constant PipelineConfig& config [[buffer(3)]],
    constant StageParams& params [[buffer(4)]],
    device PipelineMetrics* metrics [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Initialize local error tracking
    bool has_numerical_error = false;
    bool has_stability_warning = false;
    float local_max_magnitude = 0.0f;
    float local_min_magnitude = INFINITY;
    
    // Load and validate input data
    float4x4 data = input[tid];
    if (!is_valid_float4x4(data)) {
        output[tid] = float4x4(0.0f);
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
        return;
    }
    
    // Track initial magnitudes
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(data[i][j]);
            local_max_magnitude = max(local_max_magnitude, mag);
            local_min_magnitude = min(local_min_magnitude, mag);
        }
    }
    
    // Normalize input for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    if (max_magnitude > MAX_MAGNITUDE) {
        has_stability_warning = true;
    }
    
    // Apply geometric transformations with stability checks
    if (geometry) {
        atomic_fetch_add_explicit(metrics->memory_transfers, 1, memory_order_relaxed);
        
        // Project onto manifold with stability
        if (is_valid_float4x4(geometry->metric)) {
            data = normalize_matrix(
                matrix_multiply(data, geometry->metric),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
        
        // Apply parallel transport with stability
        if (is_valid_float4x4(geometry->connection)) {
            data = normalize_matrix(
                matrix_multiply(data, geometry->connection),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
        
        // Apply curvature correction with stability
        if (is_valid_float4x4(geometry->curvature)) {
            data = normalize_matrix(
                matrix_multiply(data, geometry->curvature),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
    }
    
    // Apply stage-specific processing with stability
    if (params.is_training && config.dropout_rate > 0.0f) {
        float scale = 1.0f / (1.0f - config.dropout_rate);
        float max_scaled = 0.0f;
        
        // First pass: compute maximum scaled value
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float scaled = data[i][j] * scale;
                max_scaled = max(max_scaled, abs(scaled));
            }
        }
        
        // Second pass: apply scaled dropout with stability
        if (max_scaled > MAX_MAGNITUDE) {
            scale *= (MAX_MAGNITUDE / max_scaled);
            has_stability_warning = true;
        }
        
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                data[i][j] *= scale;
            }
        }
    }
    
    // Final stability check and normalization
    float final_magnitude;
    data = normalize_matrix(data, final_magnitude);
    if (final_magnitude > MAX_MAGNITUDE) {
        has_stability_warning = true;
    }
    
    // Update metrics atomically
    if (has_numerical_error) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
    }
    if (has_stability_warning) {
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
    }
    
    // Update magnitude tracking atomically
    uint old_max = atomic_load_explicit(metrics->max_observed_magnitude, memory_order_relaxed);
    float old_max_float = as_type<float>(old_max);
    float new_max = max(local_max_magnitude, old_max_float);
    atomic_store_explicit(metrics->max_observed_magnitude, as_type<uint>(new_max), memory_order_relaxed);
    
    uint old_min = atomic_load_explicit(metrics->min_observed_magnitude, memory_order_relaxed);
    float old_min_float = as_type<float>(old_min);
    float new_min = min(local_min_magnitude, old_min_float);
    atomic_store_explicit(metrics->min_observed_magnitude, as_type<uint>(new_min), memory_order_relaxed);
    
    // Store output
    output[tid] = data;
}

// Backward pass kernel with enhanced stability and monitoring
kernel void backward_pipeline_stage(
    device const float4x4* gradient_in [[buffer(0)]],
    device float4x4* gradient_out [[buffer(1)]],
    device const GeometricTensor* geometry [[buffer(2)]],
    constant PipelineConfig& config [[buffer(3)]],
    constant StageParams& params [[buffer(4)]],
    device PipelineMetrics* metrics [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Initialize local error tracking
    bool has_numerical_error = false;
    bool has_stability_warning = false;
    float local_max_magnitude = 0.0f;
    float local_min_magnitude = INFINITY;
    
    // Load and validate gradient
    float4x4 grad = gradient_in[tid];
    if (!is_valid_float4x4(grad)) {
        gradient_out[tid] = float4x4(0.0f);
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
        return;
    }
    
    // Track initial magnitudes
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(grad[i][j]);
            local_max_magnitude = max(local_max_magnitude, mag);
            local_min_magnitude = min(local_min_magnitude, mag);
        }
    }
    
    // Normalize gradient for stability
    float max_magnitude;
    grad = normalize_matrix(grad, max_magnitude);
    if (max_magnitude > MAX_MAGNITUDE) {
        has_stability_warning = true;
    }
    
    // Apply geometric transformations in reverse with stability
    if (geometry) {
        atomic_fetch_add_explicit(metrics->memory_transfers, 1, memory_order_relaxed);
        
        // Reverse curvature correction with stability
        if (is_valid_float4x4(geometry->curvature)) {
            grad = normalize_matrix(
                matrix_multiply(grad, transpose(geometry->curvature)),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
        
        // Reverse parallel transport with stability
        if (is_valid_float4x4(geometry->connection)) {
            grad = normalize_matrix(
                matrix_multiply(grad, transpose(geometry->connection)),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
        
        // Project gradient back with stability
        if (is_valid_float4x4(geometry->metric)) {
            grad = normalize_matrix(
                matrix_multiply(grad, transpose(geometry->metric)),
                max_magnitude
            );
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
    }
    
    // Apply stage-specific gradient processing with stability
    if (params.is_training) {
        // Scale gradients based on stage position with stability
        float scale = 1.0f / float(params.stage_id + 1);
        float max_scaled = 0.0f;
        
        // First pass: compute maximum scaled gradient
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float scaled = grad[i][j] * scale;
                max_scaled = max(max_scaled, abs(scaled));
            }
        }
        
        // Second pass: apply scaled gradients with stability
        if (max_scaled > MAX_MAGNITUDE) {
            scale *= (MAX_MAGNITUDE / max_scaled);
            has_stability_warning = true;
        }
        
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                grad[i][j] *= scale;
            }
        }
    }
    
    // Final stability check and normalization
    float final_magnitude;
    grad = normalize_matrix(grad, final_magnitude);
    if (final_magnitude > MAX_MAGNITUDE) {
        has_stability_warning = true;
    }
    
    // Update metrics atomically
    if (has_numerical_error) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
    }
    if (has_stability_warning) {
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
    }
    
    // Update magnitude tracking atomically
    uint old_max = atomic_load_explicit(metrics->max_observed_magnitude, memory_order_relaxed);
    float old_max_float = as_type<float>(old_max);
    float new_max = max(local_max_magnitude, old_max_float);
    atomic_store_explicit(metrics->max_observed_magnitude, as_type<uint>(new_max), memory_order_relaxed);
    
    uint old_min = atomic_load_explicit(metrics->min_observed_magnitude, memory_order_relaxed);
    float old_min_float = as_type<float>(old_min);
    float new_min = min(local_min_magnitude, old_min_float);
    atomic_store_explicit(metrics->min_observed_magnitude, as_type<uint>(new_min), memory_order_relaxed);
    
    // Store output gradient
    gradient_out[tid] = grad;
}

// Pipeline synchronization kernel with enhanced stability and monitoring
kernel void synchronize_pipeline(
    device atomic_uint* stage_counters [[buffer(0)]],
    device float4x4* checkpoints [[buffer(1)]],
    device const float4x4* stage_outputs [[buffer(2)]],
    constant PipelineConfig& config [[buffer(3)]],
    constant StageParams& params [[buffer(4)]],
    device PipelineMetrics* metrics [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    // Initialize local error tracking
    bool has_numerical_error = false;
    bool has_stability_warning = false;
    float local_max_magnitude = 0.0f;
    float local_min_magnitude = INFINITY;
    
    // Atomic increment of stage counter with memory ordering
    atomic_fetch_add_explicit(
        stage_counters + params.stage_id,
        1,
        memory_order_relaxed
    );
    
    // Save checkpoint if enabled with stability checks
    if (config.enable_checkpointing) {
        atomic_fetch_add_explicit(metrics->memory_transfers, 1, memory_order_relaxed);
        
        float4x4 output = stage_outputs[tid];
        
        // Validate output before checkpointing
        if (!is_valid_float4x4(output)) {
            output = float4x4(0.0f);
            has_numerical_error = true;
        } else {
            // Track magnitudes
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    float mag = abs(output[i][j]);
                    local_max_magnitude = max(local_max_magnitude, mag);
                    local_min_magnitude = min(local_min_magnitude, mag);
                }
            }
            
            // Normalize checkpoint data for stability
            float max_magnitude;
            output = normalize_matrix(output, max_magnitude);
            if (max_magnitude > MAX_MAGNITUDE) {
                has_stability_warning = true;
            }
        }
        
        checkpoints[tid * config.num_stages + params.stage_id] = output;
    }
    
    // Update metrics atomically
    if (has_numerical_error) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
    }
    if (has_stability_warning) {
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
    }
    
    // Update magnitude tracking atomically
    uint old_max = atomic_load_explicit(metrics->max_observed_magnitude, memory_order_relaxed);
    float old_max_float = as_type<float>(old_max);
    float new_max = max(local_max_magnitude, old_max_float);
    atomic_store_explicit(metrics->max_observed_magnitude, as_type<uint>(new_max), memory_order_relaxed);
    
    uint old_min = atomic_load_explicit(metrics->min_observed_magnitude, memory_order_relaxed);
    float old_min_float = as_type<float>(old_min);
    float new_min = min(local_min_magnitude, old_min_float);
    atomic_store_explicit(metrics->min_observed_magnitude, as_type<uint>(new_min), memory_order_relaxed);
    
    // Mark stage as completed
    if (tid == 0) {
        metrics->stage_completed = true;
    }
}

// Pipeline error checking kernel with enhanced validation
kernel void validate_pipeline_stage(
    device const float4x4* stage_input [[buffer(0)]],
    device const float4x4* stage_output [[buffer(1)]],
    device atomic_uint* error_flags [[buffer(2)]],
    constant PipelineConfig& config [[buffer(3)]],
    constant StageParams& params [[buffer(4)]],
    device PipelineMetrics* metrics [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (tid >= config.batch_size) return;
    
    threadgroup atomic_uint local_error_flag;
    threadgroup float local_max_magnitude;
    threadgroup float local_min_magnitude;
    
    // Initialize shared memory
    if (lid == 0) {
        atomic_store_explicit(&local_error_flag, 0, memory_order_relaxed);
        local_max_magnitude = 0.0f;
        local_min_magnitude = INFINITY;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float4x4 input = stage_input[tid];
    float4x4 output = stage_output[tid];
    bool has_error = false;
    
    // Comprehensive error checking
    
    // 1. Check for invalid values
    if (!is_valid_float4x4(input) || !is_valid_float4x4(output)) {
        has_error = true;
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
    }
    
    // 2. Check dimension compatibility
    if (params.input_dim != params.output_dim) {
        has_error = true;
    }
    
    // 3. Check for numerical stability
    if (!has_error) {
        float max_input_mag = 0.0f;
        float max_output_mag = 0.0f;
        float min_input_mag = INFINITY;
        float min_output_mag = INFINITY;
        
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float in_mag = abs(input[i][j]);
                float out_mag = abs(output[i][j]);
                
                max_input_mag = max(max_input_mag, in_mag);
                max_output_mag = max(max_output_mag, out_mag);
                min_input_mag = min(min_input_mag, in_mag);
                min_output_mag = min(min_output_mag, out_mag);
            }
        }
        
        // Update local magnitude tracking
        local_max_magnitude = max(max_input_mag, max_output_mag);
        local_min_magnitude = min(min_input_mag, min_output_mag);
        
        // Check for numerical overflow/underflow
        if (max_input_mag > MAX_MAGNITUDE || max_output_mag > MAX_MAGNITUDE ||
            (max_input_mag > MIN_MAGNITUDE && max_output_mag < MIN_MAGNITUDE)) {
            has_error = true;
            atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
        }
    }
    
    // 4. Check for stage-specific constraints
    if (!has_error && params.is_training) {
        // Additional training-specific validations
        if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
            has_error = true;
        }
    }
    
    // Set local error flag if any issues found
    if (has_error) {
        atomic_fetch_or_explicit(&local_error_flag, 1, memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Only one thread per threadgroup updates the global state
    if (lid == 0) {
        if (atomic_load_explicit(&local_error_flag, memory_order_relaxed) != 0) {
            atomic_store_explicit(error_flags + params.stage_id, 1, memory_order_relaxed);
        }
        
        // Update global magnitude tracking
        uint old_max = atomic_load_explicit(metrics->max_observed_magnitude, memory_order_relaxed);
        float old_max_float = as_type<float>(old_max);
        float new_max = max(local_max_magnitude, old_max_float);
        atomic_store_explicit(metrics->max_observed_magnitude, as_type<uint>(new_max), memory_order_relaxed);
        
        uint old_min = atomic_load_explicit(metrics->min_observed_magnitude, memory_order_relaxed);
        float old_min_float = as_type<float>(old_min);
        float new_min = min(local_min_magnitude, old_min_float);
        atomic_store_explicit(metrics->min_observed_magnitude, as_type<uint>(new_min), memory_order_relaxed);
    }
}
