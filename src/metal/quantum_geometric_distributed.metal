#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;  // Used for minimum magnitude validation
constant float ERROR_THRESHOLD = 1e-6f;  // Used for convergence checks in distributed computation

// Distributed computation parameters
struct DistributedParams {
    uint num_nodes;
    uint node_id;
    uint batch_size;
    uint hidden_size;
};

// Atomic counter for distributed synchronization
struct AtomicCounter {
    device atomic_uint* value;
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float abs_val = abs(m[i][j]);
            if (isnan(m[i][j]) || isinf(m[i][j]) || 
                abs_val > MAX_MAGNITUDE || (abs_val > 0.0f && abs_val < MIN_MAGNITUDE)) {
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
    } else if (max_magnitude > 0.0f && max_magnitude < MIN_MAGNITUDE) {
        float scale = MIN_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                result[i][j] *= scale;
            }
        }
    }
    
    return result;
}

// Kernel for distributed tensor operations with enhanced stability
kernel void distributed_tensor_compute(
    device const float4x4* input [[buffer(0)]],
    device float4x4* output [[buffer(1)]],
    device const GeometricTensor& geometry [[buffer(2)]],
    constant DistributedParams& params [[buffer(3)]],
    device AtomicCounter* counter [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Load and validate input data
    float4x4 data = input[tid];
    if (!is_valid_float4x4(data)) {
        output[tid] = float4x4(0.0f);
        return;
    }
    
    // Normalize input for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    
    // Apply geometric transformations with stability checks
    float4x4 transformed = data;
    float error = INFINITY;
    
    // Iterative geometric transformation with convergence check
    while (error > ERROR_THRESHOLD) {
        float4x4 prev = transformed;
        
        if (is_valid_float4x4(geometry.metric)) {
            float4x4 metric_result = matrix_multiply(transformed, geometry.metric);
            if (is_valid_float4x4(metric_result)) {
                transformed = normalize_matrix(metric_result, max_magnitude);
            }
        }
        
        if (is_valid_float4x4(geometry.connection)) {
            float4x4 connection_result = matrix_multiply(transformed, geometry.connection);
            if (is_valid_float4x4(connection_result)) {
                transformed = normalize_matrix(connection_result, max_magnitude);
            }
        }
        
        // Compute error for convergence check
        error = 0.0f;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                error = max(error, abs(transformed[i][j] - prev[i][j]));
            }
        }
    }
    
    // Store result with final stability check
    output[tid] = normalize_matrix(transformed, max_magnitude);
    
    // Synchronize using atomic counter with memory ordering
    atomic_fetch_add_explicit(counter->value, 1, memory_order_relaxed);
}

// Kernel for distributed gradient computation with enhanced stability
kernel void distributed_gradient_compute(
    device const float4x4* gradients [[buffer(0)]],
    device float4x4* accumulated [[buffer(1)]],
    device const GeometricTensor& geometry [[buffer(2)]],
    constant DistributedParams& params [[buffer(3)]],
    device AtomicCounter* counter [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Load and validate gradient data
    float4x4 grad = gradients[tid];
    if (!is_valid_float4x4(grad)) {
        accumulated[tid] = float4x4(0.0f);
        return;
    }
    
    // Normalize gradient for stability
    float max_magnitude;
    grad = normalize_matrix(grad, max_magnitude);
    
    // Apply geometric transformations with stability checks
    float4x4 transformed = grad;
    float error = INFINITY;
    
    // Iterative geometric transformation with convergence check
    while (error > ERROR_THRESHOLD) {
        float4x4 prev = transformed;
        
        if (is_valid_float4x4(geometry.metric)) {
            float4x4 metric_result = matrix_multiply(transformed, geometry.metric);
            if (is_valid_float4x4(metric_result)) {
                transformed = normalize_matrix(metric_result, max_magnitude);
            }
        }
        
        if (is_valid_float4x4(geometry.curvature)) {
            float4x4 curvature_result = matrix_multiply(transformed, geometry.curvature);
            if (is_valid_float4x4(curvature_result)) {
                transformed = normalize_matrix(curvature_result, max_magnitude);
            }
        }
        
        // Compute error for convergence check
        error = 0.0f;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                error = max(error, abs(transformed[i][j] - prev[i][j]));
            }
        }
    }
    
    // Store accumulated gradients with final stability check
    accumulated[tid] = normalize_matrix(transformed, max_magnitude);
    
    // Synchronize using atomic counter with memory ordering
    atomic_fetch_add_explicit(counter->value, 1, memory_order_relaxed);
}

// Kernel for distributed model update with enhanced stability
kernel void distributed_model_update(
    device float4x4* model [[buffer(0)]],
    device const float4x4* gradients [[buffer(1)]],
    constant float& learning_rate [[buffer(2)]],
    constant DistributedParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;
    
    // Load and validate model parameters and gradients
    float4x4 params_data = model[tid];
    float4x4 grad_data = gradients[tid];
    
    if (!is_valid_float4x4(params_data) || !is_valid_float4x4(grad_data)) {
        return;
    }
    
    // Normalize inputs for stability
    float max_magnitude;
    params_data = normalize_matrix(params_data, max_magnitude);
    grad_data = normalize_matrix(grad_data, max_magnitude);
    
    // Update model parameters with stability checks
    float4x4 updated;
    float max_update = 0.0f;
    
    // First pass: compute updates and find maximum
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float update = learning_rate * grad_data[i][j];
            updated[i][j] = params_data[i][j] - update;
            max_update = max(max_update, abs(update));
        }
    }
    
    // Apply update with stability check
    if (max_update > MAX_MAGNITUDE || (max_update > 0.0f && max_update < MIN_MAGNITUDE)) {
        float scale = (max_update > MAX_MAGNITUDE) ? 
                     MAX_MAGNITUDE / max_update : 
                     MIN_MAGNITUDE / max_update;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                updated[i][j] = params_data[i][j] - (learning_rate * grad_data[i][j] * scale);
            }
        }
    }
    
    // Store updated parameters with final stability check
    model[tid] = normalize_matrix(updated, max_magnitude);
}

// Kernel for distributed synchronization with enhanced stability
kernel void distributed_sync(
    device float4x4* local_model [[buffer(0)]],
    device const float4x4* global_model [[buffer(1)]],
    constant float& sync_rate [[buffer(2)]],
    constant DistributedParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;
    
    // Load and validate models
    float4x4 local = local_model[tid];
    float4x4 global = global_model[tid];
    
    if (!is_valid_float4x4(local) || !is_valid_float4x4(global)) {
        return;
    }
    
    // Normalize inputs for stability
    float max_magnitude;
    local = normalize_matrix(local, max_magnitude);
    global = normalize_matrix(global, max_magnitude);
    
    // Synchronize parameters with stability checks
    float4x4 synced;
    float max_sync = 0.0f;
    float sync_error = INFINITY;
    
    // Iterative synchronization with convergence check
    while (sync_error > ERROR_THRESHOLD) {
        float4x4 prev_synced = synced;
        
        // Compute synchronized values
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = local[i][j] * (1.0f - sync_rate) + global[i][j] * sync_rate;
                synced[i][j] = val;
                max_sync = max(max_sync, abs(val));
            }
        }
        
        // Compute error for convergence check
        if (max_sync > 0.0f) {
            sync_error = 0.0f;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    sync_error = max(sync_error, abs(synced[i][j] - prev_synced[i][j]) / max_sync);
                }
            }
        } else {
            sync_error = 0.0f;  // If all values are zero, we've converged
        }
    }
    
    // Store synchronized parameters with final stability check
    local_model[tid] = normalize_matrix(synced, max_magnitude);
}
