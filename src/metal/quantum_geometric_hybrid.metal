#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Hybrid computation parameters
struct HybridParams {
    uint batch_size;
    uint hidden_size;
    float learning_rate;
    float quantum_weight;
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

// Kernel for hybrid quantum-classical computation with enhanced stability
kernel void hybrid_compute(
    device const float4x4* quantum_data [[buffer(0)]],
    device const float4x4* classical_data [[buffer(1)]],
    device float4x4* output [[buffer(2)]],
    constant HybridParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Load and validate quantum and classical data
    float4x4 quantum = quantum_data[tid];
    float4x4 classical = classical_data[tid];
    
    if (!is_valid_float4x4(quantum) || !is_valid_float4x4(classical)) {
        output[tid] = float4x4(0.0f);
        return;
    }
    
    // Normalize inputs for stability
    float max_magnitude;
    quantum = normalize_matrix(quantum, max_magnitude);
    classical = normalize_matrix(classical, max_magnitude);
    
    // Validate quantum weight
    float q_weight = clamp(params.quantum_weight, 0.0f, 1.0f);
    float c_weight = 1.0f - q_weight;
    
    // Hybrid computation with stability checks
    float4x4 result;
    float max_result = 0.0f;
    bool computation_valid = true;
    
    // First pass: compute hybrid values and validate
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = quantum[i][j] * q_weight + classical[i][j] * c_weight;
            if (isnan(val) || isinf(val) || abs(val) > MAX_MAGNITUDE) {
                computation_valid = false;
                break;
            }
            result[i][j] = val;
            max_result = max(max_result, abs(val));
        }
        if (!computation_valid) break;
    }
    
    // Store result with stability check
    if (!computation_valid || max_result > MAX_MAGNITUDE) {
        output[tid] = float4x4(0.0f); // Fallback to zero if computation fails
    } else {
        output[tid] = normalize_matrix(result, max_magnitude);
    }
}

// Kernel for hybrid resource scheduling with enhanced stability
kernel void hybrid_schedule(
    device const float4x4* workload [[buffer(0)]],
    device float* schedule [[buffer(1)]],
    constant HybridParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Load and validate workload
    float4x4 data = workload[tid];
    if (!is_valid_float4x4(data)) {
        schedule[tid] = 0.0f;
        return;
    }
    
    // Normalize workload for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    
    // Analyze workload complexity with stability checks
    float quantum_ops = 0.0f;
    float classical_ops = 0.0f;
    float total_ops = 0.0f;
    
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(data[i][j]);
            if (val > ERROR_THRESHOLD) {
                quantum_ops += (val > 0.5f) ? 1.0f : 0.0f;
                classical_ops += (val <= 0.5f) ? 1.0f : 0.0f;
                total_ops += 1.0f;
            }
        }
    }
    
    // Compute scheduling ratio with stability check
    if (total_ops < MIN_MAGNITUDE) {
        schedule[tid] = 0.0f;
    } else {
        float ratio = quantum_ops / total_ops;
        schedule[tid] = clamp(ratio, 0.0f, 1.0f);
    }
}

// Kernel for workload analysis with enhanced stability
kernel void analyze_workload(
    device const float4x4* workload [[buffer(0)]],
    device float* metrics [[buffer(1)]],
    constant HybridParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Load and validate workload
    float4x4 data = workload[tid];
    if (!is_valid_float4x4(data)) {
        metrics[tid] = 0.0f;
        return;
    }
    
    // Normalize workload for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    
    // Analyze workload patterns with stability checks
    float total_work = 0.0f;
    float max_work = 0.0f;
    
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(data[i][j]);
            if (val > ERROR_THRESHOLD) {
                total_work += 1.0f;
                max_work = max(max_work, val);
            }
        }
    }
    
    // Store metrics with stability check
    if (max_work > MAX_MAGNITUDE || total_work < MIN_MAGNITUDE) {
        metrics[tid] = 0.0f;
    } else {
        metrics[tid] = clamp(total_work / 16.0f, 0.0f, 1.0f);
    }
}

// Kernel for hybrid optimization with enhanced stability
kernel void hybrid_optimize(
    device float4x4* quantum_params [[buffer(0)]],
    device float4x4* classical_params [[buffer(1)]],
    constant HybridParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;
    
    // Load and validate parameters
    float4x4 q_params = quantum_params[tid];
    float4x4 c_params = classical_params[tid];
    
    if (!is_valid_float4x4(q_params) || !is_valid_float4x4(c_params)) {
        return;
    }
    
    // Normalize parameters for stability
    float max_magnitude;
    q_params = normalize_matrix(q_params, max_magnitude);
    c_params = normalize_matrix(c_params, max_magnitude);
    
    // Validate quantum weight
    float q_weight = clamp(params.quantum_weight, 0.0f, 1.0f);
    float c_weight = 1.0f - q_weight;
    
    // Apply hybrid optimization with stability checks
    float4x4 q_optimized, c_optimized;
    float max_q_opt = 0.0f, max_c_opt = 0.0f;
    bool optimization_valid = true;
    
    // First pass: compute optimized values and validate
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float q_val = q_params[i][j] * q_weight;
            float c_val = c_params[i][j] * c_weight;
            
            if (isnan(q_val) || isinf(q_val) || isnan(c_val) || isinf(c_val)) {
                optimization_valid = false;
                break;
            }
            
            q_optimized[i][j] = q_val;
            c_optimized[i][j] = c_val;
            
            max_q_opt = max(max_q_opt, abs(q_val));
            max_c_opt = max(max_c_opt, abs(c_val));
        }
        if (!optimization_valid) break;
    }
    
    // Store optimized parameters with stability checks
    if (optimization_valid) {
        if (max_q_opt <= MAX_MAGNITUDE) {
            quantum_params[tid] = normalize_matrix(q_optimized, max_magnitude);
        }
        if (max_c_opt <= MAX_MAGNITUDE) {
            classical_params[tid] = normalize_matrix(c_optimized, max_magnitude);
        }
    }
}
