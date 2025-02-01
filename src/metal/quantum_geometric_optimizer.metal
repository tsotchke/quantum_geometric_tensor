#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and optimization
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint MAX_ITERATIONS = 1000;
constant uint WARMUP_ITERATIONS = 5;

// Enhanced optimizer configuration with comprehensive controls
struct OptimizerConfig {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float momentum;
    float weight_decay;
    float gradient_clip;
    bool track_performance;
    bool enable_adaptive_lr;
    bool use_momentum;
    bool use_nesterov;
    bool use_amsgrad;
    uint batch_size;
    uint warmup_steps;
    float min_lr;
    float max_lr;
};

// Enhanced optimizer state with comprehensive tracking
struct OptimizerState {
    float4x4 first_moment;
    float4x4 second_moment;
    float4x4 max_second_moment;  // For AMSGrad
    float4x4 velocity;          // For momentum
    uint step;
    float adaptive_lr;
    float momentum_factor;
    float effective_lr;
    float gradient_norm;
    float update_norm;
    bool warmup_completed;
};

// Enhanced optimizer metrics with detailed tracking
struct OptimizerMetrics {
    // Core counters
    device atomic_uint* numerical_errors;      // NaN/Inf errors
    device atomic_uint* stability_warnings;    // Numerical stability issues
    device atomic_uint* memory_transfers;      // Memory operation tracking
    device atomic_uint* computation_time;      // Performance tracking
    device atomic_uint* optimization_steps;    // Number of optimization steps
    device atomic_uint* successful_updates;    // Successfully applied updates
    device atomic_uint* warmup_steps;         // Warmup iteration counter
    device atomic_uint* convergence_checks;   // Convergence check counter
    
    // Performance metrics
    float max_gradient_norm;          // Maximum gradient magnitude
    float avg_gradient_norm;          // Average gradient magnitude
    float min_gradient_norm;          // Minimum gradient magnitude
    float gradient_variance;          // Gradient magnitude variance
    
    // Learning metrics
    float learning_efficiency;        // Learning rate efficiency
    float momentum_efficiency;        // Momentum utilization
    float adaptive_lr_scale;          // Adaptive learning rate scale
    float weight_decay_impact;        // Weight decay effectiveness
    
    // Resource metrics
    float compute_utilization;        // Compute resource usage
    float memory_efficiency;          // Memory utilization
    float bandwidth_utilization;      // Memory bandwidth usage
    float resource_efficiency;        // Overall resource efficiency
    
    // Statistical metrics
    float update_magnitude_mean;      // Mean parameter update size
    float update_magnitude_var;       // Update magnitude variance
    float learning_rate_mean;         // Mean effective learning rate
    float learning_rate_var;          // Learning rate variance
    
    // Status flags
    bool has_critical_errors;         // Critical error indicator
    bool requires_restart;            // Optimization restart flag
    bool reached_plateau;             // Learning plateau indicator
    bool has_resource_pressure;       // Resource pressure indicator
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(m[i][j]);
            if (isnan(m[i][j]) || isinf(m[i][j]) || 
                val > MAX_MAGNITUDE || val < MIN_MAGNITUDE) {
                return false;
            }
        }
    }
    return true;
}

// Enhanced matrix normalization with comprehensive monitoring
inline float4x4 normalize_matrix(
    float4x4 m,
    thread float& max_magnitude,
    thread float& avg_magnitude,
    thread uint& valid_elements,
    device OptimizerMetrics* metrics
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
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
    }
    
    return result;
}

// Enhanced Adam optimizer kernel with comprehensive monitoring
kernel void adam_optimize(
    device float4x4* params [[buffer(0)]],
    device const float4x4* gradients [[buffer(1)]],
    device OptimizerState* state [[buffer(2)]],
    constant OptimizerConfig& config [[buffer(3)]],
    device OptimizerMetrics* metrics [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= config.batch_size) return;
    
    atomic_fetch_add_explicit(metrics->computation_time, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics->optimization_steps, 1, memory_order_relaxed);
    
    // Initialize optimization tracking
    float max_gradient = 0.0f;
    float sum_gradient = 0.0f;
    float sum_squared_gradient = 0.0f;
    uint valid_updates = 0;
    bool optimization_successful = true;
    
    // Load and validate inputs with performance tracking
    float4x4 param = params[tid];
    float4x4 grad = gradients[tid];
    OptimizerState opt_state = state[tid];
    atomic_fetch_add_explicit(metrics->memory_transfers, 3, memory_order_relaxed);
    
    if (!is_valid_float4x4(param) || !is_valid_float4x4(grad)) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
        optimization_successful = false;
        return;
    }
    
    // Apply gradient clipping if enabled
    if (config.gradient_clip > 0.0f) {
        float grad_norm = 0.0f;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                grad_norm += grad[i][j] * grad[i][j];
            }
        }
        grad_norm = sqrt(grad_norm);
        
        if (grad_norm > config.gradient_clip) {
            float scale = config.gradient_clip / grad_norm;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    grad[i][j] *= scale;
                }
            }
            atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
        }
    }
    
    // Track gradient statistics
    float max_magnitude;
    float avg_magnitude;
    uint valid_elements;
    grad = normalize_matrix(grad, max_magnitude, avg_magnitude, valid_elements, metrics);
    
    // Update moments with enhanced monitoring
    float effective_lr = config.learning_rate;
    if (config.enable_adaptive_lr) {
        // Implement warmup schedule
        if (opt_state.step < config.warmup_steps) {
            effective_lr *= float(opt_state.step + 1) / float(config.warmup_steps);
            atomic_fetch_add_explicit(metrics->warmup_steps, 1, memory_order_relaxed);
        } else {
            effective_lr *= opt_state.adaptive_lr;
        }
        
        // Bound learning rate
        effective_lr = clamp(effective_lr, config.min_lr, config.max_lr);
    }
    
    // Update first moment (momentum)
    float beta1_t = config.beta1;
    if (config.use_momentum) {
        beta1_t *= opt_state.momentum_factor;
    }
    
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float grad_val = grad[i][j];
            
            // Apply weight decay
            if (config.weight_decay > 0.0f) {
                grad_val += config.weight_decay * param[i][j];
            }
            
            // Update first moment
            float m = beta1_t * opt_state.first_moment[i][j] + (1.0f - beta1_t) * grad_val;
            opt_state.first_moment[i][j] = m;
            
            // Update second moment
            float v = config.beta2 * opt_state.second_moment[i][j] + 
                     (1.0f - config.beta2) * grad_val * grad_val;
            opt_state.second_moment[i][j] = v;
            
            // Update maximum second moment for AMSGrad
            if (config.use_amsgrad) {
                opt_state.max_second_moment[i][j] = max(opt_state.max_second_moment[i][j], v);
            }
            
            // Track statistics
            float grad_mag = abs(grad_val);
            max_gradient = max(max_gradient, grad_mag);
            sum_gradient += grad_mag;
            sum_squared_gradient += grad_mag * grad_mag;
            valid_updates++;
        }
    }
    
    // Compute bias corrections
    float step = float(opt_state.step + 1);
    float correction1 = 1.0f / (1.0f - pow(config.beta1, step));
    float correction2 = 1.0f / (1.0f - pow(config.beta2, step));
    
    // Apply parameter updates with enhanced monitoring
    float max_update = 0.0f;
    float sum_update = 0.0f;
    float sum_squared_update = 0.0f;
    
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float m_hat = opt_state.first_moment[i][j] * correction1;
            float v_hat = config.use_amsgrad ? 
                opt_state.max_second_moment[i][j] * correction2 :
                opt_state.second_moment[i][j] * correction2;
            
            float update = effective_lr * m_hat / (sqrt(v_hat) + config.epsilon);
            
            // Apply Nesterov momentum if enabled
            if (config.use_nesterov && config.use_momentum) {
                update = update * (1.0f + opt_state.momentum_factor);
            }
            
            // Track update statistics
            float update_mag = abs(update);
            max_update = max(max_update, update_mag);
            sum_update += update_mag;
            sum_squared_update += update_mag * update_mag;
            
            // Apply update
            param[i][j] -= update;
        }
    }
    
    // Update optimizer state
    opt_state.step++;
    if (config.use_momentum) {
        opt_state.momentum_factor = config.momentum * opt_state.momentum_factor +
                                  (1.0f - config.momentum);
    }
    
    if (config.enable_adaptive_lr && opt_state.step >= WARMUP_ITERATIONS) {
        // Adjust adaptive learning rate based on gradient statistics
        float avg_update = valid_updates > 0 ? sum_update / float(valid_updates) : 0.0f;
        float update_variance = valid_updates > 0 ? 
            (sum_squared_update / float(valid_updates)) - (avg_update * avg_update) : 0.0f;
        
        // Scale learning rate based on update statistics
        if (update_variance > 0.0f) {
            opt_state.adaptive_lr *= (1.0f + avg_update / sqrt(update_variance));
        }
        opt_state.adaptive_lr = clamp(opt_state.adaptive_lr, 0.1f, 10.0f);
    }
    
    // Store final results
    params[tid] = param;
    state[tid] = opt_state;
    
    // Update optimization metrics
    if (tid == 0 && config.track_performance) {
        // Update gradient statistics
        metrics->max_gradient_norm = max_gradient;
        metrics->min_gradient_norm = min(metrics->min_gradient_norm, max_gradient);
        
        if (valid_updates > 0) {
            float avg_grad = sum_gradient / float(valid_updates);
            metrics->avg_gradient_norm = avg_grad;
            metrics->gradient_variance = (sum_squared_gradient / float(valid_updates)) - 
                                      (avg_grad * avg_grad);
        }
        
        // Update learning metrics
        metrics->learning_efficiency = float(atomic_load_explicit(metrics->successful_updates, memory_order_relaxed)) /
                                    float(atomic_load_explicit(metrics->optimization_steps, memory_order_relaxed));
        
        if (config.use_momentum) {
            metrics->momentum_efficiency = opt_state.momentum_factor;
        }
        
        if (config.enable_adaptive_lr) {
            metrics->adaptive_lr_scale = opt_state.adaptive_lr;
        }
        
        // Update resource metrics
        metrics->compute_utilization = float(atomic_load_explicit(metrics->computation_time, memory_order_relaxed)) /
                                     float(MAX_ITERATIONS);
        metrics->memory_efficiency = float(atomic_load_explicit(metrics->successful_updates, memory_order_relaxed)) /
                                   float(atomic_load_explicit(metrics->memory_transfers, memory_order_relaxed));
        metrics->bandwidth_utilization = float(atomic_load_explicit(metrics->memory_transfers, memory_order_relaxed)) *
                                       sizeof(float4x4) / metrics->compute_utilization;
        
        // Update statistical metrics
        if (valid_updates > 0) {
            float avg_update = sum_update / float(valid_updates);
            metrics->update_magnitude_mean = avg_update;
            metrics->update_magnitude_var = (sum_squared_update / float(valid_updates)) - 
                                         (avg_update * avg_update);
        }
        
        metrics->learning_rate_mean = effective_lr;
        
        // Update status flags
        metrics->has_critical_errors = !optimization_successful;
        metrics->requires_restart = metrics->gradient_variance > 100.0f * metrics->avg_gradient_norm;
        metrics->reached_plateau = metrics->gradient_variance < 0.01f * metrics->avg_gradient_norm;
        metrics->has_resource_pressure = metrics->bandwidth_utilization > 0.9f || 
                                       metrics->compute_utilization > 0.9f;
    }
    
    // Track successful completion
    if (optimization_successful) {
        atomic_fetch_add_explicit(metrics->successful_updates, 1, memory_order_relaxed);
    }
}
