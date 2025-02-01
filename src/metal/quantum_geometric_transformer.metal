#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and optimization
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant float SQRT_DIM_SCALE = 0.125f; // 1/sqrt(64) precomputed for efficiency
constant uint MAX_ATTENTION_HEADS = 16;
constant uint MAX_SEQUENCE_LENGTH = 512;

// Enhanced transformer metrics with comprehensive monitoring
struct TransformerMetrics {
    // Core counters
    device atomic_uint* numerical_errors;      // NaN/Inf errors
    device atomic_uint* stability_warnings;    // Numerical stability issues
    device atomic_uint* memory_transfers;      // Memory operation tracking
    device atomic_uint* computation_time;      // Performance tracking
    device atomic_uint* successful_ops;        // Successfully completed operations
    device atomic_uint* convergence_checks;    // Number of convergence checks
    device atomic_uint* cache_misses;         // Cache miss counter
    device atomic_uint* memory_stalls;        // Memory stall counter
    
    // Attention metrics
    float max_attention_score;         // Maximum attention score
    float min_attention_score;         // Minimum attention score
    float avg_attention_score;         // Average attention score
    float attention_entropy;           // Attention distribution entropy
    float attention_sparsity;          // Measure of attention sparsity
    float head_diversity;             // Diversity across attention heads
    float attention_stability;        // Stability of attention patterns
    float attention_coherence;        // Coherence of attention weights
    
    // Geometric metrics
    float max_geometric_norm;          // Maximum geometric transformation norm
    float avg_geometric_norm;          // Average geometric transformation norm
    float curvature_magnitude;         // Magnitude of geometric curvature
    float parallel_transport_error;    // Error in parallel transport
    float manifold_distortion;        // Distortion of geometric manifold
    float geometric_consistency;      // Consistency of geometric operations
    
    // Performance metrics
    float avg_magnitude;               // Average magnitude of values
    float gradient_norm;               // Norm of gradients
    float learning_efficiency;         // Learning efficiency metric
    float memory_efficiency;           // Memory usage efficiency
    float compute_utilization;        // Compute resource utilization
    float bandwidth_utilization;      // Memory bandwidth utilization
    float cache_efficiency;           // Cache hit rate
    float pipeline_efficiency;        // Pipeline utilization
    
    // Statistical metrics
    float value_distribution_mean;    // Mean of value distribution
    float value_distribution_var;     // Variance of value distribution
    float temporal_correlation;       // Temporal correlation metric
    float spatial_correlation;        // Spatial correlation metric
    
    // Resource metrics
    float peak_memory_usage;          // Peak memory consumption
    float avg_memory_usage;           // Average memory usage
    float compute_intensity;          // Compute to memory ratio
    float resource_balance;           // Resource utilization balance
    
    // Status flags
    bool has_convergence_issues;       // Convergence problem indicator
    bool has_gradient_instability;     // Gradient stability indicator
    bool has_memory_pressure;          // Memory pressure indicator
    bool requires_rescaling;           // Rescaling requirement flag
    bool has_attention_instability;   // Attention stability issue
    bool requires_reinitialization;   // Reinitialization needed
    bool has_resource_contention;     // Resource contention detected
    bool requires_optimization;       // Performance optimization needed
};

// Enhanced transformer configuration
struct TransformerConfig {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint sequence_length;
    float attention_dropout;
    float ffn_dropout;
    float head_dropout;
    float attention_scale;
    bool enable_geometric;             // Enable geometric transformations
    bool track_convergence;            // Enable convergence tracking
    bool adaptive_scaling;             // Enable adaptive scaling
    bool use_relative_position;       // Use relative position encoding
    bool enable_head_pruning;         // Enable attention head pruning
    bool use_sparse_attention;        // Enable sparse attention patterns
    float stability_threshold;         // Threshold for stability checks
    float pruning_threshold;          // Threshold for head pruning
    float sparsity_target;            // Target attention sparsity
    uint warmup_steps;                // Number of warmup steps
    uint max_cache_size;              // Maximum cache size
    float learning_rate;              // Current learning rate
};

// Enhanced transformer parameters
struct TransformerParams {
    device const float4x4* query;
    device const float4x4* key;
    device const float4x4* value;
    device float4x4* output;
    device const GeometricTensor* geometry;
    device TransformerMetrics* metrics;
    device float* workspace;           // Temporary workspace buffer
    device float* attention_cache;     // Cache for attention patterns
    device float* head_importance;     // Head importance scores
    device float* position_bias;       // Relative position biases
};

// Helper functions for numerical stability
inline bool is_valid_float(float x) {
    return !isnan(x) && !isinf(x) && abs(x) <= MAX_MAGNITUDE && abs(x) >= MIN_MAGNITUDE;
}

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
    device TransformerMetrics* metrics
) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    float sum_magnitude = 0.0f;
    float sum_squared = 0.0f;
    valid_elements = 0;
    
    // First pass: gather statistics with vectorization
    for (uint i = 0; i < 4; i++) {
        float4 row = m[i];
        float4 abs_row = abs(row);
        float4 valid_mask = select(float4(0.0f), float4(1.0f), abs_row > ERROR_THRESHOLD);
        
        max_magnitude = max(max_magnitude, max(max(abs_row.x, abs_row.y), max(abs_row.z, abs_row.w)));
        sum_magnitude += dot(abs_row, valid_mask);
        sum_squared += dot(abs_row * abs_row, valid_mask);
        valid_elements += uint(valid_mask.x + valid_mask.y + valid_mask.z + valid_mask.w);
    }
    
    // Compute statistics
    avg_magnitude = valid_elements > 0 ? sum_magnitude / float(valid_elements) : 0.0f;
    float variance = valid_elements > 0 ? 
        (sum_squared / float(valid_elements)) - (avg_magnitude * avg_magnitude) : 0.0f;
    
    // Update distribution metrics
    if (valid_elements > 0) {
        metrics->value_distribution_mean = avg_magnitude;
        metrics->value_distribution_var = variance;
    }
    
    // Second pass: normalize if needed with stability tracking
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            result[i] = m[i] * scale;
        }
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
        metrics->requires_rescaling = true;
    }
    
    return result;
}

// Enhanced attention computation with comprehensive monitoring
static float4x4 compute_attention(
    float4x4 Q,
    float4x4 K,
    float4x4 V,
    device const float* position_bias,
    float scale,
    float dropout_prob,
    uint head_index,
    device TransformerMetrics* metrics,
    device float* attention_cache,
    thread bool& valid_result
) {
    atomic_fetch_add_explicit(metrics->computation_time, 1, memory_order_relaxed);
    valid_result = true;
    
    // Initialize attention statistics
    float max_score = -INFINITY;
    float min_score = INFINITY;
    float sum_score = 0.0f;
    float sum_squared_score = 0.0f;
    uint valid_scores = 0;
    float attention_entropy = 0.0f;
    
    // Compute attention scores with enhanced stability and proper scaling
    float4x4 scores = matrix_multiply(Q, transpose(K)) * SQRT_DIM_SCALE * scale;
    
    // Apply relative position bias if available
    if (position_bias) {
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                scores[i][j] += position_bias[i * 4 + j];
            }
        }
    }
    
    // Track score statistics and compute softmax with stability
    float4x4 attention_weights;
    
    for (uint i = 0; i < 4; i++) {
        // First pass: find maximum for stability
        float row_max = -INFINITY;
        for (uint j = 0; j < 4; j++) {
            if (!isnan(scores[i][j]) && !isinf(scores[i][j])) {
                row_max = max(row_max, scores[i][j]);
            }
        }
        
        if (row_max == -INFINITY) {
            atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
            valid_result = false;
            return float4x4(0.0f);
        }
        
        // Second pass: compute stable softmax and track statistics
        float row_sum = 0.0f;
        float row_entropy = 0.0f;
        
        for (uint j = 0; j < 4; j++) {
            if (!isnan(scores[i][j]) && !isinf(scores[i][j])) {
                attention_weights[i][j] = exp(scores[i][j] - row_max);
                row_sum += attention_weights[i][j];
                
                // Track score statistics
                float score = scores[i][j];
                max_score = max(max_score, score);
                min_score = min(min_score, score);
                sum_score += score;
                sum_squared_score += score * score;
                valid_scores++;
            } else {
                attention_weights[i][j] = 0.0f;
            }
        }
        
        // Normalize weights and apply dropout
        if (row_sum > MIN_MAGNITUDE) {
            float inv_sum = 1.0f / row_sum;
            for (uint j = 0; j < 4; j++) {
                attention_weights[i][j] *= inv_sum;
                
                // Compute attention entropy
                if (attention_weights[i][j] > ERROR_THRESHOLD) {
                    float p = attention_weights[i][j];
                    row_entropy -= p * log2(p);
                }
                
                // Apply dropout with scaling
                if (dropout_prob > 0.0f) {
                    attention_weights[i][j] *= (1.0f / (1.0f - dropout_prob));
                }
            }
            attention_entropy += row_entropy;
        } else {
            // Fallback to uniform attention
            float uniform_weight = 0.25f;
            for (uint j = 0; j < 4; j++) {
                attention_weights[i][j] = uniform_weight;
            }
            attention_entropy += 2.0f;
            atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
        }
        
        // Cache attention pattern if cache is available
        if (attention_cache) {
            for (uint j = 0; j < 4; j++) {
                attention_cache[head_index * 16 + i * 4 + j] = attention_weights[i][j];
            }
        }
    }
    
    // Update attention metrics
    if (valid_scores > 0) {
        metrics->max_attention_score = max_score;
        metrics->min_attention_score = min_score;
        metrics->avg_attention_score = sum_score / float(valid_scores);
        metrics->attention_entropy = attention_entropy / 4.0f;
        
        // Compute attention sparsity
        float significant_weights = 0.0f;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                if (attention_weights[i][j] > 0.1f) {
                    significant_weights += 1.0f;
                }
            }
        }
        metrics->attention_sparsity = 1.0f - (significant_weights / 16.0f);
        
        // Update stability metrics
        float score_variance = (sum_squared_score / float(valid_scores)) - 
                             pow(metrics->avg_attention_score, 2.0f);
        metrics->attention_stability = 1.0f / (1.0f + sqrt(score_variance));
    }
    
    // Compute attention output with monitoring
    float4x4 output = matrix_multiply(attention_weights, V);
    atomic_fetch_add_explicit(metrics->computation_time, 1, memory_order_relaxed);
    
    // Final validation and normalization
    float max_magnitude;
    float avg_magnitude;
    uint valid_elements;
    output = normalize_matrix(output, max_magnitude, avg_magnitude, valid_elements, metrics);
    
    // Update success metrics
    if (valid_result) {
        atomic_fetch_add_explicit(metrics->successful_ops, 1, memory_order_relaxed);
    }
    
    return output;
}

// Enhanced geometric attention kernel with comprehensive monitoring
kernel void geometric_attention(
    constant TransformerConfig& config [[buffer(0)]],
    device const TransformerParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Validate configuration against constants
    if (tid >= config.batch_size || 
        config.num_heads > MAX_ATTENTION_HEADS ||
        config.sequence_length > MAX_SEQUENCE_LENGTH) {
        return;
    }
    
    // Initialize performance tracking
    uint start_time = atomic_load_explicit(params.metrics->computation_time, memory_order_relaxed);
    atomic_fetch_add_explicit(params.metrics->computation_time, 1, memory_order_relaxed);
    
    // Load input matrices with validation
    float4x4 Q = params.query[tid];
    float4x4 K = params.key[tid];
    float4x4 V = params.value[tid];
    atomic_fetch_add_explicit(params.metrics->memory_transfers, 3, memory_order_relaxed);
    
    if (!is_valid_float4x4(Q) || !is_valid_float4x4(K) || !is_valid_float4x4(V)) {
        atomic_fetch_add_explicit(params.metrics->numerical_errors, 1, memory_order_relaxed);
        params.metrics->has_convergence_issues = true;
        params.output[tid] = float4x4(0.0f);
        return;
    }
    
    // Apply geometric transformations with enhanced monitoring
    if (config.enable_geometric && params.geometry) {
        float max_magnitude;
        float avg_magnitude;
        uint valid_elements;
        bool has_geometric_warning = false;
        
        // Apply metric tensor
        if (is_valid_float4x4(params.geometry->metric)) {
            Q = normalize_matrix(
                matrix_multiply(Q, params.geometry->metric),
                max_magnitude, avg_magnitude, valid_elements,
                params.metrics
            );
            
            params.metrics->max_geometric_norm = max(params.metrics->max_geometric_norm, max_magnitude);
            params.metrics->avg_geometric_norm = (params.metrics->avg_geometric_norm + avg_magnitude) * 0.5f;
            
            if (max_magnitude > config.stability_threshold) {
                has_geometric_warning = true;
            }
            
            K = normalize_matrix(
                matrix_multiply(K, params.geometry->metric),
                max_magnitude, avg_magnitude, valid_elements,
                params.metrics
            );
        }
        
        // Apply connection coefficients
        if (is_valid_float4x4(params.geometry->connection)) {
            float4x4 transported = normalize_matrix(
                matrix_multiply(Q, params.geometry->connection),
                max_magnitude, avg_magnitude, valid_elements,
                params.metrics
            );
            
            // Track parallel transport error
            float transport_error = 0.0f;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    transport_error += abs(transported[i][j] - Q[i][j]);
                }
            }
            params.metrics->parallel_transport_error = transport_error / 16.0f;
            
            Q = transported;
        }
        
        // Apply curvature correction
        if (is_valid_float4x4(params.geometry->curvature)) {
            float curvature_magnitude = 0.0f;
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    curvature_magnitude += abs(params.geometry->curvature[i][j]);
                }
            }
            params.metrics->curvature_magnitude = curvature_magnitude / 16.0f;
            
            Q = normalize_matrix(
                matrix_multiply(Q, params.geometry->curvature),
                max_magnitude, avg_magnitude, valid_elements,
                params.metrics
            );
        }
        
        if (has_geometric_warning) {
            atomic_fetch_add_explicit(params.metrics->stability_warnings, 1, memory_order_relaxed);
            params.metrics->has_gradient_instability = true;
        }
    }
    
    // Compute attention with enhanced monitoring
    bool valid_attention;
    float4x4 attention_output = compute_attention(
        Q, K, V,
        params.position_bias,
        config.attention_scale,
        config.attention_dropout,
        tid % config.num_heads,
        params.metrics,
        params.attention_cache,
        valid_attention
    );
    
    if (!valid_attention) {
        params.output[tid] = float4x4(0.0f);
        return;
    }
    
    // Store result and update metrics
    params.output[tid] = attention_output;
    
    // Update performance metrics
    uint end_time = atomic_load_explicit(params.metrics->computation_time, memory_order_relaxed);
    float compute_time = float(end_time - start_time);
    
    params.metrics->compute_utilization = compute_time / float(config.batch_size);
    params.metrics->memory_efficiency = float(atomic_load_explicit(params.metrics->successful_ops, memory_order_relaxed)) /
                                      float(atomic_load_explicit(params.metrics->memory_transfers, memory_order_relaxed));
    
    // Update resource metrics
    params.metrics->peak_memory_usage = max(params.metrics->peak_memory_usage,
                                          float(atomic_load_explicit(params.metrics->memory_transfers, memory_order_relaxed)) * sizeof(float4x4));
    params.metrics->compute_intensity = compute_time / float(atomic_load_explicit(params.metrics->memory_transfers, memory_order_relaxed));
    
    // Check for resource pressure
    params.metrics->has_resource_contention = params.metrics->compute_utilization > 0.9f ||
                                            params.metrics->memory_efficiency < 0.5f;
    
    // Update optimization flags
    params.metrics->requires_optimization = params.metrics->has_resource_contention ||
                                          params.metrics->compute_intensity < 1.0f;
}
