#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and error detection
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint MAX_ERROR_PATTERNS = 16;
constant uint MAX_PATTERN_HISTORY = 32;
constant uint WARP_SIZE = 32;

// Enhanced error detection parameters with comprehensive controls
struct ErrorParams {
    uint batch_size;
    uint error_threshold;
    uint min_pattern_size;
    uint max_pattern_size;
    uint history_window = MAX_PATTERN_HISTORY;
    uint thread_group_size = WARP_SIZE;
    float tolerance;
    float confidence_threshold;
    float correlation_threshold;
    float stability_threshold;
    bool validate_results;
    bool track_patterns;
    bool enable_correction;
    bool use_adaptive_threshold;
    bool track_history;
    bool enable_prediction;
    bool use_pattern_matching;
    float min_confidence;
    float max_error_rate;
    uint warmup_steps;
    uint max_iterations;
};

// Enhanced error metrics with comprehensive monitoring
struct ErrorMetrics {
    // Core error counters
    device atomic_uint* numerical_errors;      // NaN/Inf errors
    device atomic_uint* magnitude_errors;      // Value range errors
    device atomic_uint* stability_warnings;    // Numerical stability issues
    device atomic_uint* correction_attempts;   // Number of correction attempts
    device atomic_uint* successful_corrections; // Successfully corrected errors
    device atomic_uint* computation_time;      // Performance tracking
    device atomic_uint* pattern_detections;    // Number of pattern detections
    device atomic_uint* false_positives;       // False error detections
    device atomic_uint* memory_operations;    // Memory operation counter
    device atomic_uint* pattern_matches;      // Pattern match counter
    
    // Error statistics
    float max_observed_error;          // Maximum error magnitude
    float min_observed_error;          // Minimum error magnitude
    float avg_error_magnitude;         // Average error magnitude
    float error_variance;              // Error magnitude variance
    float error_distribution;         // Error distribution metric
    float error_correlation;          // Error correlation metric
    float error_persistence;          // Error persistence metric
    float error_propagation;          // Error propagation rate
    
    // Pattern analysis
    float pattern_confidence;          // Pattern detection confidence
    float pattern_frequency;           // Error pattern frequency
    float pattern_correlation;         // Error pattern correlation
    float pattern_stability;           // Pattern stability metric
    float pattern_evolution;          // Pattern evolution rate
    float pattern_complexity;         // Pattern complexity metric
    float pattern_predictability;     // Pattern predictability
    float pattern_significance;       // Pattern significance score
    
    // Performance metrics
    float detection_accuracy;          // Error detection accuracy
    float correction_efficiency;       // Error correction efficiency
    float recovery_rate;              // Error recovery success rate
    float response_time;              // Error response latency
    float detection_latency;          // Error detection delay
    float correction_latency;         // Error correction delay
    float prediction_accuracy;        // Error prediction accuracy
    float adaptation_rate;            // System adaptation rate
    
    // Resource metrics
    float memory_overhead;            // Memory usage for error handling
    float compute_overhead;           // Computational overhead
    float bandwidth_utilization;      // Memory bandwidth usage
    float resource_efficiency;        // Resource utilization efficiency
    float cache_efficiency;          // Cache utilization
    float thread_efficiency;         // Thread utilization
    float memory_coherence;          // Memory coherence metric
    float resource_balance;          // Resource balance metric
    
    // Status flags
    bool has_critical_errors;         // Critical error indicator
    bool requires_intervention;       // Manual intervention needed
    bool pattern_detected;            // Error pattern detected
    bool recovery_possible;           // Recovery possibility flag
    bool needs_optimization;         // Optimization needed flag
    bool requires_recalibration;     // Recalibration needed flag
    bool has_resource_pressure;      // Resource pressure indicator
    bool requires_reinitialization;  // Reinitialization needed flag
};

// Enhanced error pattern structure with comprehensive tracking
struct ErrorPattern {
    float4x4 signature;              // Error signature matrix
    float4x4 correlation_matrix;     // Pattern correlation matrix
    float4x4 evolution_history[MAX_PATTERN_HISTORY];  // Pattern evolution history
    float confidence;                // Pattern confidence score
    float stability;                // Pattern stability score
    float persistence;              // Pattern persistence score
    float predictability;           // Pattern predictability score
    uint frequency;                  // Pattern occurrence count
    float correlation;              // Pattern correlation score
    float evolution_rate;           // Pattern evolution rate
    float complexity;               // Pattern complexity score
    bool is_correctable;            // Correction possibility flag
    bool is_stable;                // Pattern stability flag
    bool requires_attention;        // Pattern attention flag
    bool has_evolved;              // Pattern evolution flag
    uint history_index;            // Current history index
    float history_confidence[MAX_PATTERN_HISTORY];  // Historical confidence scores
};

// Pattern statistics tracking structure
struct PatternStats {
    float correlation;
    float stability;
    float evolution_rate;
    bool has_evolved;
    uint matches;
    float history_correlation[MAX_PATTERN_HISTORY];
    uint history_index;
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
    device ErrorMetrics* metrics
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
    
    // Second pass: normalize if needed with stability tracking
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            result[i] = m[i] * scale;
        }
        atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
    }
    
    // Update metrics
    metrics->error_variance = variance;
    
    return result;
}

// Enhanced error pattern detection with comprehensive analysis
inline bool detect_error_pattern(
    float4x4 data,
    device const ErrorPattern* patterns,
    thread float& confidence,
    device ErrorMetrics* metrics,
    constant ErrorParams& params
) {
    float max_correlation = 0.0f;
    float avg_correlation = 0.0f;
    uint best_pattern = 0;
    bool pattern_found = false;
    uint valid_correlations = 0;
    
    // Local pattern statistics
    PatternStats pattern_stats[MAX_ERROR_PATTERNS];
    
    // Compare with known patterns
    for (uint p = 0; p < MAX_ERROR_PATTERNS; p++) {
        pattern_stats[p].correlation = 0.0f;
        pattern_stats[p].stability = 0.0f;
        pattern_stats[p].matches = 0;
        pattern_stats[p].has_evolved = false;
        pattern_stats[p].evolution_rate = 0.0f;
        pattern_stats[p].history_index = 0;
        
        // Initialize history correlation
        for (uint h = 0; h < MAX_PATTERN_HISTORY; h++) {
            pattern_stats[p].history_correlation[h] = 0.0f;
        }
        
        // Compute pattern correlation and stability
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float diff = abs(data[i][j] - patterns[p].signature[i][j]);
                if (diff < params.tolerance) {
                    float match_strength = 1.0f - (diff / params.tolerance);
                    pattern_stats[p].correlation += match_strength;
                    pattern_stats[p].stability += 1.0f - abs(patterns[p].evolution_rate);
                    pattern_stats[p].matches++;
                    
                    // Track historical correlation
                    uint hist_idx = (patterns[p].history_index + 0) % MAX_PATTERN_HISTORY;
                    pattern_stats[p].history_correlation[hist_idx] += match_strength;
                }
            }
        }
        
        // Normalize metrics
        if (pattern_stats[p].matches > 0) {
            pattern_stats[p].correlation /= float(pattern_stats[p].matches);
            pattern_stats[p].stability /= float(pattern_stats[p].matches);
            
            // Normalize historical correlation
            for (uint h = 0; h < MAX_PATTERN_HISTORY; h++) {
                pattern_stats[p].history_correlation[h] /= float(pattern_stats[p].matches);
            }
            
            // Track evolution using historical data
            float avg_hist_correlation = 0.0f;
            for (uint hist_idx = 0; hist_idx < params.history_window; hist_idx++) {
                avg_hist_correlation += pattern_stats[p].history_correlation[hist_idx];
            }
            avg_hist_correlation /= float(params.history_window);
            
            if (pattern_stats[p].correlation > avg_hist_correlation) {
                pattern_stats[p].has_evolved = true;
                pattern_stats[p].evolution_rate = (pattern_stats[p].correlation - avg_hist_correlation) / pattern_stats[p].correlation;
            }
            
            // Update correlation statistics
            max_correlation = max(max_correlation, pattern_stats[p].correlation);
            avg_correlation += pattern_stats[p].correlation;
            valid_correlations++;
            
            if (pattern_stats[p].correlation > params.correlation_threshold) {
                best_pattern = p;
                pattern_found = true;
            }
        }
    }
    
    // Update pattern metrics
    if (valid_correlations > 0) {
        avg_correlation /= float(valid_correlations);
        metrics->pattern_correlation = avg_correlation;
        metrics->pattern_stability = pattern_stats[best_pattern].stability;
        metrics->pattern_evolution = pattern_stats[best_pattern].evolution_rate;
        metrics->pattern_predictability = pattern_found ? 
            patterns[best_pattern].predictability : 0.0f;
    }
    
    confidence = max_correlation;
    return pattern_found && confidence >= params.confidence_threshold;
}

// Enhanced kernel for error detection with comprehensive monitoring
kernel void detect_errors(
    device const float4x4* data [[buffer(0)]],
    device const ErrorPattern* patterns [[buffer(1)]],
    device ErrorMetrics* metrics [[buffer(2)]],
    constant ErrorParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Initialize performance tracking
    uint start_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics->computation_time, 1, memory_order_relaxed);
    
    // Load and validate data with error tracking
    float4x4 tensor = data[tid];
    atomic_fetch_add_explicit(metrics->memory_operations, 1, memory_order_relaxed);
    
    bool has_numerical_error = false;
    bool has_magnitude_error = false;
    bool has_pattern = false;
    bool requires_correction = false;
    
    // Track error statistics
    float max_error = 0.0f;
    float min_error = INFINITY;
    float sum_error = 0.0f;
    float sum_squared_error = 0.0f;
    uint valid_elements = 0;
    
    // Normalize input with error tracking
    float max_magnitude;
    float avg_magnitude;
    uint norm_valid_elements;
    tensor = normalize_matrix(tensor, max_magnitude, avg_magnitude, norm_valid_elements, metrics);
    
    // Detailed error analysis with pattern detection
    if (params.track_patterns) {
        float pattern_confidence;
        has_pattern = detect_error_pattern(tensor, patterns, pattern_confidence, metrics, params);
        
        if (has_pattern) {
            atomic_fetch_add_explicit(metrics->pattern_detections, 1, memory_order_relaxed);
            metrics->pattern_confidence = max(metrics->pattern_confidence, pattern_confidence);
            requires_correction = pattern_confidence >= params.confidence_threshold;
        }
    }
    
    // Element-wise error analysis with comprehensive tracking
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = tensor[i][j];
            
            // Check for numerical errors
            if (isnan(val) || isinf(val)) {
                has_numerical_error = true;
                continue;
            }
            
            // Track error magnitudes
            float error_mag = abs(val);
            max_error = max(max_error, error_mag);
            min_error = min(min_error, error_mag);
            sum_error += error_mag;
            sum_squared_error += error_mag * error_mag;
            valid_elements++;
            
            // Check magnitude errors with adaptive threshold
            float threshold = params.use_adaptive_threshold ? 
                params.tolerance * (1.0f + log2(1.0f + error_mag)) : 
                params.tolerance;
            
            if (error_mag > threshold) {
                has_magnitude_error = true;
                requires_correction |= error_mag > params.max_error_rate;
            }
        }
    }
    
    // Update error metrics atomically
    if (has_numerical_error) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
    }
    if (has_magnitude_error) {
        atomic_fetch_add_explicit(metrics->magnitude_errors, 1, memory_order_relaxed);
    }
    
    // Update error statistics (thread 0 only)
    if (tid == 0 && valid_elements > 0) {
        // Update error magnitude metrics
        metrics->max_observed_error = max_error;
        metrics->min_observed_error = min_error;
        metrics->avg_error_magnitude = sum_error / float(valid_elements);
        
        // Compute error distribution metrics
        float mean = sum_error / float(valid_elements);
        metrics->error_variance = (sum_squared_error / float(valid_elements)) - (mean * mean);
        metrics->error_distribution = metrics->error_variance / (mean * mean);
        
        // Update detection metrics
        metrics->detection_accuracy = float(atomic_load_explicit(metrics->pattern_detections, memory_order_relaxed)) /
                                    float(atomic_load_explicit(metrics->computation_time, memory_order_relaxed));
        
        // Update performance metrics
        uint end_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
        metrics->detection_latency = float(end_time - start_time);
        
        // Update resource metrics
        metrics->memory_overhead = float(atomic_load_explicit(metrics->memory_operations, memory_order_relaxed)) * 
                                 sizeof(float4x4) / metrics->compute_overhead;
        
        // Set status flags
        metrics->has_critical_errors = has_numerical_error || 
                                     (has_magnitude_error && max_error > MAX_MAGNITUDE);
        metrics->pattern_detected = has_pattern;
        metrics->requires_intervention = metrics->has_critical_errors && 
                                       metrics->detection_accuracy < params.min_confidence;
        metrics->recovery_possible = !metrics->has_critical_errors && 
                                   (metrics->pattern_detected || max_error <= MAX_MAGNITUDE);
        metrics->needs_optimization = metrics->memory_overhead > 0.9f || 
                                    metrics->detection_latency > params.tolerance;
    }
}

// Enhanced kernel for error correction with comprehensive monitoring
kernel void correct_errors(
    device float4x4* data [[buffer(0)]],
    device const ErrorPattern* patterns [[buffer(1)]],
    device ErrorMetrics* metrics [[buffer(2)]],
    constant ErrorParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size || !params.enable_correction) return;
    
    // Initialize correction tracking
    uint start_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics->correction_attempts, 1, memory_order_relaxed);
    
    // Load and validate data
    float4x4 tensor = data[tid];
    atomic_fetch_add_explicit(metrics->memory_operations, 1, memory_order_relaxed);
    
    bool needs_correction = false;
    bool correction_successful = true;
    bool pattern_based_correction = false;
    
    // Track correction statistics
    float max_correction = 0.0f;
    float sum_correction = 0.0f;
    float sum_squared_correction = 0.0f;
    uint corrected_elements = 0;
    
    // Pattern-based correction with comprehensive monitoring
    if (params.track_patterns && params.use_pattern_matching) {
        float pattern_confidence;
        bool has_pattern = detect_error_pattern(tensor, patterns, pattern_confidence, metrics, params);
        
        if (has_pattern && pattern_confidence >= params.confidence_threshold) {
            pattern_based_correction = true;
            
            // Apply pattern-based correction with monitoring
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    float val = tensor[i][j];
                    float pattern_val = patterns[0].signature[i][j];
                    
                    if (abs(val - pattern_val) > params.tolerance) {
                        float correction = pattern_val - val;
                        tensor[i][j] = pattern_val;
                        
                        // Track correction statistics
                        float correction_mag = abs(correction);
                        max_correction = max(max_correction, correction_mag);
                        sum_correction += correction_mag;
                        sum_squared_correction += correction_mag * correction_mag;
                        corrected_elements++;
                        needs_correction = true;
                    }
                }
            }
            
            atomic_fetch_add_explicit(metrics->pattern_matches, 1, memory_order_relaxed);
        }
    }
    
    // Value-based correction for remaining errors
    if (!pattern_based_correction) {
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = tensor[i][j];
                
                // Handle numerical errors
                if (isnan(val) || isinf(val)) {
                    tensor[i][j] = 0.0f;
                    needs_correction = true;
                    continue;
                }
                
                // Apply value range correction
                float abs_val = abs(val);
                if (abs_val > params.tolerance) {
                    float corrected_val = clamp(val, -params.tolerance, params.tolerance);
                    float correction = abs(val - corrected_val);
                    
                    tensor[i][j] = corrected_val;
                    max_correction = max(max_correction, correction);
                    sum_correction += correction;
                    sum_squared_correction += correction * correction;
                    corrected_elements++;
                    needs_correction = true;
                }
            }
        }
    }
    
    // Validate correction with comprehensive monitoring
    if (needs_correction) {
        // Normalize corrected tensor
        float max_magnitude;
        float avg_magnitude;
        uint valid_elements;
        tensor = normalize_matrix(tensor, max_magnitude, avg_magnitude, valid_elements, metrics);
        
        if (max_magnitude <= MAX_MAGNITUDE && is_valid_float4x4(tensor)) {
            data[tid] = tensor;
            atomic_fetch_add_explicit(metrics->successful_corrections, 1, memory_order_relaxed);
            
            // Update correction metrics (thread 0 only)
            if (tid == 0 && corrected_elements > 0) {
                // Update correction magnitude metrics
                metrics->max_observed_error = max_correction;
                float avg_correction = sum_correction / float(corrected_elements);
                metrics->avg_error_magnitude = avg_correction;
                
                // Update correction efficiency metrics
                metrics->correction_efficiency = float(atomic_load_explicit(metrics->successful_corrections, memory_order_relaxed)) /
                                              float(atomic_load_explicit(metrics->correction_attempts, memory_order_relaxed));
                
                // Update performance metrics
                uint end_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
                metrics->correction_latency = float(end_time - start_time);
                
                // Update resource metrics
                metrics->compute_overhead = float(end_time - start_time) / float(params.max_iterations);
                metrics->thread_efficiency = float(corrected_elements) / float(16); // 4x4 matrix
                
                // Update adaptation metrics
                metrics->adaptation_rate = pattern_based_correction ? 
                    metrics->pattern_correlation : metrics->correction_efficiency;
            }
        } else {
            data[tid] = float4x4(0.0f);
            atomic_fetch_add_explicit(metrics->stability_warnings, 1, memory_order_relaxed);
            correction_successful = false;
        }
    }
    
    // Update final metrics (thread 0 only)
    if (tid == 0) {
        metrics->recovery_rate = float(atomic_load_explicit(metrics->successful_corrections, memory_order_relaxed)) /
                               float(atomic_load_explicit(metrics->correction_attempts, memory_order_relaxed));
        metrics->recovery_possible = correction_successful;
        metrics->requires_recalibration = !correction_successful || 
                                        metrics->recovery_rate < params.min_confidence;
        metrics->has_resource_pressure = metrics->compute_overhead > 0.9f || 
                                       metrics->memory_overhead > 0.9f;
    }
}
