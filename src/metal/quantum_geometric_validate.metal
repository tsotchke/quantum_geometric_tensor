#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and validation
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint MAX_VALIDATION_PATTERNS = 16;
constant uint MAX_VALIDATION_HISTORY = 32;
constant uint WARP_SIZE = 32;

// Enhanced validation parameters with comprehensive controls
struct ValidationParams {
    uint batch_size;
    uint min_pattern_size;
    uint max_pattern_size;
    uint history_window;
    float tolerance;
    float confidence_threshold;
    float stability_threshold;
    float convergence_threshold;
    bool check_symmetry;
    bool check_unitarity;
    bool check_orthogonality;
    bool check_positivity;
    bool check_hermiticity;
    bool track_performance;
    bool enable_adaptive_tolerance;
    bool track_history;
    bool enable_prediction;
    bool use_pattern_matching;
    float min_confidence;
    float max_error_rate;
    uint warmup_steps;
    uint max_iterations;
};

// Enhanced validation metrics with comprehensive monitoring
struct ValidationMetrics {
    // Core validation counters
    device atomic_uint* numerical_errors;      // NaN/Inf errors
    device atomic_uint* symmetry_violations;   // Symmetry check failures
    device atomic_uint* unitarity_violations;  // Unitarity check failures
    device atomic_uint* hermiticity_violations;// Hermiticity check failures
    device atomic_uint* positivity_violations; // Positivity check failures
    device atomic_uint* magnitude_errors;      // Value range errors
    device atomic_uint* computation_time;      // Performance tracking
    device atomic_uint* memory_transfers;      // Memory operation tracking
    device atomic_uint* validation_attempts;   // Number of validation attempts
    device atomic_uint* successful_validations;// Successfully validated operations
    device atomic_uint* cache_misses;         // Cache miss counter
    device atomic_uint* pattern_matches;      // Pattern match counter
    
    // Statistical metrics
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
    float validation_efficiency;       // Validation success rate
    float memory_efficiency;           // Memory usage efficiency
    float compute_efficiency;          // Computational efficiency
    float response_time;              // Validation response time
    float detection_latency;          // Error detection delay
    float correction_latency;         // Error correction delay
    float prediction_accuracy;        // Error prediction accuracy
    float adaptation_rate;            // System adaptation rate
    
    // Resource metrics
    float memory_overhead;            // Memory usage overhead
    float compute_overhead;           // Computational overhead
    float bandwidth_utilization;      // Memory bandwidth usage
    float resource_efficiency;        // Resource utilization efficiency
    float cache_efficiency;          // Cache utilization
    float thread_efficiency;         // Thread utilization
    float memory_coherence;          // Memory coherence metric
    float resource_balance;          // Resource balance metric
    
    // Status flags
    bool has_critical_errors;         // Critical error indicator
    bool requires_revalidation;       // Revalidation requirement flag
    bool reached_iteration_limit;     // Iteration limit indicator
    bool has_resource_pressure;       // Resource pressure indicator
    bool needs_optimization;         // Optimization needed flag
    bool requires_recalibration;     // Recalibration needed flag
    bool has_pattern_instability;    // Pattern instability flag
    bool requires_reinitialization;  // Reinitialization needed flag
};

// Enhanced validation pattern structure with comprehensive tracking
struct ValidationPattern {
    float4x4 signature;              // Validation pattern matrix
    float4x4 correlation_matrix;     // Pattern correlation matrix
    float4x4 evolution_history[MAX_VALIDATION_HISTORY];  // Pattern evolution history using constant
    float confidence;                // Pattern confidence score
    float stability;                // Pattern stability score
    float persistence;              // Pattern persistence score
    float predictability;           // Pattern predictability score
    uint frequency;                  // Pattern occurrence count
    float correlation;              // Pattern correlation score
    float evolution_rate;           // Pattern evolution rate
    float complexity;               // Pattern complexity score
    bool is_critical;               // Critical pattern flag
    bool is_stable;                // Pattern stability flag
    bool requires_attention;        // Pattern attention flag
    bool has_evolved;              // Pattern evolution flag
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(m[i][j]);
            if (isnan(m[i][j]) || isinf(m[i][j]) || 
                val > MAX_MAGNITUDE || val < MIN_MAGNITUDE) { // Now using MIN_MAGNITUDE
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
    device ValidationMetrics* metrics
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
    if (max_magnitude > MAX_MAGNITUDE || max_magnitude < MIN_MAGNITUDE) { // Added MIN_MAGNITUDE check
        float scale = (max_magnitude > MAX_MAGNITUDE) ? 
                     MAX_MAGNITUDE / max_magnitude : 
                     MIN_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            result[i] = m[i] * scale;
        }
        atomic_fetch_add_explicit(metrics->magnitude_errors, 1, memory_order_relaxed);
    }
    
    // Update metrics
    metrics->error_variance = variance;
    
    return result;
}

// Enhanced validation pattern detection with comprehensive analysis
inline bool detect_validation_pattern(
    float4x4 data,
    device ValidationPattern* patterns,
    thread float& confidence,
    device ValidationMetrics* metrics,
    constant ValidationParams& params
) {
    float max_correlation = 0.0f;
    float avg_correlation = 0.0f;
    uint best_pattern = 0;
    bool pattern_found = false;
    uint valid_correlations = 0;
    
    // Process patterns in WARP_SIZE chunks for better memory access patterns
    for (uint p = 0; p < MAX_VALIDATION_PATTERNS; p += WARP_SIZE) {
        uint chunk_size = min(WARP_SIZE, MAX_VALIDATION_PATTERNS - p);
        
        // Compare with known patterns
        for (uint i = 0; i < chunk_size; i++) {
            uint pattern_idx = p + i;
            float correlation = 0.0f;
            float stability = 0.0f;
            uint matches = 0;
            
            // Compute pattern correlation and stability
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    float diff = abs(data[i][j] - patterns[pattern_idx].signature[i][j]);
                    if (diff < params.tolerance) {
                        float match_strength = 1.0f - (diff / params.tolerance);
                        correlation += match_strength;
                        stability += 1.0f - abs(patterns[pattern_idx].evolution_rate);
                        matches++;
                    }
                }
            }
            
            // Normalize metrics
            if (matches > 0) {
                correlation /= float(matches);
                stability /= float(matches);
                
                // Update pattern statistics
                patterns[pattern_idx].correlation = correlation;
                patterns[pattern_idx].stability = stability;
                patterns[pattern_idx].frequency++;
                
                // Track evolution history using MAX_VALIDATION_HISTORY
                uint history_idx = patterns[pattern_idx].frequency % MAX_VALIDATION_HISTORY;
                patterns[pattern_idx].evolution_history[history_idx] = data;
                
                // Track evolution
                if (correlation > patterns[pattern_idx].confidence) {
                    patterns[pattern_idx].has_evolved = true;
                    patterns[pattern_idx].evolution_rate = (correlation - patterns[pattern_idx].confidence) / correlation;
                }
                
                // Update correlation statistics
                max_correlation = max(max_correlation, correlation);
                avg_correlation += correlation;
                valid_correlations++;
                
                if (correlation > params.confidence_threshold) {
                    best_pattern = pattern_idx;
                    pattern_found = true;
                }
            }
        }
    }
    
    // Update pattern metrics
    if (valid_correlations > 0) {
        avg_correlation /= float(valid_correlations);
        metrics->pattern_correlation = avg_correlation;
        metrics->pattern_stability = patterns[best_pattern].stability;
        metrics->pattern_evolution = patterns[best_pattern].evolution_rate;
        metrics->pattern_predictability = pattern_found ? 
            patterns[best_pattern].predictability : 0.0f;
    }
    
    confidence = max_correlation;
    return pattern_found && confidence >= params.confidence_threshold;
}

// Enhanced kernel for comprehensive tensor validation
kernel void validate_quantum_tensor(
    device const float4x4* tensor [[buffer(0)]],
    device ValidationPattern* patterns [[buffer(1)]],
    device ValidationMetrics* metrics [[buffer(2)]],
    constant ValidationParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Use WARP_SIZE for thread group optimization
    uint warp_id = tid / WARP_SIZE;
    uint lane_id = tid % WARP_SIZE;
    uint batch_id = warp_id * WARP_SIZE + lane_id;
    
    if (batch_id >= params.batch_size) return;
    
    // Initialize performance tracking
    uint start_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics->validation_attempts, 1, memory_order_relaxed);
    
    // Load and validate tensor with performance tracking
    float4x4 data = tensor[batch_id];
    atomic_fetch_add_explicit(metrics->memory_transfers, 1, memory_order_relaxed);
    
    bool has_numerical_error = false;
    bool has_symmetry_violation = false;
    bool has_unitarity_violation = false;
    bool has_hermiticity_violation = false;
    bool has_positivity_violation = false;
    bool has_pattern = false;
    
    // Track validation statistics
    float max_error = 0.0f;
    float min_error = INFINITY;
    float sum_error = 0.0f;
    float sum_squared_error = 0.0f;
    uint valid_elements = 0;
    
    if (!is_valid_float4x4(data)) {
        atomic_fetch_add_explicit(metrics->numerical_errors, 1, memory_order_relaxed);
        has_numerical_error = true;
    } else {
        // Normalize tensor for stability
        float max_magnitude;
        float avg_magnitude;
        uint norm_valid_elements;
        data = normalize_matrix(data, max_magnitude, avg_magnitude, norm_valid_elements, metrics);
        
        // Pattern-based validation with comprehensive monitoring
        if (params.track_history && params.use_pattern_matching) {
            float pattern_confidence;
            has_pattern = detect_validation_pattern(data, patterns, pattern_confidence, metrics, params);
            
            if (has_pattern) {
                atomic_fetch_add_explicit(metrics->pattern_matches, 1, memory_order_relaxed);
                metrics->pattern_confidence = max(metrics->pattern_confidence, pattern_confidence);
            }
        }
        
        // Comprehensive validation checks
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                float val = data[i][j];
                
                // Track error statistics
                float error_mag = abs(val);
                max_error = max(max_error, error_mag);
                min_error = min(min_error, error_mag);
                sum_error += error_mag;
                sum_squared_error += error_mag * error_mag;
                valid_elements++;
                
                // Check symmetry if required
                if (params.check_symmetry && i != j) {
                    float sym_diff = abs(data[i][j] - data[j][i]);
                    if (sym_diff > params.tolerance) {
                        atomic_fetch_add_explicit(metrics->symmetry_violations, 1, memory_order_relaxed);
                        has_symmetry_violation = true;
                    }
                }
                
                // Check hermiticity if required
                if (params.check_hermiticity && i != j) {
                    float herm_diff = abs(data[i][j] - data[j][i]);
                    if (herm_diff > params.tolerance) {
                        atomic_fetch_add_explicit(metrics->hermiticity_violations, 1, memory_order_relaxed);
                        has_hermiticity_violation = true;
                    }
                }
                
                // Check positivity if required
                if (params.check_positivity && val < -params.tolerance) {
                    atomic_fetch_add_explicit(metrics->positivity_violations, 1, memory_order_relaxed);
                    has_positivity_violation = true;
                }
            }
        }
        
        // Check unitarity if required
        if (params.check_unitarity) {
            float4x4 product = matrix_multiply(data, transpose(data));
            float max_unitarity_error = 0.0f;
            
            for (uint i = 0; i < 4; i++) {
                float diff = abs(product[i][i] - 1.0f);
                max_unitarity_error = max(max_unitarity_error, diff);
                
                if (diff > params.tolerance) {
                    atomic_fetch_add_explicit(metrics->unitarity_violations, 1, memory_order_relaxed);
                    has_unitarity_violation = true;
                }
            }
            
            max_error = max(max_error, max_unitarity_error);
        }
    }
    
    // Update validation metrics atomically
    if (lane_id == 0) { // Only first thread in warp updates shared metrics
        // Update error statistics
        metrics->max_observed_error = max_error;
        metrics->min_observed_error = min_error;
        metrics->avg_error_magnitude = sum_error / float(valid_elements);
        
        // Compute error distribution metrics
        float mean = sum_error / float(valid_elements);
        metrics->error_variance = (sum_squared_error / float(valid_elements)) - (mean * mean);
        metrics->error_distribution = metrics->error_variance / (mean * mean);
        
        // Update validation metrics
        metrics->validation_efficiency = float(atomic_load_explicit(metrics->successful_validations, memory_order_relaxed)) /
                                      float(atomic_load_explicit(metrics->validation_attempts, memory_order_relaxed));
        
        // Update performance metrics
        uint end_time = atomic_load_explicit(metrics->computation_time, memory_order_relaxed);
        metrics->response_time = float(end_time - start_time);
        
        metrics->memory_efficiency = float(atomic_load_explicit(metrics->successful_validations, memory_order_relaxed)) /
                                   float(atomic_load_explicit(metrics->memory_transfers, memory_order_relaxed));
        
        metrics->compute_efficiency = float(atomic_load_explicit(metrics->successful_validations, memory_order_relaxed)) /
                                    float(atomic_load_explicit(metrics->computation_time, memory_order_relaxed));
        
        // Update resource metrics
        metrics->memory_overhead = float(atomic_load_explicit(metrics->memory_transfers, memory_order_relaxed)) * 
                                 sizeof(float4x4);
        metrics->compute_overhead = float(end_time - start_time) / float(params.max_iterations);
        metrics->bandwidth_utilization = metrics->memory_overhead / metrics->compute_overhead;
        metrics->resource_efficiency = metrics->validation_efficiency * metrics->memory_efficiency;
        
        // Update thread efficiency based on WARP_SIZE utilization
        metrics->thread_efficiency = float(params.batch_size % WARP_SIZE) / float(WARP_SIZE);
        
        // Update status flags
        metrics->has_critical_errors = has_numerical_error || 
                                     (has_symmetry_violation && params.check_symmetry) ||
                                     (has_unitarity_violation && params.check_unitarity) ||
                                     (has_hermiticity_violation && params.check_hermiticity) ||
                                     (has_positivity_violation && params.check_positivity);
        
        metrics->requires_revalidation = metrics->has_critical_errors && 
                                       metrics->validation_efficiency < params.min_confidence;
        
        metrics->reached_iteration_limit = atomic_load_explicit(metrics->validation_attempts, memory_order_relaxed) >= 
                                         params.max_iterations;
        
        metrics->has_resource_pressure = metrics->compute_overhead > 0.9f || 
                                       metrics->memory_overhead > 0.9f;
        
        metrics->needs_optimization = metrics->has_resource_pressure || 
                                    metrics->validation_efficiency < 0.7f;
        
        metrics->requires_recalibration = metrics->has_critical_errors || 
                                        metrics->error_variance > params.stability_threshold;
        
        metrics->has_pattern_instability = has_pattern && 
                                         metrics->pattern_evolution > params.stability_threshold;
    }
    
    // Track successful validation
    if (!metrics->has_critical_errors) {
        atomic_fetch_add_explicit(metrics->successful_validations, 1, memory_order_relaxed);
    }
}
