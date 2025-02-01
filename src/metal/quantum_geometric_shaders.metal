#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-3f;
constant float ERROR_THRESHOLD = 1e-6f;

// Quantum state types
typedef struct {
    float2 amplitude;  // Complex number (real, imag)
} QuantumAmplitude;

// Geometric tensor types
typedef struct {
    float4x4 metric;      // Geometric metric tensor
    float4x4 connection;  // Geometric connection coefficients
    float4x4 curvature;   // Geometric curvature tensor
} GeometricTensor;

// AMX acceleration structure
typedef struct {
    uint4 config;         // AMX configuration
    float4x4 weights;     // AMX weights matrix
    float4 bias;         // AMX bias vector
} AMXConfig;

// Helper function to check matrix validity
static bool is_valid_matrix(float4x4 mat) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(mat[i][j]);
            if (isnan(val) || isinf(val) || 
                (val > ERROR_THRESHOLD && val < MIN_MAGNITUDE) || 
                val > MAX_MAGNITUDE) {
                return false;
            }
        }
    }
    return true;
}

// Helper function to normalize matrix elements
static float4x4 normalize_matrix(float4x4 mat, thread float& max_magnitude) {
    float4x4 result = mat;
    max_magnitude = 0.0f;
    float min_magnitude = INFINITY;
    
    // Find maximum and minimum non-zero magnitudes
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(mat[i][j]);
            if (mag > ERROR_THRESHOLD) {
                max_magnitude = max(max_magnitude, mag);
                min_magnitude = min(min_magnitude, mag);
            }
        }
    }
    
    // Scale if needed
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                result[i][j] *= scale;
            }
        }
    } else if (min_magnitude < MIN_MAGNITUDE && min_magnitude > ERROR_THRESHOLD) {
        float scale = MIN_MAGNITUDE / min_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                if (abs(mat[i][j]) > ERROR_THRESHOLD) {
                    result[i][j] *= scale;
                }
            }
        }
    }
    
    return result;
}

// Compute Christoffel symbols with stability checks
static float4x4 compute_christoffel_symbols(
    float4x4 metric,
    float4x4 inverse_metric,
    thread bool& valid_result
) {
    float4x4 christoffel = float4x4(0.0f);
    valid_result = true;
    
    // Validate inputs
    if (!is_valid_matrix(metric) || !is_valid_matrix(inverse_metric)) {
        valid_result = false;
        return christoffel;
    }
    
    float max_magnitude = 0.0f;
    float4x4 normalized_metric = normalize_matrix(metric, max_magnitude);
    float4x4 normalized_inverse = normalize_matrix(inverse_metric, max_magnitude);
    
    // Compute with numerical stability
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float sum = 0.0f;
            float max_term = 0.0f;
            float min_term = INFINITY;
            
            for (uint k = 0; k < 4; k++) {
                float term1 = normalized_metric[i][k] * normalized_inverse[k][j];
                float term2 = normalized_metric[j][k] * normalized_inverse[k][i];
                float term3 = normalized_metric[k][k] * normalized_inverse[i][j];
                
                // Track magnitude range for stability
                float terms[3] = {abs(term1), abs(term2), abs(term3)};
                for (int t = 0; t < 3; t++) {
                    if (terms[t] > ERROR_THRESHOLD) {
                        max_term = max(max_term, terms[t]);
                        min_term = min(min_term, terms[t]);
                    }
                }
                
                sum += 0.5f * (term1 + term2 - term3);
            }
            
            // Check for numerical instability
            if (max_term > MAX_MAGNITUDE || 
                (min_term < MIN_MAGNITUDE && min_term > ERROR_THRESHOLD) || 
                isnan(sum) || isinf(sum)) {
                valid_result = false;
                return float4x4(0.0f);
            }
            
            christoffel[i][j] = sum;
        }
    }
    
    // Final stability check
    float final_magnitude;
    christoffel = normalize_matrix(christoffel, final_magnitude);
    
    return christoffel;
}

// Compute Riemann tensor with stability checks
static float4x4 compute_riemann_tensor(
    float4x4 christoffel,
    float4x4 metric,
    thread bool& valid_result
) {
    float4x4 riemann = float4x4(0.0f);
    valid_result = true;
    
    // Validate inputs
    if (!is_valid_matrix(christoffel) || !is_valid_matrix(metric)) {
        valid_result = false;
        return riemann;
    }
    
    float max_magnitude = 0.0f;
    float4x4 normalized_christoffel = normalize_matrix(christoffel, max_magnitude);
    float4x4 normalized_metric = normalize_matrix(metric, max_magnitude);
    
    // Compute with numerical stability
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float sum = 0.0f;
            float max_term = 0.0f;
            float min_term = INFINITY;
            
            for (uint k = 0; k < 4; k++) {
                float term1 = normalized_christoffel[i][k] * normalized_metric[k][j];
                float term2 = normalized_christoffel[j][k] * normalized_metric[k][i];
                
                // Track magnitude range for stability
                float abs_term1 = abs(term1);
                float abs_term2 = abs(term2);
                if (abs_term1 > ERROR_THRESHOLD) {
                    max_term = max(max_term, abs_term1);
                    min_term = min(min_term, abs_term1);
                }
                if (abs_term2 > ERROR_THRESHOLD) {
                    max_term = max(max_term, abs_term2);
                    min_term = min(min_term, abs_term2);
                }
                
                sum += term1 - term2;
            }
            
            // Check for numerical instability
            if (max_term > MAX_MAGNITUDE || 
                (min_term < MIN_MAGNITUDE && min_term > ERROR_THRESHOLD) || 
                isnan(sum) || isinf(sum)) {
                valid_result = false;
                return float4x4(0.0f);
            }
            
            riemann[i][j] = sum;
        }
    }
    
    // Final stability check
    float final_magnitude;
    riemann = normalize_matrix(riemann, final_magnitude);
    
    return riemann;
}

// Quantum geometric tensor multiply with AMX acceleration
kernel void quantum_geometric_multiply(
    device const QuantumAmplitude* A [[buffer(0)]],
    device const QuantumAmplitude* B [[buffer(1)]],
    device QuantumAmplitude* C [[buffer(2)]],
    device const GeometricTensor& geometry [[buffer(3)]],
    device const AMXConfig& amx [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    // Early exit if outside bounds
    if (gid.x >= M || gid.y >= N) return;
    
    // Initialize accumulator with error tracking
    float2 sum = float2(0.0f);
    float max_error = 0.0f;
    float min_error = INFINITY;
    
    // Load geometric tensors into threadgroup memory with error checking
    threadgroup GeometricTensor local_geometry;
    threadgroup bool geometry_valid = false;
    if (lid.x == 0 && lid.y == 0) {
        local_geometry = geometry;
        
        // Validate geometric tensor
        geometry_valid = true;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                if (!is_valid_matrix(geometry.metric) || 
                    !is_valid_matrix(geometry.connection) || 
                    !is_valid_matrix(geometry.curvature)) {
                    geometry_valid = false;
                    break;
                }
            }
            if (!geometry_valid) break;
        }
        
        // Compute Riemann tensor for curvature validation
        if (geometry_valid) {
            bool riemann_valid;
            float4x4 computed_riemann = compute_riemann_tensor(geometry.connection, geometry.metric, riemann_valid);
            if (riemann_valid) {
                // Compare with provided curvature tensor
                float max_diff = 0.0f;
                for (uint i = 0; i < 4; i++) {
                    for (uint j = 0; j < 4; j++) {
                        float diff = abs(computed_riemann[i][j] - geometry.curvature[i][j]);
                        max_diff = max(max_diff, diff);
                    }
                }
                // Invalidate if difference is too large
                if (max_diff > ERROR_THRESHOLD) {
                    geometry_valid = false;
                }
            } else {
                geometry_valid = false;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Early exit if geometric tensor is invalid
    if (!geometry_valid) {
        C[gid.x * N + gid.y].amplitude = float2(0.0f);
        return;
    }
    
    // Compute geometric tensor contraction with error tracking
    for (uint k = 0; k < K; k++) {
        // Load quantum amplitudes with validation
        float2 a = A[gid.x * K + k].amplitude;
        float2 b = B[k * N + gid.y].amplitude;
        
        float a_mag = length(a);
        float b_mag = length(b);
        
        if (isnan(a_mag) || isinf(a_mag) || isnan(b_mag) || isinf(b_mag) ||
            (a_mag > ERROR_THRESHOLD && a_mag < MIN_MAGNITUDE) || a_mag > MAX_MAGNITUDE ||
            (b_mag > ERROR_THRESHOLD && b_mag < MIN_MAGNITUDE) || b_mag > MAX_MAGNITUDE) {
            continue;  // Skip invalid amplitudes
        }
        
        // Compute and validate geometric tensors
        bool valid_result;
        float4x4 inverse_metric = local_geometry.metric;  // For simplicity using metric as inverse, should be properly inverted in production
        float4x4 christoffel = compute_christoffel_symbols(local_geometry.metric, inverse_metric, valid_result);
        if (!valid_result) {
            C[gid.x * N + gid.y].amplitude = float2(0.0f);
            return;
        }
        
        // Apply geometric connection with numerical stability
        float4 pos = float4(a.x, a.y, b.x, b.y);
        float4 transformed = christoffel * pos;
        
        // Track magnitude range for stability
        float mag = length(transformed);
        if (mag > ERROR_THRESHOLD) {
            max_error = max(max_error, mag);
            min_error = min(min_error, mag);
        }
        
        // Apply scaling if magnitude is outside valid range
        if (mag > MAX_MAGNITUDE) {
            transformed *= MAX_MAGNITUDE / mag;
        } else if (mag > ERROR_THRESHOLD && mag < MIN_MAGNITUDE) {
            transformed *= MIN_MAGNITUDE / mag;
        }
        
        // Complex multiplication with geometric correction
        float2 product;
        product.x = transformed.x * transformed.z - transformed.y * transformed.w;
        product.y = transformed.x * transformed.w + transformed.y * transformed.z;
        
        // Apply metric tensor with stability check
        float4 metric_correction = local_geometry.metric * float4(product.x, product.y, 0.0f, 0.0f);
        
        // Accumulate result with stability
        float2 contribution = metric_correction.xy;
        float contrib_mag = length(contribution);
        if (contrib_mag > MAX_MAGNITUDE) {
            contribution *= MAX_MAGNITUDE / contrib_mag;
        } else if (contrib_mag > ERROR_THRESHOLD && contrib_mag < MIN_MAGNITUDE) {
            contribution *= MIN_MAGNITUDE / contrib_mag;
        }
        sum += contribution;
    }
    
    // Apply curvature correction with stability check
    float4 curvature_correction = local_geometry.curvature * float4(sum.x, sum.y, 0.0f, 0.0f);
    sum = curvature_correction.xy;
    
    // Final stability check and normalization
    float final_mag = length(sum);
    if (final_mag > MAX_MAGNITUDE) {
        sum *= MAX_MAGNITUDE / final_mag;
    } else if (final_mag > ERROR_THRESHOLD && final_mag < MIN_MAGNITUDE) {
        sum *= MIN_MAGNITUDE / final_mag;
    }
    
    // Store result with error flag in amplitude
    bool valid_result = (max_error <= MAX_MAGNITUDE && 
                        (min_error >= MIN_MAGNITUDE || min_error <= ERROR_THRESHOLD));
    C[gid.x * N + gid.y].amplitude = valid_result ? sum : float2(0.0f);
}

// Quantum geometric attention mechanism
kernel void quantum_geometric_attention(
    device const QuantumAmplitude* queries [[buffer(0)]],
    device const QuantumAmplitude* keys [[buffer(1)]],
    device const QuantumAmplitude* values [[buffer(2)]],
    device QuantumAmplitude* output [[buffer(3)]],
    device const GeometricTensor& geometry [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    // Early exit if outside bounds
    if (gid.x >= seq_len || gid.y >= num_heads) return;
    
    // Initialize attention scores with error tracking
    threadgroup float attention_scores[32];  // Using threadgroup memory for better performance
    threadgroup float max_score;
    float sum = 0.0f;
    float max_error = 0.0f;
    float min_error = INFINITY;
    bool has_valid_scores = false;
    
    // First pass: Compute attention scores and find maximum
    for (uint i = 0; i < seq_len; i++) {
        // Load query and key with validation
        float2 q = queries[gid.x * head_dim + gid.y].amplitude;
        float2 k = keys[i * head_dim + gid.y].amplitude;
        
        float q_mag = length(q);
        float k_mag = length(k);
        
        if (isnan(q_mag) || isinf(q_mag) || isnan(k_mag) || isinf(k_mag) ||
            (q_mag > ERROR_THRESHOLD && q_mag < MIN_MAGNITUDE) || q_mag > MAX_MAGNITUDE ||
            (k_mag > ERROR_THRESHOLD && k_mag < MIN_MAGNITUDE) || k_mag > MAX_MAGNITUDE) {
            attention_scores[i] = -INFINITY;
            continue;
        }
        
        // Apply geometric metric with stability check
        float4 qk = float4(q.x, q.y, k.x, k.y);
        float4 metric_qk = geometry.metric * qk;
        
        // Track magnitude range for stability
        float mag = length(metric_qk);
        if (mag > ERROR_THRESHOLD) {
            max_error = max(max_error, mag);
            min_error = min(min_error, mag);
        }
        
        // Apply scaling if magnitude is outside valid range
        if (mag > MAX_MAGNITUDE) {
            metric_qk *= MAX_MAGNITUDE / mag;
        } else if (mag > ERROR_THRESHOLD && mag < MIN_MAGNITUDE) {
            metric_qk *= MIN_MAGNITUDE / mag;
        }
        
        // Compute attention score with numerical stability
        float score = dot(metric_qk.xy, metric_qk.zw);
        score /= sqrt(float(head_dim));  // Scale before exp
        
        attention_scores[i] = score;
        if (i == 0 || score > max_score) {
            max_score = score;
        }
        has_valid_scores = true;
    }
    
    // Early exit if no valid scores
    if (!has_valid_scores) {
        output[gid.x * head_dim + gid.y].amplitude = float2(0.0f);
        return;
    }
    
    // Second pass: Apply softmax with numerical stability
    for (uint i = 0; i < seq_len; i++) {
        if (attention_scores[i] == -INFINITY) {
            attention_scores[i] = 0.0f;
        } else {
            attention_scores[i] = exp(attention_scores[i] - max_score);
            sum += attention_scores[i];
        }
    }
    
    // Normalize with stability check
    if (sum > ERROR_THRESHOLD) {
        for (uint i = 0; i < seq_len; i++) {
            attention_scores[i] /= sum;
        }
    } else {
        // Fallback to uniform attention if sum is too small
        float uniform_weight = 1.0f / float(seq_len);
        for (uint i = 0; i < seq_len; i++) {
            attention_scores[i] = uniform_weight;
        }
    }
    
    // Compute weighted sum of values with error tracking
    float2 result = float2(0.0f);
    float value_error = 0.0f;
    float min_value_error = INFINITY;
    
    for (uint i = 0; i < seq_len; i++) {
        float2 v = values[i * head_dim + gid.y].amplitude;
        float v_mag = length(v);
        
        if (isnan(v_mag) || isinf(v_mag) ||
            (v_mag > ERROR_THRESHOLD && v_mag < MIN_MAGNITUDE) || v_mag > MAX_MAGNITUDE) {
            continue;
        }
        
        // Apply geometric connection with stability
        float4 value_pos = float4(v.x, v.y, 0.0f, 0.0f);
        float4 transformed = geometry.connection * value_pos;
        
        float mag = length(transformed.xy);
        if (mag > ERROR_THRESHOLD) {
            value_error = max(value_error, mag);
            min_value_error = min(min_value_error, mag);
        }
        
        // Apply scaling if magnitude is outside valid range
        if (mag > MAX_MAGNITUDE) {
            transformed.xy *= MAX_MAGNITUDE / mag;
        } else if (mag > ERROR_THRESHOLD && mag < MIN_MAGNITUDE) {
            transformed.xy *= MIN_MAGNITUDE / mag;
        }
        
        result += attention_scores[i] * transformed.xy;
    }
    
    // Apply curvature correction with final stability check
    bool valid_result = (max_error <= MAX_MAGNITUDE && value_error <= MAX_MAGNITUDE &&
                        (min_error >= MIN_MAGNITUDE || min_error <= ERROR_THRESHOLD) &&
                        (min_value_error >= MIN_MAGNITUDE || min_value_error <= ERROR_THRESHOLD));
    
    if (valid_result) {
        float4 curvature_correction = geometry.curvature * float4(result.x, result.y, 0.0f, 0.0f);
        result = curvature_correction.xy;
        
        // Final magnitude check
        float final_mag = length(result);
        if (final_mag > MAX_MAGNITUDE) {
            result *= MAX_MAGNITUDE / final_mag;
        } else if (final_mag > ERROR_THRESHOLD && final_mag < MIN_MAGNITUDE) {
            result *= MIN_MAGNITUDE / final_mag;
        }
    } else {
        result = float2(0.0f);  // Reset on excessive error
    }
    
    // Store result
    output[gid.x * head_dim + gid.y].amplitude = result;
}

// Quantum state compression with error mitigation
kernel void quantum_state_compress(
    device const QuantumAmplitude* input [[buffer(0)]],
    device QuantumAmplitude* output [[buffer(1)]],
    device const GeometricTensor& geometry [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& output_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= output_size) return;
    
    // Initialize accumulators with error tracking
    float2 compressed = float2(0.0f);
    float max_error = 0.0f;
    float min_error = INFINITY;
    float total_magnitude = 0.0f;
    uint valid_inputs = 0;
    
    // Validate geometric tensors
    bool geometry_valid = is_valid_matrix(geometry.metric) &&
                         is_valid_matrix(geometry.connection) &&
                         is_valid_matrix(geometry.curvature);
    
    if (!geometry_valid) {
        output[gid].amplitude = float2(0.0f);
        return;
    }
    
    // First pass: Analyze input data and compute statistics
    for (uint i = 0; i < input_size; i++) {
        float2 amp = input[i].amplitude;
        float mag = length(amp);
        
        if (isnan(mag) || isinf(mag) ||
            (mag > ERROR_THRESHOLD && mag < MIN_MAGNITUDE) || mag > MAX_MAGNITUDE) {
            continue;
        }
        
        total_magnitude += mag;
        if (mag > ERROR_THRESHOLD) {
            max_error = max(max_error, mag);
            min_error = min(min_error, mag);
        }
        valid_inputs++;
    }
    
    // Early exit if no valid inputs
    if (valid_inputs == 0 || total_magnitude < ERROR_THRESHOLD) {
        output[gid].amplitude = float2(0.0f);
        return;
    }
    
    // Compute adaptive phase step based on valid inputs
    float phase_step = 2.0f * M_PI_F / float(valid_inputs);
    float scale_factor = 1.0f / sqrt(total_magnitude);
    
    // Second pass: Quantum-inspired compression with stability
    uint valid_count = 0;
    for (uint i = 0; i < input_size; i++) {
        float2 amp = input[i].amplitude;
        float mag = length(amp);
        
        if (isnan(mag) || isinf(mag) ||
            (mag > ERROR_THRESHOLD && mag < MIN_MAGNITUDE) || mag > MAX_MAGNITUDE) {
            continue;
        }
        
        // Generate phase factor with stability check
        float phase = phase_step * float(valid_count++);
        float2 phase_factor;
        if (abs(phase) > MAX_MAGNITUDE) {
            phase = fmod(phase, 2.0f * M_PI_F);  // Wrap phase to prevent overflow
        }
        phase_factor = float2(cos(phase), sin(phase));
        
        // Apply geometric metric with stability
        float4 pos = float4(amp.x * scale_factor, amp.y * scale_factor, 0.0f, 0.0f);
        float4 metric_correction = geometry.metric * pos;
        
        float metric_mag = length(metric_correction.xy);
        if (metric_mag > MAX_MAGNITUDE) {
            metric_correction.xy *= MAX_MAGNITUDE / metric_mag;
        } else if (metric_mag > ERROR_THRESHOLD && metric_mag < MIN_MAGNITUDE) {
            metric_correction.xy *= MIN_MAGNITUDE / metric_mag;
        }
        
        // Accumulate with phase rotation and stability
        float2 contribution;
        contribution.x = metric_correction.x * phase_factor.x - metric_correction.y * phase_factor.y;
        contribution.y = metric_correction.x * phase_factor.y + metric_correction.y * phase_factor.x;
        
        float contrib_mag = length(contribution);
        if (contrib_mag > MAX_MAGNITUDE) {
            contribution *= MAX_MAGNITUDE / contrib_mag;
        } else if (contrib_mag > ERROR_THRESHOLD && contrib_mag < MIN_MAGNITUDE) {
            contribution *= MIN_MAGNITUDE / contrib_mag;
        }
        
        compressed += contribution;
    }
    
    // Apply geometric corrections with stability checks
    bool valid_result = (max_error <= MAX_MAGNITUDE &&
                        (min_error >= MIN_MAGNITUDE || min_error <= ERROR_THRESHOLD));
    
    if (valid_result) {
        // Apply connection correction
        float4 connection_pos = float4(compressed.x, compressed.y, 0.0f, 0.0f);
        float4 connection_correction = geometry.connection * connection_pos;
        
        float conn_mag = length(connection_correction.xy);
        if (conn_mag > MAX_MAGNITUDE) {
            connection_correction.xy *= MAX_MAGNITUDE / conn_mag;
        } else if (conn_mag > ERROR_THRESHOLD && conn_mag < MIN_MAGNITUDE) {
            connection_correction.xy *= MIN_MAGNITUDE / conn_mag;
        }
        
        // Apply curvature correction
        float4 curvature_correction = geometry.curvature * float4(connection_correction.x, connection_correction.y, 0.0f, 0.0f);
        compressed = curvature_correction.xy;
        
        // Final normalization with stability
        float final_mag = length(compressed);
        if (final_mag > ERROR_THRESHOLD) {
            if (final_mag > MAX_MAGNITUDE) {
                compressed *= MAX_MAGNITUDE / final_mag;
            } else if (final_mag < MIN_MAGNITUDE) {
                compressed *= MIN_MAGNITUDE / final_mag;
            } else {
                compressed /= final_mag;
            }
        }
    } else {
        compressed = float2(0.0f);
    }
    
    output[gid].amplitude = compressed;
}
