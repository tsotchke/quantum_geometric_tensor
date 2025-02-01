#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability and performance
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Workgroup size optimization
constant uint WORKGROUP_SIZE = 32;
constant uint VECTOR_SIZE = 4;
constant uint TILE_SIZE = 16;

// Memory layout optimization
struct alignas(16) GeometricTensorOptimized {
    packed_float4 metric[4];      // Aligned geometric metric tensor
    packed_float4 connection[4];  // Aligned geometric connection coefficients
    packed_float4 curvature[4];  // Aligned geometric curvature tensor
    packed_float4 riemann[4];    // Aligned Riemann curvature tensor
};

// Optimized AMX configuration for Apple Silicon
struct alignas(16) AMXConfigOptimized {
    packed_uint4 config;        // AMX configuration
    packed_float4 weights[4];   // AMX weights matrix (aligned)
    packed_float4 bias;         // AMX bias vector (aligned)
    uint workgroup_size;        // Dynamic workgroup size
    uint vector_width;          // SIMD vector width
    uint use_fast_math : 1;     // Fast math optimizations
    uint pad : 31;              // Padding for alignment
};

// Threadgroup memory for shared data
struct ThreadgroupMemory {
    float4x4 shared_metric;
    float4x4 shared_connection;
    float4 shared_results[WORKGROUP_SIZE];
};

// Helper functions for numerical stability
inline bool is_valid_float(float x) {
    float abs_x = abs(x);
    return !isnan(x) && !isinf(x) && 
           (abs_x == 0.0f || (abs_x >= MIN_MAGNITUDE && abs_x <= MAX_MAGNITUDE));
}

inline bool is_valid_float4(float4 v) {
    float4 abs_v = abs(v);
    return !any(isnan(v)) && !any(isinf(v)) && 
           all((abs_v == float4(0.0f)) || 
               (abs_v >= float4(MIN_MAGNITUDE) && abs_v <= float4(MAX_MAGNITUDE)));
}

inline bool is_valid_float4x4(float4x4 m) {
    return is_valid_float4(m[0]) && is_valid_float4(m[1]) && 
           is_valid_float4(m[2]) && is_valid_float4(m[3]);
}

inline float4 normalize_magnitude(float4 v, float max_allowed = MAX_MAGNITUDE) {
    float mag = length(v);
    if (mag > max_allowed) {
        return v * (max_allowed / mag);
    }
    if (mag > 0.0f && mag < MIN_MAGNITUDE) {
        return v * (MIN_MAGNITUDE / mag);
    }
    return v;
}

inline float4x4 normalize_matrix(float4x4 m, thread float& max_magnitude) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    float min_magnitude = INFINITY;
    
    // Find maximum and minimum non-zero magnitude
    for (uint i = 0; i < 4; i++) {
        float mag = length(m[i]);
        if (mag > 0.0f) {
            max_magnitude = max(max_magnitude, mag);
            min_magnitude = min(min_magnitude, mag);
        }
    }
    
    // Scale if needed
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            result[i] *= scale;
        }
    } else if (min_magnitude < MIN_MAGNITUDE && min_magnitude > 0.0f) {
        float scale = MIN_MAGNITUDE / min_magnitude;
        for (uint i = 0; i < 4; i++) {
            if (length(m[i]) > 0.0f) {
                result[i] *= scale;
            }
        }
    }
    
    return result;
}

// Matrix multiplication helper
inline float4x4 matrix_multiply(float4x4 a, float4x4 b) {
    float4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

// Vector-matrix multiplication helper
inline float4 matrix_multiply(float4x4 m, float4 v) {
    return float4(dot(m[0], v), dot(m[1], v), dot(m[2], v), dot(m[3], v));
}

// Quantum geometric tensor operations with enhanced stability
kernel void quantum_geometric_tensor_multiply(
    device const float4x4* input [[buffer(0)]],
    device const GeometricTensorOptimized& geometry [[buffer(1)]],
    device float4x4* output [[buffer(2)]],
    device const AMXConfigOptimized& amx [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 gs [[threadgroups_per_grid]],
    threadgroup ThreadgroupMemory& shared [[threadgroup(0)]]
) {
    // Optimized bounds checking
    if (any(gid >= uint2(4))) return;
    
    // Load geometry into threadgroup memory for faster access
    if (all(lid < uint2(4))) {
        shared.shared_metric[lid.x][lid.y] = geometry.metric[lid.x][lid.y];
        shared.shared_connection[lid.x][lid.y] = geometry.connection[lid.x][lid.y];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Validate tensors using vectorized operations
    bool4 valid_metric = is_valid_float4(shared.shared_metric[gid.x]);
    bool4 valid_connection = is_valid_float4(shared.shared_connection[gid.x]);
    if (!all(valid_metric) || !all(valid_connection)) {
        output[gid.x] = float4x4(0.0f);
        return;
    }
    
    // Optimized input loading with SIMD operations
    device atomic_uint* valid_count = (device atomic_uint*)(output + gs.x * gs.y);
    device atomic_uint* max_magnitude_bits = (device atomic_uint*)(output + gs.x * gs.y + 1);
    
    if (lid.x == 0 && lid.y == 0) {
        atomic_store_explicit(valid_count, 0, memory_order_relaxed);
        atomic_store_explicit(max_magnitude_bits, as_type<uint>(0.0f), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (all(lid < uint2(4))) {
        float4 row = input[gid.x][lid.x];
        
        if (all(is_valid_float4(row))) {
            atomic_fetch_add_explicit(valid_count, 1, memory_order_relaxed);
        }
        
        float row_magnitude = length(row);
        uint current_max_bits = atomic_load_explicit(max_magnitude_bits, memory_order_relaxed);
        float current_max = as_type<float>(current_max_bits);
        
        if (row_magnitude > current_max) {
            uint expected = current_max_bits;
            uint desired = as_type<uint>(row_magnitude);
            while (row_magnitude > current_max) {
                bool success = atomic_compare_exchange_weak_explicit(
                    max_magnitude_bits,
                    &expected,
                    desired,
                    memory_order_relaxed,
                    memory_order_relaxed
                );
                if (success) break;
                current_max = as_type<float>(expected);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Check input validity
    uint valid_inputs = atomic_load_explicit(valid_count, memory_order_relaxed);
    if (valid_inputs < 4) {
        output[gid.x] = float4x4(0.0f);
        return;
    }
    
    // Scale input if needed
    float current_max = as_type<float>(atomic_load_explicit(max_magnitude_bits, memory_order_relaxed));
    float scale = 1.0f;
    if (current_max > MAX_MAGNITUDE) {
        scale = MAX_MAGNITUDE / current_max;
    }
    
    // Optimized geometric operations using SIMD and tiling
    float4 row = input[gid.x][gid.y] * scale;
    
    // Process in tiles for better cache utilization
    float4 accumulated_result = 0.0f;
    for (uint tile = 0; tile < 4; tile += TILE_SIZE) {
        // Load tile into threadgroup memory
        if (all(lid < uint2(min(TILE_SIZE, 4 - tile)))) {
            shared.shared_results[lid.x] = shared.shared_metric[lid.x + tile];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process tile
        for (uint i = 0; i < min(TILE_SIZE, 4 - tile); i++) {
            accumulated_result += shared.shared_results[i] * row[i + tile];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Error checking using ERROR_THRESHOLD
    bool valid_result = true;
    for (uint i = 0; i < VECTOR_SIZE; i++) {
        if (abs(accumulated_result[i]) < ERROR_THRESHOLD) {
            valid_result = false;
            break;
        }
    }
    
    float4 metric_correction = valid_result ? normalize_magnitude(accumulated_result) : 0.0f;
    
    float4 connection_correction = matrix_multiply(shared.shared_connection, metric_correction);
    connection_correction = normalize_magnitude(connection_correction);
    
    // Optimized curvature and Riemann calculations
    float4x4 curvature_matrix = float4x4(
        geometry.curvature[0], geometry.curvature[1],
        geometry.curvature[2], geometry.curvature[3]
    );
    float4 curvature_correction = matrix_multiply(curvature_matrix, connection_correction);
    curvature_correction = normalize_magnitude(curvature_correction);
    
    float4x4 riemann_matrix = float4x4(
        geometry.riemann[0], geometry.riemann[1],
        geometry.riemann[2], geometry.riemann[3]
    );
    float4 riemann_correction = matrix_multiply(riemann_matrix, curvature_correction);
    riemann_correction = normalize_magnitude(riemann_correction);
    
    // Optimized AMX acceleration with SIMD
    if (amx.config.x > 0) {
        bool4 valid_weights = is_valid_float4(amx.weights[gid.x]);
        bool valid_bias = is_valid_float4(amx.bias);
        
        if (all(valid_weights) && valid_bias) {
            float4x4 weight_matrix = float4x4(
                amx.weights[0], amx.weights[1],
                amx.weights[2], amx.weights[3]
            );
            float4 amx_result = matrix_multiply(weight_matrix, riemann_correction);
            amx_result = fma(amx_result, amx.bias, 0.0f);
            output[gid.x][gid.y] = normalize_magnitude(amx_result);
            return;
        }
    }
    
    // Fallback path
    output[gid.x][gid.y] = riemann_correction;
}

// Quantum geometric attention with enhanced stability
kernel void quantum_geometric_attention_tensor(
    device const float4x4* queries [[buffer(0)]],
    device const float4x4* keys [[buffer(1)]],
    device const float4x4* values [[buffer(2)]],
    device const GeometricTensorOptimized& geometry [[buffer(3)]],
    device float4x4* output [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup ThreadgroupMemory& shared [[threadgroup(0)]]
) {
    // Optimized bounds checking
    if (any(gid >= uint2(seq_len))) return;
    
    // Load geometry into threadgroup memory
    if (all(lid < uint2(4))) {
        shared.shared_metric[lid.x][lid.y] = geometry.metric[lid.x][lid.y];
        shared.shared_connection[lid.x][lid.y] = geometry.connection[lid.x][lid.y];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Vectorized validation
    bool4 valid_metric = is_valid_float4(shared.shared_metric[gid.x]);
    bool4 valid_connection = is_valid_float4(shared.shared_connection[gid.x]);
    if (!all(valid_metric) || !all(valid_connection)) {
        output[gid.x] = float4x4(0.0f);
        return;
    }
    
    // Load and validate matrices
    float4x4 Q = queries[gid.x];
    float4x4 K = keys[gid.y];
    float4x4 V = values[gid.y];
    
    if (!is_valid_float4x4(Q) || !is_valid_float4x4(K) || !is_valid_float4x4(V)) {
        output[gid.x] = float4x4(0.0f);
        return;
    }
    
    // Scale matrices if needed
    float max_magnitude = 0.0f;
    for (int i = 0; i < 4; i++) {
        max_magnitude = max(max_magnitude, length(Q[i]));
        max_magnitude = max(max_magnitude, length(K[i]));
        max_magnitude = max(max_magnitude, length(V[i]));
    }
    
    float scale = 1.0f;
    if (max_magnitude > MAX_MAGNITUDE) {
        scale = MAX_MAGNITUDE / max_magnitude;
        Q *= scale;
        K *= scale;
        V *= scale;
    }
    
    // Optimized attention computation with SIMD and tiling
    float4x4 QK;
    for (uint tile = 0; tile < VECTOR_SIZE; tile += TILE_SIZE) {
        // Load tile into shared memory
        if (all(lid < uint2(min(TILE_SIZE, VECTOR_SIZE - tile)))) {
            shared.shared_results[lid.x] = Q[lid.x + tile];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process tile
        for (uint i = 0; i < min(TILE_SIZE, VECTOR_SIZE - tile); i++) {
            float4 q_vec = shared.shared_results[i];
            float4 k_vec = K[i + tile];
            QK[tile + i] = q_vec * k_vec;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float4x4 metric_QK = matrix_multiply(shared.shared_metric, QK);
    
    // Find maximum value for numerical stability with error threshold
    float max_val = -INFINITY;
    bool valid_computation = true;
    for (uint i = 0; i < VECTOR_SIZE; i++) {
        for (uint j = 0; j < VECTOR_SIZE; j++) {
            float val = metric_QK[i][j];
            if (abs(val) < ERROR_THRESHOLD) {
                valid_computation = false;
                break;
            }
            max_val = max(max_val, val);
        }
        if (!valid_computation) break;
    }
    
    // Use uniform attention if computation is invalid
    if (!valid_computation) {
        output[gid.x] = float4x4(1.0f / float(VECTOR_SIZE));
        return;
    }
    
    // Compute softmax with stability
    float4x4 attention_weights;
    float sum_exp = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            attention_weights[i][j] = exp(metric_QK[i][j] - max_val);
            sum_exp += attention_weights[i][j];
        }
    }
    
    // Normalize attention weights
    if (sum_exp > MIN_MAGNITUDE) {
        attention_weights *= (1.0f / sum_exp);
    } else {
        attention_weights = float4x4(0.25f); // Uniform attention fallback
    }
    
    // Apply attention
    float4x4 attention_output = matrix_multiply(attention_weights, V);
    
    // Apply geometric corrections
    float4x4 curved_output = matrix_multiply(float4x4(
        geometry.curvature[0], geometry.curvature[1],
        geometry.curvature[2], geometry.curvature[3]
    ), attention_output);
    
    float4x4 final_output = matrix_multiply(float4x4(
        geometry.riemann[0], geometry.riemann[1],
        geometry.riemann[2], geometry.riemann[3]
    ), curved_output);
    
    // Store result with normalization
    for (int i = 0; i < 4; i++) {
        output[gid.x][i] = normalize_magnitude(final_output[i]);
    }
}
