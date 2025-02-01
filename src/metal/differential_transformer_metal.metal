#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

// Constants for shader configuration
constant int BLOCK_SIZE = 256;
constant float EPSILON = 1e-6;

// Structure definitions
struct DiffTransformerParams {
    uint seq_length;
    uint hidden_dim;
    uint num_heads;
    float learning_rate;
};

// Compute token derivatives
kernel void compute_token_derivatives(
    device const float* values [[buffer(0)]],
    device float* derivatives [[buffer(1)]],
    constant DiffTransformerParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.seq_length * params.hidden_dim) return;
    
    uint i = idx / params.hidden_dim;
    uint j = idx % params.hidden_dim;
    
    // Central difference approximation with gradient clipping
    float h = max(abs(values[idx]) * 1e-4, EPSILON);
    float raw_deriv = (values[idx + 1] - values[idx - 1]) / (2.0 * h);
    
    // Use i and j for positional scaling
    float scale = 1.0 / (1.0 + 0.1 * float(i + j)); // Decay factor based on position
    derivatives[idx] = raw_deriv * scale;
}

// Differential attention scores computation
kernel void differential_attention_scores(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* query_deriv [[buffer(2)]],
    device const float* key_deriv [[buffer(3)]],
    device float* scores [[buffer(4)]],
    device float* score_derivs [[buffer(5)]],
    constant DiffTransformerParams& params [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint row = pos.y;
    uint col = pos.x;
    
    if (row >= params.seq_length || col >= params.seq_length) return;
    
    uint head_dim = params.hidden_dim / params.num_heads;
    
    // Compute attention score and derivative
    float score = 0.0;
    float score_deriv = 0.0;
    
    for (uint k = 0; k < head_dim; k++) {
        float q = query[row * head_dim + k];
        float k_val = key[col * head_dim + k];
        float q_deriv = query_deriv[row * head_dim + k];
        float k_deriv = key_deriv[col * head_dim + k];
        
        score += q * k_val;
        score_deriv += q_deriv * k_val + q * k_deriv;
    }
    
    // Scale and store results
    float scale = 1.0 / sqrt(float(head_dim));
    scores[row * params.seq_length + col] = score * scale;
    score_derivs[row * params.seq_length + col] = score_deriv * scale;
}

// Differential softmax implementation
kernel void differential_softmax(
    device float* values [[buffer(0)]],
    device float* derivatives [[buffer(1)]],
    constant DiffTransformerParams& params [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]
) {
    uint row = pos.y;
    uint tid = local_pos.x;
    
    if (row >= params.seq_length) return;
    
    threadgroup float shared_max[BLOCK_SIZE];
    threadgroup float shared_sum[BLOCK_SIZE];
    threadgroup float shared_deriv_sum[BLOCK_SIZE];
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (uint j = tid; j < params.seq_length; j += group_size.x) {
        max_val = max(max_val, values[row * params.seq_length + j]);
    }
    shared_max[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global max
    for (uint stride = group_size.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute exp and sums
    float local_sum = 0.0;
    float local_deriv_sum = 0.0;
    for (uint j = tid; j < params.seq_length; j += group_size.x) {
        uint idx = row * params.seq_length + j;
        float exp_val = exp(values[idx] - max_val);
        values[idx] = exp_val;
        local_sum += exp_val;
        local_deriv_sum += derivatives[idx] * exp_val;
    }
    shared_sum[tid] = local_sum;
    shared_deriv_sum[tid] = local_deriv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sums
    for (uint stride = group_size.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_deriv_sum[tid] += shared_deriv_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared_sum[0];
    float deriv_sum = shared_deriv_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize values and compute derivatives
    float inv_sum = 1.0 / sum;
    for (uint j = tid; j < params.seq_length; j += group_size.x) {
        uint idx = row * params.seq_length + j;
        float softmax_val = values[idx] * inv_sum;
        float softmax_deriv = derivatives[idx] * softmax_val - 
                            softmax_val * deriv_sum * inv_sum;
        
        values[idx] = softmax_val;
        derivatives[idx] = softmax_deriv;
    }
}

// Differential attention output computation
kernel void differential_attention_output(
    device const float* attention_weights [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* attention_derivs [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* output_derivs [[buffer(4)]],
    constant DiffTransformerParams& params [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.y;
    uint head_idx = pos.x;
    
    if (seq_idx >= params.seq_length || head_idx >= params.hidden_dim) return;
    
    uint head_dim = params.hidden_dim / params.num_heads;
    
    float sum = 0.0;
    float deriv_sum = 0.0;
    
    for (uint k = 0; k < params.seq_length; k++) {
        float attn = attention_weights[seq_idx * params.seq_length + k];
        float attn_deriv = attention_derivs[seq_idx * params.seq_length + k];
        float v = values[k * head_dim + head_idx];
        
        sum += attn * v;
        deriv_sum += attn_deriv * v;
    }
    
    output[seq_idx * head_dim + head_idx] = sum;
    output_derivs[seq_idx * head_dim + head_idx] = deriv_sum;
}
