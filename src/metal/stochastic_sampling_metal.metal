#include <metal_stdlib>
#include "metal_common.h"

// Constants for optimization
constant uint MAX_SAMPLES = 1024;
constant uint WARP_SIZE = 32;

// Stochastic sampling kernel
kernel void stochastic_sampling(
    device const ComplexFloat* input_states [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device ComplexFloat* output_states [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    constant uint& num_samples [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Validate number of samples against MAX_SAMPLES
    if (thread_id >= num_samples || num_samples > MAX_SAMPLES) return;
    
    // Process in WARP_SIZE chunks for better memory access patterns
    uint warp_id = thread_id / WARP_SIZE;
    uint lane_id = thread_id % WARP_SIZE;
    
    // Use warp_id to offset the random seed for better distribution
    float r = fract(sin(float(thread_id + warp_id * WARP_SIZE) * 12.9898) * 43758.5453);
    
    // Cumulative probability selection
    float cumsum = 0.0;
    uint selected_state = 0;
    
    // Process probabilities in WARP_SIZE chunks for coalesced memory access
    for (uint i = 0; i < num_states; i += WARP_SIZE) {
        uint chunk_size = min(WARP_SIZE, num_states - i);
        float local_sum = 0.0;
        
        // Load probability chunk
        if (lane_id < chunk_size) {
            local_sum = probabilities[i + lane_id];
        }
        
        // Accumulate probabilities
        cumsum += local_sum;
        if (r <= cumsum) {
            selected_state = i + lane_id;
            break;
        }
    }
    
    // Copy selected state to output with warp-aligned access
    uint output_idx = warp_id * WARP_SIZE + lane_id;
    if (output_idx < num_samples) {
        output_states[output_idx] = input_states[selected_state];
    }
}

// Quantum stochastic gradient kernel
kernel void stochastic_gradient(
    device const ComplexFloat* states [[buffer(0)]],
    device const ComplexFloat* gradients [[buffer(1)]],
    device ComplexFloat* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Process in WARP_SIZE chunks for better memory access patterns
    uint warp_id = thread_id / WARP_SIZE;
    uint lane_id = thread_id % WARP_SIZE;
    uint batch_id = warp_id * WARP_SIZE + lane_id;
    
    if (batch_id >= batch_size) return;
    
    // Load state and gradient with coalesced memory access
    ComplexFloat state = states[batch_id];
    ComplexFloat gradient = gradients[batch_id];
    
    // Apply stochastic gradient update
    output[batch_id] = {
        state.real - learning_rate * gradient.real,
        state.imag - learning_rate * gradient.imag
    };
}

// Importance sampling kernel
kernel void importance_sampling(
    device const ComplexFloat* states [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    device float* resampled_weights [[buffer(3)]],
    constant uint& num_particles [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Process in WARP_SIZE chunks for better memory access patterns
    uint warp_id = thread_id / WARP_SIZE;
    uint lane_id = thread_id % WARP_SIZE;
    uint particle_id = warp_id * WARP_SIZE + lane_id;
    
    if (particle_id >= num_particles) return;
    
    // Compute cumulative weights in chunks for better memory access
    float cumsum = 0.0;
    for (uint i = 0; i < particle_id; i += WARP_SIZE) {
        uint chunk_size = min(WARP_SIZE, particle_id - i);
        float local_sum = 0.0;
        
        // Load weight chunk
        if (lane_id < chunk_size) {
            local_sum = weights[i + lane_id];
        }
        
        cumsum += local_sum;
    }
    
    // Generate random number with warp-based offset for better distribution
    float r = fract(sin(float(thread_id + warp_id * WARP_SIZE) * 12.9898) * 43758.5453);
    
    // Binary search for resampling with WARP_SIZE stride
    uint left = 0;
    uint right = num_particles - 1;
    
    while (left < right) {
        uint mid = (left + right) / 2;
        float mid_cumsum = 0.0;
        
        // Compute cumsum up to mid in chunks
        for (uint i = 0; i <= mid; i += WARP_SIZE) {
            uint chunk_size = min(WARP_SIZE, mid + 1 - i);
            float local_sum = 0.0;
            
            if (lane_id < chunk_size) {
                local_sum = weights[i + lane_id];
            }
            
            mid_cumsum += local_sum;
        }
        
        if (mid_cumsum < r) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Store resampled index and weight with coalesced memory access
    indices[particle_id] = left;
    resampled_weights[particle_id] = 1.0 / num_particles;
}
