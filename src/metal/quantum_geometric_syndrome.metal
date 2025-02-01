#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability and optimization
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint VECTOR_SIZE = 4;  // Size of vectors/matrices
constant uint TILE_SIZE = 2;    // Process in 2x2 tiles for 4x4 matrices

// Helper functions for numerical stability
inline bool is_valid_float2(float2 v) {
    return !any(isnan(v)) && !any(isinf(v)) && all(abs(v) <= float2(MAX_MAGNITUDE));
}

inline float2 normalize_amplitude(float2 amplitude) {
    float mag = length(amplitude);
    if (mag > MAX_MAGNITUDE) {
        return amplitude * (MAX_MAGNITUDE / mag);
    }
    return amplitude;
}

struct SyndromeVertex {
    uint x;
    uint y;
    uint z;
    float weight;
    bool is_boundary;
    uint timestamp;
    float confidence;
    float correlation_weight;
    bool part_of_chain;
};

struct SyndromeConfig {
    float detection_threshold;
    float confidence_threshold;
    float weight_scale_factor;
    float pattern_threshold;
    uint parallel_group_size;
    uint min_pattern_occurrences;
    bool enable_parallel;
    bool use_boundary_matching;
};

kernel void extract_syndromes(
    device const float2* state_amplitudes [[buffer(0)]],
    device const uint* state_indices [[buffer(1)]],
    device const SyndromeConfig& config [[buffer(2)]],
    device SyndromeVertex* vertices [[buffer(3)]],
    uint vertex_id [[thread_position_in_grid]]
) {
    // Get dimensions from state indices
    uint width = state_indices[0];
    uint height = state_indices[1];
    uint depth = state_indices[2];
    
    // Calculate x,y,z coordinates
    uint x = vertex_id % width;
    uint y = (vertex_id / width) % height;
    uint z = vertex_id / (width * height);
    
    if (z >= depth) return;
    
    // Initialize stabilizer measurements
    float x_stabilizer = 0.0f;
    float z_stabilizer = 0.0f;
    bool valid_measurement = true;
    
    // X stabilizer measurement with vectorized validation
    float4 x_stabilizers = 0.0f;
    for (uint tile = 0; tile < VECTOR_SIZE; tile += TILE_SIZE) {
        // Process stabilizers in tiles for better cache utilization
        for (uint i = tile; i < min(tile + TILE_SIZE, VECTOR_SIZE); i++) {
            uint idx = state_indices[x + y*width + z*width*height + i*width*height*depth];
            float2 amp = state_amplitudes[idx];
            
            if (!is_valid_float2(amp)) {
                valid_measurement = false;
                break;
            }
            
            amp = normalize_amplitude(amp);
            x_stabilizers[i] = amp.x * amp.x + amp.y * amp.y;
        }
        if (!valid_measurement) break;
    }
    
    // Sum up stabilizer contributions
    if (valid_measurement) {
        x_stabilizer = x_stabilizers[0] + x_stabilizers[1] + x_stabilizers[2] + x_stabilizers[3];
    }
    
    // Z stabilizer measurement with vectorized validation
    float4 z_stabilizers = 0.0f;
    if (valid_measurement) {
        for (uint tile = 0; tile < VECTOR_SIZE; tile += TILE_SIZE) {
            // Process stabilizers in tiles for better cache utilization
            for (uint i = tile; i < min(tile + TILE_SIZE, VECTOR_SIZE); i++) {
                uint idx = state_indices[x + y*width + z*width*height + (i+4)*width*height*depth];
                float2 amp = state_amplitudes[idx];
                
                if (!is_valid_float2(amp)) {
                    valid_measurement = false;
                    break;
                }
                
                amp = normalize_amplitude(amp);
                z_stabilizers[i] = amp.x * amp.x - amp.y * amp.y;
            }
            if (!valid_measurement) break;
        }
        
        // Sum up stabilizer contributions
        if (valid_measurement) {
            z_stabilizer = z_stabilizers[0] + z_stabilizers[1] + z_stabilizers[2] + z_stabilizers[3];
        }
    }
    
    // Calculate total weight with stability check
    float weight = 0.0f;
    if (valid_measurement) {
        x_stabilizer = clamp(x_stabilizer, -MAX_MAGNITUDE, MAX_MAGNITUDE);
        z_stabilizer = clamp(z_stabilizer, -MAX_MAGNITUDE, MAX_MAGNITUDE);
        weight = sqrt(x_stabilizer * x_stabilizer + z_stabilizer * z_stabilizer);
        weight = min(weight, MAX_MAGNITUDE);
        
        // Filter out noise below MIN_MAGNITUDE
        if (weight < MIN_MAGNITUDE) {
            weight = 0.0f;
        }
        
        // Check for measurement errors
        if (abs(x_stabilizer) < ERROR_THRESHOLD || abs(z_stabilizer) < ERROR_THRESHOLD) {
            valid_measurement = false;
            weight = 0.0f;
        }
    }
    
    // Check if this is a syndrome
    if (weight > config.detection_threshold) {
        vertices[vertex_id].x = x;
        vertices[vertex_id].y = y;
        vertices[vertex_id].z = z;
        vertices[vertex_id].weight = weight;
        vertices[vertex_id].is_boundary = (x == 0 || y == 0 || z == 0);
        vertices[vertex_id].timestamp = vertex_id;
        vertices[vertex_id].confidence = 1.0f;
        vertices[vertex_id].correlation_weight = 0.0f;
        vertices[vertex_id].part_of_chain = false;
    }
}

kernel void compute_syndrome_correlations(
    device const SyndromeVertex* vertices [[buffer(0)]],
    device const SyndromeConfig& config [[buffer(1)]],
    device float* correlation_matrix [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint i = pos.x;
    uint j = pos.y;
    
    if (i >= j) return; // Only compute upper triangle
    
    // Skip if either vertex is not a syndrome
    if (vertices[i].weight <= config.detection_threshold || 
        vertices[j].weight <= config.detection_threshold) {
        correlation_matrix[i * config.parallel_group_size + j] = 0.0f;
        correlation_matrix[j * config.parallel_group_size + i] = 0.0f;
        return;
    }
    
    // Calculate spatial correlation with bounds checking
    float dx = clamp(float(vertices[i].x) - float(vertices[j].x), -MAX_MAGNITUDE, MAX_MAGNITUDE);
    float dy = clamp(float(vertices[i].y) - float(vertices[j].y), -MAX_MAGNITUDE, MAX_MAGNITUDE);
    float dz = clamp(float(vertices[i].z) - float(vertices[j].z), -MAX_MAGNITUDE, MAX_MAGNITUDE);
    float distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    // Calculate spatial correlation with error threshold
    float spatial_correlation = 1.0f / (1.0f + distance);
    spatial_correlation = clamp(spatial_correlation, 0.0f, 1.0f);
    
    // Filter out weak correlations
    if (spatial_correlation < ERROR_THRESHOLD) {
        spatial_correlation = 0.0f;
    }
    
    // Ensure minimum correlation strength
    if (spatial_correlation < MIN_MAGNITUDE) {
        spatial_correlation = 0.0f;
    }
    
    // Calculate temporal correlation with stability
    float temporal_correlation = 0.0f;
    float time_diff;
    if (vertices[i].timestamp > vertices[j].timestamp) {
        time_diff = min(float(vertices[i].timestamp - vertices[j].timestamp), MAX_MAGNITUDE);
    } else {
        time_diff = min(float(vertices[j].timestamp - vertices[i].timestamp), MAX_MAGNITUDE);
    }
    temporal_correlation = exp(-time_diff * 0.1f);
    temporal_correlation = clamp(temporal_correlation, 0.0f, 1.0f);
    
    // Combine correlations with stability
    float correlation = clamp(0.7f * spatial_correlation + 0.3f * temporal_correlation, 0.0f, 1.0f);
    
    // Store in correlation matrix (symmetric)
    correlation_matrix[i * config.parallel_group_size + j] = correlation;
    correlation_matrix[j * config.parallel_group_size + i] = correlation;
}
