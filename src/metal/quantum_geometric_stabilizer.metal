#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Helper functions for numerical stability
inline bool is_valid_float2(float2 v) {
    return !any(isnan(v)) && !any(isinf(v)) && all(abs(v) <= float2(MAX_MAGNITUDE));
}

inline float2 normalize_amplitude(float2 amplitude) {
    float mag = length(amplitude);
    if (mag > MAX_MAGNITUDE || mag < MIN_MAGNITUDE) {
        return amplitude * (MAX_MAGNITUDE / mag);
    }
    return amplitude;
}

inline float2 apply_pauli_x(float2 amplitude) {
    float2 result = float2(amplitude.y, amplitude.x);
    return normalize_amplitude(result);
}

inline float2 apply_pauli_y(float2 amplitude) {
    float2 result = float2(-amplitude.y, amplitude.x);
    return normalize_amplitude(result);
}

inline float2 apply_pauli_z(float2 amplitude) {
    float2 result = float2(amplitude.x, -amplitude.y);
    return normalize_amplitude(result);
}

// Error tracking and mitigation
inline float calculate_error_contribution(float base_error, uint operation_type) {
    switch (operation_type) {
        case 0: return base_error;           // X gate
        case 1: return base_error * 1.5f;    // Y gate (higher error)
        case 2: return base_error;           // Z gate
        default: return base_error;
    }
}

inline float2 apply_error_mitigation(float2 state, float error_rate, float threshold) {
    if (error_rate > threshold) {
        float scale = 1.0f - (error_rate - threshold);
        return state * scale;
    }
    return state;
}

// Stabilizer types and structures
typedef struct {
    float2 amplitude;  // Complex amplitude (real, imag)
    float error_rate;  // Error rate for this qubit
    uint flags;        // Status flags
} StabilizerQubit;

typedef struct {
    uint type;         // X, Y, or Z stabilizer
    uint num_qubits;   // Number of qubits in stabilizer
    float weight;      // Stabilizer weight
    float confidence;  // Measurement confidence
} StabilizerConfig;

// Stabilizer measurement kernel
kernel void measure_stabilizer(
    device const StabilizerQubit* qubits [[buffer(0)]],
    device const uint* qubit_indices [[buffer(1)]],
    device const StabilizerConfig& config [[buffer(2)]],
    device float2* result [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= config.num_qubits) return;
    
    // Load and validate qubit data
    uint qubit_idx = qubit_indices[tid];
    StabilizerQubit qubit = qubits[qubit_idx];
    
    if (!is_valid_float2(qubit.amplitude)) {
        result[tid] = float2(0.0f);
        return;
    }
    
    // Initialize accumulator with error tracking
    float2 measurement = float2(0.0f);
    float error_accumulator = 0.0f;
    qubit.amplitude = normalize_amplitude(qubit.amplitude);
    
    // Apply stabilizer operation based on type
    switch (config.type) {
        case 0: { // X stabilizer
            // Apply Pauli X with error tracking
            measurement.x = qubit.amplitude.y;
            measurement.y = qubit.amplitude.x;
            error_accumulator = qubit.error_rate;
            break;
        }
        case 1: { // Y stabilizer
            // Apply Pauli Y with error tracking
            measurement.x = -qubit.amplitude.y;
            measurement.y = qubit.amplitude.x;
            error_accumulator = qubit.error_rate * 1.5f; // Y gates have higher error
            break;
        }
        case 2: { // Z stabilizer
            // Apply Pauli Z with error tracking
            measurement = qubit.amplitude;
            measurement.y = -measurement.y;
            error_accumulator = qubit.error_rate;
            break;
        }
    }
    
    // Apply error mitigation
    if (error_accumulator > ERROR_THRESHOLD) {
        float scale = 1.0f - error_accumulator;
        measurement *= scale;
    }
    
    // Store result with confidence and stability check
    measurement = normalize_amplitude(measurement);
    result[tid] = measurement * config.confidence;
}

// Error correction kernel
kernel void apply_correction(
    device StabilizerQubit* qubits [[buffer(0)]],
    device const uint* qubit_indices [[buffer(1)]],
    device const StabilizerConfig& config [[buffer(2)]],
    device const float2* syndrome [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= config.num_qubits) return;
    
    // Load and validate qubit data
    uint qubit_idx = qubit_indices[tid];
    StabilizerQubit qubit = qubits[qubit_idx];
    
    if (!is_valid_float2(qubit.amplitude)) {
        return;
    }
    
    // Load and validate syndrome measurement
    float2 correction = syndrome[tid];
    if (!is_valid_float2(correction)) {
        return;
    }
    float magnitude = length(correction);
    
    // Skip if correction is too small
    if (magnitude < ERROR_THRESHOLD) return;
    
    // Apply correction based on stabilizer type
    switch (config.type) {
        case 0: { // X correction
            float2 temp = qubit.amplitude;
            qubit.amplitude.x = temp.y * correction.x - temp.x * correction.y;
            qubit.amplitude.y = temp.y * correction.y + temp.x * correction.x;
            break;
        }
        case 1: { // Y correction
            float2 temp = qubit.amplitude;
            qubit.amplitude.x = -temp.y * correction.x - temp.x * correction.y;
            qubit.amplitude.y = temp.x * correction.x - temp.y * correction.y;
            break;
        }
        case 2: { // Z correction
            qubit.amplitude.y = -qubit.amplitude.y;
            break;
        }
    }
    
    // Update error rate
    qubit.error_rate = max(0.0f, qubit.error_rate - magnitude * config.weight);
    
    // Normalize and store updated qubit
    qubit.amplitude = normalize_amplitude(qubit.amplitude);
    qubits[qubit_idx] = qubit;
}

// Stabilizer correlation kernel
kernel void compute_correlations(
    device const StabilizerQubit* qubits [[buffer(0)]],
    device const uint* qubit_indices [[buffer(1)]],
    device const StabilizerConfig* configs [[buffer(2)]],
    device float* correlations [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint stabilizer1 = tid.x;
    uint stabilizer2 = tid.y;
    
    if (stabilizer1 >= configs[0].num_qubits || 
        stabilizer2 >= configs[0].num_qubits) return;
    
    // Skip self-correlation
    if (stabilizer1 == stabilizer2) {
        correlations[stabilizer1 * configs[0].num_qubits + stabilizer2] = 1.0f;
        return;
    }
    
    // Load stabilizer configs
    StabilizerConfig config1 = configs[stabilizer1];
    StabilizerConfig config2 = configs[stabilizer2];
    
    // Initialize correlation calculation
    float correlation = 0.0f;
    float total_weight = 0.0f;
    
    // Calculate correlation between stabilizers with stability checks
    for (uint i = 0; i < config1.num_qubits; i++) {
        uint idx1 = qubit_indices[stabilizer1 * config1.num_qubits + i];
        StabilizerQubit qubit1 = qubits[idx1];
        
        if (!is_valid_float2(qubit1.amplitude)) {
            continue;
        }
        
        for (uint j = 0; j < config2.num_qubits; j++) {
            uint idx2 = qubit_indices[stabilizer2 * config2.num_qubits + j];
            if (idx1 == idx2) {
                StabilizerQubit qubit2 = qubits[idx2];
                
                if (!is_valid_float2(qubit2.amplitude)) {
                    continue;
                }
                
                // Normalize amplitudes before correlation
                float2 amp1 = normalize_amplitude(qubit1.amplitude);
                float2 amp2 = normalize_amplitude(qubit2.amplitude);
                
                // Calculate correlation contribution
                float weight = (1.0f - qubit1.error_rate) * (1.0f - qubit2.error_rate);
                correlation += weight * dot(amp1, amp2);
                total_weight += weight;
            }
        }
    }
    
    // Normalize correlation
    if (total_weight > 0.0f) {
        correlation /= total_weight;
    }
    
    // Store correlation with bounds check
    correlation = clamp(correlation, -MAX_MAGNITUDE, MAX_MAGNITUDE);
    correlations[stabilizer1 * configs[0].num_qubits + stabilizer2] = correlation;
    correlations[stabilizer2 * configs[0].num_qubits + stabilizer1] = correlation;
}
