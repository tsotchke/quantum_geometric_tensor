#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability and optimization
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// MNIST-specific constants optimized for M1/M2
constant int MNIST_IMAGE_SIZE = 784;  // 28x28
constant int BATCH_SIZE = 1;   // Match test batch size
constant int NUM_CLASSES = 10;
constant int MAX_QUBITS = 16;  // Increased for better MNIST encoding
constant int THREADS_PER_GROUP = 256;  // Optimized for M1/M2 GPU

// Mathematical constants
constant float PI = 3.14159265359f;
constant float INV_SQRT2 = 0.70710678118f;

// Helper functions for numerical stability
inline bool is_valid_float(float x) {
    return !isnan(x) && !isinf(x) && 
           abs(x) >= MIN_MAGNITUDE && 
           abs(x) <= MAX_MAGNITUDE;
}

inline bool is_valid_float2(float2 v) {
    return !isnan(v.x) && !isinf(v.x) && abs(v.x) <= MAX_MAGNITUDE &&
           !isnan(v.y) && !isinf(v.y) && abs(v.y) <= MAX_MAGNITUDE;
}

inline float2 normalize_magnitude(float2 v, float max_allowed = MAX_MAGNITUDE) {
    float mag = length(v);
    if (mag > max_allowed) {
        return v * (max_allowed / mag);
    }
    return v;
}

// Complex number operations with stability checks
struct Complex {
    float2 val;
    
    Complex() thread : val(float2(0.0f)) {}
    Complex(float r) thread : val(float2(r, 0.0f)) {}
    Complex(float r, float i) thread : val(float2(r, i)) {}
    Complex(float2 v) thread : val(v) {}
    
    bool is_valid() const {
        return is_valid_float2(val);
    }
    
    Complex operator*(thread const Complex& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Complex();
        }
        
        // Scale inputs if needed
        float2 a = normalize_magnitude(val);
        float2 b = normalize_magnitude(other.val);
        
        // Use SIMD operations with FMA
        float2 result = float2(
            fma(-a.y, b.y, a.x * b.x),
            fma(a.x, b.y, a.y * b.x)
        );
        
        // Normalize output
        return Complex(normalize_magnitude(result));
    }
    
    Complex operator+(thread const Complex& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Complex();
        }
        return Complex(normalize_magnitude(val + other.val));
    }
    
    Complex operator-(thread const Complex& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Complex();
        }
        return Complex(normalize_magnitude(val - other.val));
    }
    
    Complex operator*(thread const float scalar) const {
        if (!is_valid() || !is_valid_float(scalar)) {
            return Complex();
        }
        return Complex(normalize_magnitude(val * scalar));
    }
    
    Complex conj() const {
        if (!is_valid()) return Complex();
        return Complex(float2(val.x, -val.y));
    }
    
    float norm() const {
        if (!is_valid()) return 0.0f;
        float n = dot(val, val);  // Use SIMD dot product
        return min(n, MAX_MAGNITUDE);
    }
};

// Enhanced geometric encoding with stability
Complex geometric_phase(float x, float y, uint qubit, float pixel_value) {
    // Validate inputs with error threshold
    if (!is_valid_float(x) || !is_valid_float(y) || 
        !is_valid_float(pixel_value) || qubit >= MAX_QUBITS ||
        abs(x) < ERROR_THRESHOLD || abs(y) < ERROR_THRESHOLD) {
        return Complex();
    }
    
    // Convert to geometric space [-1,1] with bounds checking
    float2 coords = clamp(float2(x, y) * 2.0f - 1.0f, -1.0f, 1.0f);
    float radius = min(length(coords), 1.0f);
    float angle = atan2(coords.y, coords.x);
    
    // Enhanced phase calculation with stability
    float geometric_phase = angle + 2.0f * PI * radius;
    float topology_phase = 2.0f * PI * clamp(
        pixel_value * (1.0f - pow(radius, 2.0f)) +  // Stronger center weights
        0.25f * (1.0f + cos(4.0f * angle)) * pixel_value + // Enhanced angular features
        0.5f * exp(-2.0f * radius * radius) * pixel_value,  // Gaussian radial intensity
        -MAX_MAGNITUDE, MAX_MAGNITUDE
    );
    
    // Qubit-specific phase with stability
    float qubit_scale = exp(-float(qubit) / 4.0f);  // Slower exponential decay
    float final_phase = qubit_scale * (geometric_phase + topology_phase);
    final_phase = clamp(final_phase, -PI, PI);
    
    // Generate quantum state with stability
    float cosval;
    float sinval = sincos(final_phase, cosval);
    float2 result = float2(cosval, sinval);
    return Complex(normalize_magnitude(result));
}

// Helper for quantum state initialization with stability
Complex initial_state(uint qubit) {
    if (qubit >= MAX_QUBITS) return Complex();
    
    float phase = (qubit % 2 == 0) ? 0.0f : PI;
    float cosval;
    float sinval = sincos(phase, cosval);
    float2 result = float2(cosval * INV_SQRT2, sinval * INV_SQRT2);
    return Complex(normalize_magnitude(result));
}

// Kernel for quantum state encoding with enhanced stability
kernel void encode_quantum_state(
    device const float* input [[buffer(0)]],
    device float2* quantum_state [[buffer(1)]],
    device const float* phases [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_threadgroup]])
{
    // Validate thread configuration and thread group size
    if (id >= BATCH_SIZE * MNIST_IMAGE_SIZE || threads != THREADS_PER_GROUP) {
        return;
    }
    
    // Get batch and pixel indices
    uint batch_idx = id / MNIST_IMAGE_SIZE;
    uint pixel_idx = id % MNIST_IMAGE_SIZE;
    
    // Get x,y coordinates with bounds checking
    float x = clamp(float(pixel_idx % 28) / 27.0f, 0.0f, 1.0f);
    float y = clamp(float(pixel_idx / 28) / 27.0f, 0.0f, 1.0f);
    
    // Normalize input pixel value with validation
    float raw_pixel = input[id];
    if (!is_valid_float(raw_pixel) || abs(raw_pixel) < MIN_MAGNITUDE) {
        quantum_state[batch_idx * (1 << MAX_QUBITS) + pixel_idx] = float2(0.0f);
        return;
    }
    float pixel_value = clamp(raw_pixel / 255.0f, 0.0f, 1.0f);
    
    // Initialize quantum state with stability check
    Complex state = initial_state(0);
    if (!state.is_valid()) {
        quantum_state[batch_idx * (1 << MAX_QUBITS) + pixel_idx] = float2(0.0f);
        return;
    }
    
    // Apply geometric encoding for each qubit with stability
    for (uint q = 0; q < MAX_QUBITS; q++) {
        // Get geometric phase factor
        Complex phase_factor = geometric_phase(x, y, q, pixel_value);
        if (!phase_factor.is_valid()) continue;
        
        // Apply phase and mix with initial state
        state = state * phase_factor;
        if (!state.is_valid() || state.norm() < ERROR_THRESHOLD) {
            state = Complex(1.0f, 0.0f);  // Reset to |0⟩ state
            continue;
        }
        
        // Mix with next qubit's initial state if not last qubit
        if (q < MAX_QUBITS - 1) {
            Complex next_state = initial_state(q + 1);
            if (next_state.is_valid()) {
                state = (state + next_state) * INV_SQRT2;
            }
        }
    }
    
    // Store final encoded state with validation
    uint state_idx = batch_idx * (1 << MAX_QUBITS) + pixel_idx;
    quantum_state[state_idx] = state.is_valid() ? state.val : float2(1.0f, 0.0f);
}

// Kernel for quantum geometric transformations with stability
kernel void apply_geometric_transform(
    device float2* quantum_state [[buffer(0)]],
    device const float* parameters [[buffer(1)]],
    device float* metric_tensor [[buffer(2)]],
    device const float* phases [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_threadgroup]])
{
    // Validate thread configuration and thread group size
    uint batch_idx = id / (MAX_QUBITS * MAX_QUBITS);
    uint tensor_idx = id % (MAX_QUBITS * MAX_QUBITS);
    if (batch_idx >= BATCH_SIZE || threads != THREADS_PER_GROUP) {
        return;
    }
    uint i = tensor_idx / MAX_QUBITS;
    uint j = tensor_idx % MAX_QUBITS;
    
    // Validate parameters with error thresholds
    if (!is_valid_float(phases[i]) || !is_valid_float(phases[j]) ||
        !is_valid_float(parameters[i]) || !is_valid_float(parameters[j]) ||
        abs(phases[i]) < ERROR_THRESHOLD || abs(phases[j]) < ERROR_THRESHOLD ||
        abs(parameters[i]) < MIN_MAGNITUDE || abs(parameters[j]) < MIN_MAGNITUDE) {
        metric_tensor[batch_idx * MAX_QUBITS * MAX_QUBITS + tensor_idx] = 0.0f;
        return;
    }
    
    // Load quantum state with validation
    uint state_idx = batch_idx * (1 << MAX_QUBITS);
    Complex state(quantum_state[state_idx]);
    if (!state.is_valid() || state.norm() < ERROR_THRESHOLD) {
        metric_tensor[batch_idx * MAX_QUBITS * MAX_QUBITS + tensor_idx] = 0.0f;
        return;
    }
    
    // Apply geometric transformations with stability
    float phase_i = clamp(phases[i], -PI, PI);
    float phase_j = clamp(phases[j], -PI, PI);
    float param_i = clamp(parameters[i], -MAX_MAGNITUDE, MAX_MAGNITUDE);
    float param_j = clamp(parameters[j], -MAX_MAGNITUDE, MAX_MAGNITUDE);
    
    // Compute derivatives with stability
    Complex d_i = state * Complex(float2(
        -sin(phase_i * param_i),
        cos(phase_i * param_i)
    ));
    Complex d_j = state * Complex(float2(
        -sin(phase_j * param_j),
        cos(phase_j * param_j)
    ));
    
    if (!d_i.is_valid() || !d_j.is_valid() || 
        d_i.norm() < ERROR_THRESHOLD || d_j.norm() < ERROR_THRESHOLD) {
        metric_tensor[batch_idx * MAX_QUBITS * MAX_QUBITS + tensor_idx] = 0.0f;
        return;
    }
    
    // Compute metric tensor element with stability
    Complex overlap = d_i.conj() * d_j;
    float metric_element = overlap.is_valid() ? 
        clamp(overlap.norm(), MIN_MAGNITUDE, MAX_MAGNITUDE) : 0.0f;
    
    metric_tensor[batch_idx * MAX_QUBITS * MAX_QUBITS + tensor_idx] = metric_element;
}

// Kernel for quantum measurements with enhanced stability
kernel void measure_quantum_state(
    device const float2* quantum_state [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_threadgroup]])
{
    // Validate thread configuration and thread group size
    if (id >= BATCH_SIZE * NUM_CLASSES || threads != THREADS_PER_GROUP) {
        return;
    }
    
    // Get batch and class indices
    uint batch_idx = id / NUM_CLASSES;
    uint class_idx = id % NUM_CLASSES;
    
    // Compute measurement probability with stability
    float prob = 0.0f;
    float max_term = 0.0f;
    
    uint start_idx = batch_idx * (1 << MAX_QUBITS) + 
                    class_idx * ((1 << MAX_QUBITS) / NUM_CLASSES);
    uint end_idx = start_idx + ((1 << MAX_QUBITS) / NUM_CLASSES);
    
    // First pass: find maximum term
    for (uint i = start_idx; i < end_idx; i++) {
        float2 amp = quantum_state[i];
        if (!is_valid_float2(amp)) continue;
        
        float term = dot(amp, amp);
        max_term = max(max_term, term);
    }
    
    // Second pass: accumulate with scaling
    if (max_term > MIN_MAGNITUDE && max_term < MAX_MAGNITUDE) {
        float scale = min(1.0f, MAX_MAGNITUDE / max_term);
        
        for (uint i = start_idx; i < end_idx; i++) {
            float2 amp = quantum_state[i];
            if (!is_valid_float2(amp)) continue;
            
            prob += dot(amp * scale, amp * scale);
        }
        
        // Rescale result
        prob /= (scale * scale);
    }
    
    // Apply geometric normalization with bounds
    float norm_factor = (1 << MAX_QUBITS) / NUM_CLASSES;
    prob = clamp(prob / norm_factor, MIN_MAGNITUDE, 1.0f);
    
    probabilities[id] = prob;
}
