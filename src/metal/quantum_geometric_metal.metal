#include <metal_stdlib>
using namespace metal;

// Constants for optimal performance on M1/M2
constant uint WORKGROUP_SIZE = 32;  // Used for thread group and shared memory sizing
constant uint MAX_RANK = 32;
constant float TOLERANCE = 1e-6f;
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;

// Complex number operations
struct complex_t {
    float2 value;
};

// Hierarchical matrix structure
struct HierarchicalBlock {
    uint row_start;
    uint row_end;
    uint col_start;
    uint col_end;
    uint rank;
    bool is_low_rank;
};

// Helper functions for numerical stability
inline bool is_valid_float(float x) {
    float abs_x = abs(x);
    return !isnan(x) && !isinf(x) && 
           (abs_x == 0.0f || (abs_x >= MIN_MAGNITUDE && abs_x <= MAX_MAGNITUDE));
}

inline bool is_valid_float2(float2 v) {
    return is_valid_float(v.x) && is_valid_float(v.y);
}

inline float safe_length(float2 v) {
    if (!is_valid_float2(v)) return 0.0f;
    return length(v);
}

inline float2 normalize_magnitude(float2 v, float max_allowed = MAX_MAGNITUDE) {
    float mag = safe_length(v);
    if (mag > max_allowed) {
        return v * (max_allowed / mag);
    }
    if (mag > 0.0f && mag < MIN_MAGNITUDE) {
        return v * (MIN_MAGNITUDE / mag);
    }
    return v;
}

// Complex number operations with stability checks
inline complex_t complex_mul(complex_t a, complex_t b) {
    if (!is_valid_float2(a.value) || !is_valid_float2(b.value)) {
        return complex_t{float2(0.0f)};
    }
    
    // Normalize inputs if needed
    float2 a_norm = normalize_magnitude(a.value);
    float2 b_norm = normalize_magnitude(b.value);
    
    float2 result = float2(
        a_norm.x * b_norm.x - a_norm.y * b_norm.y,
        a_norm.x * b_norm.y + a_norm.y * b_norm.x
    );
    
    // Normalize output
    return complex_t{normalize_magnitude(result)};
}

inline complex_t complex_add(complex_t a, complex_t b) {
    if (!is_valid_float2(a.value) || !is_valid_float2(b.value)) {
        return complex_t{float2(0.0f)};
    }
    return complex_t{normalize_magnitude(a.value + b.value)};
}

// Helper function for Jacobi rotation with stability
void jacobi_rotate(
    complex_t g,
    complex_t h,
    complex_t a,
    thread float& c,
    thread float& s
) {
    float gx = g.value.x;
    float hx = h.value.x;
    float ax = a.value.x;
    float ay = a.value.y;
    
    if (!is_valid_float(gx) || !is_valid_float(hx) || 
        !is_valid_float(ax) || !is_valid_float(ay) ||
        abs(ax) + abs(ay) < TOLERANCE) {
        c = 1.0f;
        s = 0.0f;
        return;
    }
    
    // Compute with numerical stability
    float denom = 2.0f * sqrt(ax * ax + ay * ay);
    if (abs(denom) < TOLERANCE) {
        c = 1.0f;
        s = 0.0f;
        return;
    }
    
    float tau = (hx - gx) / denom;
    float t = 1.0f / (abs(tau) + sqrt(1.0f + tau * tau));
    if (tau < 0.0f) t = -t;
    
    // Normalize results
    c = 1.0f / sqrt(1.0f + t * t);
    s = t * c;
    
    c = clamp(c, -MAX_MAGNITUDE, MAX_MAGNITUDE);
    s = clamp(s, -MAX_MAGNITUDE, MAX_MAGNITUDE);
}

// Helper function for QR decomposition with stability
kernel void householder_qr(
    device const complex_t* A [[buffer(0)]],
    device complex_t* Q [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // Early bounds check
    if (lid.y >= rows || lid.x >= cols) return;
    
    threadgroup complex_t shared_A[WORKGROUP_SIZE][WORKGROUP_SIZE];
    threadgroup atomic_uint max_magnitude;
    threadgroup float convergence_error[1];  // Array for proper alignment
    
    // Initialize atomic and convergence tracking
    if (lid.x == 0 && lid.y == 0) {
        atomic_store_explicit(&max_magnitude, 0u, memory_order_relaxed);
        convergence_error[0] = INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load and validate data
    if (lid.y < rows && lid.x < cols) {
        complex_t val = A[lid.y * cols + lid.x];
        shared_A[lid.y][lid.x] = is_valid_float2(val.value) ? val : complex_t{float2(0.0f)};
        
        // Track maximum magnitude
        if (lid.x == 0) {
            float mag = safe_length(val.value);
            uint mag_bits = as_type<uint>(mag);
            atomic_fetch_max_explicit(&max_magnitude, mag_bits, memory_order_relaxed);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scale data if needed
    float max_magnitude_val = as_type<float>(atomic_load_explicit(&max_magnitude, memory_order_relaxed));
    if (max_magnitude_val > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude_val;
        if (lid.y < rows && lid.x < cols) {
            shared_A[lid.y][lid.x].value *= scale;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Implement blocked Householder QR with enhanced stability
    uint max_iterations = min(rows, cols);
    for (uint iter = 0; iter < max_iterations && convergence_error[0] > TOLERANCE; iter++) {
        if (lid.y >= iter && lid.x == 0) {
            // Compute Householder vector with stability
            float norm = 0.0f;
            float max_component = 0.0f;
            
            // Two-pass normalization for better stability
            for (uint i = iter; i < rows; i++) {
                float2 val = shared_A[i][iter].value;
                if (is_valid_float2(val)) {
                    max_component = max(max_component, abs(val.x));
                    max_component = max(max_component, abs(val.y));
                }
            }
            
            if (max_component > TOLERANCE) {
                float scale = 1.0f / max_component;
                for (uint i = iter; i < rows; i++) {
                    float2 val = shared_A[i][iter].value * scale;
                    norm += val.x * val.x + val.y * val.y;
                }
                norm = sqrt(norm) * max_component;
            }
            
            // Update column with stability
            complex_t alpha = shared_A[iter][iter];
            if (is_valid_float2(alpha.value) && norm > TOLERANCE) {
                float sign = (alpha.value.x > 0.0f) ? 1.0f : -1.0f;
                shared_A[iter][iter].value.x += sign * norm;
                
                // Safe normalization
                float denom = sqrt(norm * (norm + abs(alpha.value.x)));
                float scale = (denom > TOLERANCE) ? (1.0f / denom) : 0.0f;
                scale = min(scale, MAX_MAGNITUDE);
                
                for (uint i = iter; i < rows; i++) {
                    shared_A[i][iter].value *= scale;
                }
                
                // Update convergence error
                if (lid.y == iter) {
                    convergence_error[0] = abs(norm - abs(alpha.value.x));
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply Householder reflection with stability
        if (lid.y >= iter && lid.x > iter) {
            complex_t dot = complex_t{float2(0.0f)};
            float max_term = 0.0f;
            
            // Two-pass summation for stability
            for (uint i = iter; i < rows; i++) {
                complex_t term = complex_mul(shared_A[i][iter], shared_A[i][lid.x]);
                max_term = max(max_term, safe_length(term.value));
                dot = complex_add(dot, term);
            }
            
            // Apply reflection with magnitude check
            if (max_term > TOLERANCE && max_term < MAX_MAGNITUDE) {
                for (uint i = iter; i < rows; i++) {
                    shared_A[i][lid.x] = complex_add(
                        shared_A[i][lid.x],
                        complex_mul(complex_t{float2(-2.0f, 0.0f)}, 
                            complex_mul(shared_A[i][iter], dot))
                    );
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Form Q explicitly with final stability check
    if (lid.y < rows && lid.x < cols) {
        Q[lid.y * cols + lid.x] = complex_t{
            normalize_magnitude(shared_A[lid.y][lid.x].value)
        };
    }
}

// Helper function for small matrix SVD with enhanced stability
kernel void svd_small_matrix(
    device complex_t* U [[buffer(0)]],
    device float* S [[buffer(1)]],
    device complex_t* V [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    if (lid.x >= size || lid.y >= size) return;
    
    threadgroup complex_t work[WORKGROUP_SIZE][WORKGROUP_SIZE];
    threadgroup float diag[WORKGROUP_SIZE];
    threadgroup atomic_uint max_error;
    threadgroup bool valid_computation;
    
    // Initialize shared memory
    if (lid.x == 0 && lid.y == 0) {
        atomic_store_explicit(&max_error, 0u, memory_order_relaxed);
        valid_computation = true;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load and validate input
    work[lid.y][lid.x] = U[lid.y * size + lid.x];
    if (!is_valid_float2(work[lid.y][lid.x].value)) {
        valid_computation = false;
        work[lid.y][lid.x] = complex_t{float2(0.0f)};
    }
    
    if (lid.x == lid.y) {
        float mag = safe_length(work[lid.x][lid.x].value);
        diag[lid.x] = mag;
        uint mag_bits = as_type<uint>(mag);
        atomic_fetch_max_explicit(&max_error, mag_bits, memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scale matrix if needed
    float max_error_val = as_type<float>(atomic_load_explicit(&max_error, memory_order_relaxed));
    if (max_error_val > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_error_val;
        work[lid.y][lid.x].value *= scale;
        if (lid.x == lid.y) {
            diag[lid.x] *= scale;
        }
    }
    
    // Jacobi iterations with stability checks
    if (valid_computation) {
        for (uint sweep = 0; sweep < 30 && max_error_val > TOLERANCE; sweep++) {
            float sweep_error = 0.0f;
            
            for (uint i = 0; i < size-1; i++) {
                for (uint j = i+1; j < size; j++) {
                    // 2x2 SVD with stability
                    complex_t g = work[i][i];
                    complex_t h = work[j][j];
                    complex_t a = work[i][j];
                    
                    float c, s;
                    jacobi_rotate(g, h, a, c, s);
                    
                    // Apply rotation with bounds checking
                    if (is_valid_float(c) && is_valid_float(s)) {
                        for (uint k = 0; k < size; k++) {
                            complex_t temp1 = work[i][k];
                            complex_t temp2 = work[j][k];
                            
                            work[i][k].value = normalize_magnitude(float2(
                                c * temp1.value.x + s * temp2.value.x,
                                c * temp1.value.y + s * temp2.value.y
                            ));
                            
                            work[j][k].value = normalize_magnitude(float2(
                                -s * temp1.value.x + c * temp2.value.x,
                                -s * temp1.value.y + c * temp2.value.y
                            ));
                        }
                        
                        // Update diagonal and error tracking
                        float new_diag_i = safe_length(work[i][i].value);
                        float new_diag_j = safe_length(work[j][j].value);
                        
                        sweep_error = max(sweep_error,
                            abs(new_diag_i - diag[i]) +
                            abs(new_diag_j - diag[j])
                        );
                        
                        diag[i] = new_diag_i;
                        diag[j] = new_diag_j;
                    }
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (lid.x == 0 && lid.y == 0) {
                max_error_val = sweep_error;
                atomic_store_explicit(&max_error, as_type<uint>(sweep_error), memory_order_relaxed);
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Store results with final stability check
    if (lid.x == lid.y) {
        S[lid.x] = clamp(diag[lid.x], 0.0f, MAX_MAGNITUDE);
    }
    
    U[lid.y * size + lid.x] = complex_t{
        normalize_magnitude(work[lid.y][lid.x].value)
    };
}

// Fast hierarchical matrix multiplication with enhanced stability
kernel void hierarchical_multiply(
    device const complex_t* a_data [[buffer(0)]],
    device const complex_t* b_data [[buffer(1)]],
    device complex_t* c_data [[buffer(2)]],
    device const HierarchicalBlock* blocks [[buffer(3)]],
    device const float* geometric_distances [[buffer(4)]],
    constant uint& num_blocks [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 grid_size [[threads_per_grid]]
) {
    // Shared memory for block multiplication
    threadgroup complex_t shared_a[WORKGROUP_SIZE][WORKGROUP_SIZE];
    threadgroup complex_t shared_b[WORKGROUP_SIZE][WORKGROUP_SIZE];
    threadgroup atomic_uint block_max_magnitude;
    threadgroup bool block_valid;
    
    // Process blocks in parallel with stability
    for (uint block_idx = gid.x; block_idx < num_blocks; block_idx += grid_size.x) {
        HierarchicalBlock block = blocks[block_idx];
        
        if (lid.x == 0 && lid.y == 0) {
            atomic_store_explicit(&block_max_magnitude, 0u, memory_order_relaxed);
            block_valid = true;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Check geometric distance for pruning
        if (geometric_distances[block_idx] > TOLERANCE) {
            // Far field: use low-rank approximation with stability
            if (lid.x == 0 && lid.y == 0) {
                uint rows = block.row_end - block.row_start;
                uint cols = block.col_end - block.col_start;
                uint rank = min(block.rank, MAX_RANK);
                
                // Two-pass multiplication for stability
                for (uint i = 0; i < rows; i++) {
                    for (uint j = 0; j < cols; j++) {
                        // First pass: find maximum term
                        float max_term = 0.0f;
                        for (uint k = 0; k < rank; k++) {
                            complex_t term = complex_mul(
                                a_data[(block.row_start + i) * rank + k],
                                b_data[(block.col_start + j) * rank + k]
                            );
                            max_term = max(max_term, safe_length(term.value));
                        }
                        
                        // Second pass: scaled accumulation
                        complex_t sum = complex_t{float2(0.0f)};
                        if (max_term > TOLERANCE && max_term < MAX_MAGNITUDE) {
                            float scale = min(1.0f, MAX_MAGNITUDE / max_term);
                            for (uint k = 0; k < rank; k++) {
                                complex_t term = complex_mul(
                                    a_data[(block.row_start + i) * rank + k],
                                    b_data[(block.col_start + j) * rank + k]
                                );
                                sum = complex_add(sum, complex_t{term.value * scale});
                            }
                            sum.value /= scale;  // Rescale result
                        }
                        
                        c_data[(block.row_start + i) * cols + j] = sum;
                    }
                }
            }
        } else {
            // Near field: full multiplication with shared memory
            uint rows = block.row_end - block.row_start;
            uint cols = block.col_end - block.col_start;
            
            // Load blocks with validation
            for (uint i = lid.y; i < rows; i += WORKGROUP_SIZE) {
                for (uint j = lid.x; j < cols; j += WORKGROUP_SIZE) {
                    complex_t a_val = a_data[(block.row_start + i) * cols + j];
                    complex_t b_val = b_data[i * cols + (block.col_start + j)];
                    
                    if (!is_valid_float2(a_val.value) || !is_valid_float2(b_val.value)) {
                        block_valid = false;
                        shared_a[i][j] = complex_t{float2(0.0f)};
                        shared_b[i][j] = complex_t{float2(0.0f)};
                    } else {
                        shared_a[i][j] = a_val;
                        shared_b[i][j] = b_val;
                        
                        float mag = max(safe_length(a_val.value), 
                                      safe_length(b_val.value));
                        uint mag_bits = as_type<uint>(mag);
                        atomic_fetch_max_explicit(&block_max_magnitude, 
                                                mag_bits,
                                                memory_order_relaxed);
                    }
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Scale if needed
            float block_max_magnitude_val = as_type<float>(atomic_load_explicit(&block_max_magnitude, memory_order_relaxed));
            if (block_max_magnitude_val > MAX_MAGNITUDE) {
                float scale = MAX_MAGNITUDE / block_max_magnitude_val;
                for (uint i = lid.y; i < rows; i += WORKGROUP_SIZE) {
                    for (uint j = lid.x; j < cols; j += WORKGROUP_SIZE) {
                        shared_a[i][j].value *= scale;
                        shared_b[i][j].value *= scale;
                    }
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute block product with stability
            if (block_valid && lid.y < rows && lid.x < cols) {
                // Two-pass multiplication for stability
                float max_term = 0.0f;
                for (uint k = 0; k < cols; k++) {
                    complex_t term = complex_mul(
                        shared_a[lid.y][k],
                        shared_b[k][lid.x]
                    );
                    max_term = max(max_term, safe_length(term.value));
                }
                
                complex_t sum = complex_t{float2(0.0f)};
                if (max_term > TOLERANCE && max_term < MAX_MAGNITUDE) {
                    float scale = min(1.0f, MAX_MAGNITUDE / max_term);
                    for (uint k = 0; k < cols; k++) {
                        complex_t term = complex_mul(
                            shared_a[lid.y][k],
                            shared_b[k][lid.x]
                        );
                        sum = complex_add(sum, complex_t{term.value * scale});
                    }
                    sum.value /= scale;  // Rescale result
                }
                
                c_data[(block.row_start + lid.y) * cols + 
                       (block.col_start + lid.x)] = sum;
            }
        }
    }
}

// MNIST specific kernel for quantum geometric processing
kernel void process_mnist_data(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& output_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (tid >= output_size) return;
    
    // Use shared memory for better performance
    threadgroup float shared_input[784];  // 28x28 MNIST image
    threadgroup float shared_weights[784];
    threadgroup atomic_uint max_magnitude;
    threadgroup bool valid_computation;
    
    // Initialize atomic
    if (lid == 0) {
        atomic_store_explicit(&max_magnitude, 0u, memory_order_relaxed);
        valid_computation = true;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load data with validation
    for (uint i = lid; i < input_size && i < 784; i += WORKGROUP_SIZE) {
        float in_val = input[i];
        float weight_val = weights[tid * input_size + i];
        
        if (!is_valid_float(in_val) || !is_valid_float(weight_val)) {
            valid_computation = false;
            shared_input[i] = 0.0f;
            shared_weights[i] = 0.0f;
        } else {
            shared_input[i] = in_val;
            shared_weights[i] = weight_val;
            
            float mag = max(abs(in_val), abs(weight_val));
            uint mag_bits = as_type<uint>(mag);
            atomic_fetch_max_explicit(&max_magnitude,
                                    mag_bits,
                                    memory_order_relaxed);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scale inputs if needed
    float max_magnitude_val = as_type<float>(atomic_load_explicit(&max_magnitude, memory_order_relaxed));
    if (max_magnitude_val > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude_val;
        for (uint i = lid; i < input_size && i < 784; i += WORKGROUP_SIZE) {
            shared_input[i] *= scale;
            shared_weights[i] *= scale;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process with numerical stability
    float sum = 0.0f;
    if (valid_computation) {
        // Two-pass summation for stability
        float max_term = 0.0f;
        for (uint i = 0; i < input_size && i < 784; i++) {
            float term = shared_input[i] * shared_weights[i];
            max_term = max(max_term, abs(term));
        }
        
        if (max_term > TOLERANCE && max_term < MAX_MAGNITUDE) {
            float scale = min(1.0f, MAX_MAGNITUDE / max_term);
            for (uint i = 0; i < input_size && i < 784; i++) {
                float term = shared_input[i] * shared_weights[i] * scale;
                sum += term;
            }
            sum /= scale;  // Rescale result
        }
    }
    
    // Apply quantum geometric transformation with bounds
    sum = clamp(sum, -MAX_MAGNITUDE, MAX_MAGNITUDE);
    float phase = M_PI_F * 0.5f * (1.0f + tanh(sum / 10.0f));  // Scaled tanh for stability
    
    // Compute quantum state with error checking
    float2 quantum_state;
    if (!is_valid_float(phase)) {
        quantum_state = float2(1.0f, 0.0f);  // Default to |0⟩ state
    } else {
        quantum_state = float2(cos(phase), sin(phase));
        
        // Normalize if needed
        float norm = length(quantum_state);
        if (norm > TOLERANCE && norm != 1.0f) {
            quantum_state /= norm;
        }
    }
    
    // Store result with probability bounds
    output[tid] = clamp(quantum_state.x * quantum_state.x, 0.0f, 1.0f);
}
