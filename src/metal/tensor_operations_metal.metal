#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Constants for numerical stability and performance
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-3f;
constant uint TILE_SIZE = 32;
constant uint MATRIX_SIZE = 8;

// Helper functions for numerical stability
inline bool is_valid_float(float x) {
    float abs_x = abs(x);
    return !isnan(x) && !isinf(x) && 
           (abs_x == 0.0f || (abs_x >= MIN_MAGNITUDE && abs_x <= MAX_MAGNITUDE));
}

// Complex number struct optimized for SIMD
struct ComplexFloat {
    half2 value;  // Pack real/imag into half2 for better bandwidth
    bool valid;   // Track numerical validity
};

// Helper functions for ComplexFloat
inline ComplexFloat make_complex_float() {
    ComplexFloat result;
    result.value = half2(0.0h);
    result.valid = true;
    return result;
}

inline ComplexFloat make_complex_float(float r, float i) {
    ComplexFloat result;
    result.valid = is_valid_float(r) && is_valid_float(i);
    if (result.valid) {
        float mag = sqrt(r*r + i*i);
        if (mag > MAX_MAGNITUDE) {
            float scale = MAX_MAGNITUDE / mag;
            result.value = half2(r * scale, i * scale);
        } else if (mag > 0.0f && mag < MIN_MAGNITUDE) {
            float scale = MIN_MAGNITUDE / mag;
            result.value = half2(r * scale, i * scale);
        } else {
            result.value = half2(r, i);
        }
    } else {
        result.value = half2(0.0h);
    }
    return result;
}

inline float magnitude(ComplexFloat x) {
    if (!x.valid) return 0.0f;
    float mag = sqrt(x.value.x * x.value.x + x.value.y * x.value.y);
    if (mag > 0.0f && mag < MIN_MAGNITUDE) {
        return MIN_MAGNITUDE;
    }
    return mag;
}

// Matrix multiplication using Metal's SIMD operations
METAL_FUNC void matrix_multiply_add(
    thread const half2* a_data,
    thread const half2* b_data,
    thread half2* c_data,
    uint M, uint N, uint K)
{
    // Create temporary arrays for matrix elements
    thread half a_real_arr[64];
    thread half a_imag_arr[64];
    thread half b_real_arr[64];
    thread half b_imag_arr[64];
    thread half c_real_arr[64];
    thread half c_imag_arr[64];
    
    // Load matrices into temporary arrays
    for (uint i = 0; i < 8; i++) {
        for (uint j = 0; j < 8; j++) {
            a_real_arr[i * 8 + j] = a_data[i * K + j].x;
            b_real_arr[i * 8 + j] = b_data[i * N + j].x;
            a_imag_arr[i * 8 + j] = a_data[i * K + j].y;
            b_imag_arr[i * 8 + j] = b_data[i * N + j].y;
        }
    }
    
    // Process blocks
    for (uint i = 0; i < 8; i++) {
        for (uint j = 0; j < 8; j++) {
            half sum_real = 0.0h;
            half sum_imag = 0.0h;
            for (uint k = 0; k < 8; k++) {
                // Complex multiplication
                sum_real += a_real_arr[i * 8 + k] * b_real_arr[k * 8 + j] - 
                           a_imag_arr[i * 8 + k] * b_imag_arr[k * 8 + j];
                sum_imag += a_real_arr[i * 8 + k] * b_imag_arr[k * 8 + j] + 
                           a_imag_arr[i * 8 + k] * b_real_arr[k * 8 + j];
            }
            c_real_arr[i * 8 + j] = sum_real;
            c_imag_arr[i * 8 + j] = sum_imag;
        }
    }
    
    // Store final results
    for (uint i = 0; i < 8; i++) {
        for (uint j = 0; j < 8; j++) {
            c_data[i * N + j] = half2(c_real_arr[i * 8 + j], c_imag_arr[i * 8 + j]);
        }
    }
}

// Optimized matrix multiplication kernel for Apple Silicon
kernel void matrix_multiply_optimized(
    device const ComplexFloat* A [[buffer(0)]],
    device const ComplexFloat* B [[buffer(1)]],
    device ComplexFloat* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 bid [[thread_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
    // Shared memory for tiles
    threadgroup half2 A_tile_values[TILE_SIZE][TILE_SIZE];
    threadgroup bool A_tile_valid[TILE_SIZE][TILE_SIZE];
    threadgroup half2 B_tile_values[TILE_SIZE][TILE_SIZE];
    threadgroup bool B_tile_valid[TILE_SIZE][TILE_SIZE];
    
    // Error tracking
    threadgroup atomic_uint error_count;
    threadgroup atomic_uint max_magnitude;
    
    // Initialize shared memory
    if (simdgroup_index == 0 && tid.x == 0 && tid.y == 0) {
        atomic_store_explicit(&error_count, 0, memory_order_relaxed);
        atomic_store_explicit(&max_magnitude, as_type<uint>(0.0f), memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate tile indices
    const uint tile_row = bid.y;
    const uint tile_col = bid.x;
    const uint local_row = tid.y;
    const uint local_col = tid.x;
    
    // Accumulator for this thread
    thread half2 acc[MATRIX_SIZE][MATRIX_SIZE];
    for (uint i = 0; i < MATRIX_SIZE; i++) {
        for (uint j = 0; j < MATRIX_SIZE; j++) {
            acc[i][j] = half2(0.0h);
        }
    }
    
    // Process tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        const uint a_row = tile_row * TILE_SIZE + local_row;
        const uint a_col = t * TILE_SIZE + local_col;
        const uint b_row = t * TILE_SIZE + local_row;
        const uint b_col = tile_col * TILE_SIZE + local_col;
        
        if (a_row < M && a_col < K) {
            ComplexFloat a_val = A[a_row * K + a_col];
            A_tile_values[local_row][local_col] = a_val.value;
            A_tile_valid[local_row][local_col] = a_val.valid;
            float mag = magnitude(a_val);
            if (!is_valid_float(mag)) {
                atomic_fetch_add_explicit(&error_count, 1, memory_order_relaxed);
            } else {
                uint current_max = atomic_load_explicit(&max_magnitude, memory_order_relaxed);
                while (as_type<float>(current_max) < mag) {
                    atomic_compare_exchange_weak_explicit(&max_magnitude,
                        &current_max,
                        as_type<uint>(mag),
                        memory_order_relaxed,
                        memory_order_relaxed);
                }
            }
        }
        
        if (b_row < K && b_col < N) {
            ComplexFloat b_val = B[b_row * N + b_col];
            B_tile_values[local_row][local_col] = b_val.value;
            B_tile_valid[local_row][local_col] = b_val.valid;
            float mag = magnitude(b_val);
            if (!is_valid_float(mag)) {
                atomic_fetch_add_explicit(&error_count, 1, memory_order_relaxed);
            } else {
                uint current_max = atomic_load_explicit(&max_magnitude, memory_order_relaxed);
                while (as_type<float>(current_max) < mag) {
                    atomic_compare_exchange_weak_explicit(&max_magnitude,
                        &current_max,
                        as_type<uint>(mag),
                        memory_order_relaxed,
                        memory_order_relaxed);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Skip tile if too many errors
        if (atomic_load_explicit(&error_count, memory_order_relaxed) > TILE_SIZE * TILE_SIZE / 4) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        
        // Process tile with SIMD operations
        for (uint k = 0; k < TILE_SIZE; k += MATRIX_SIZE) {
            // Extract data for SIMD multiplication
            thread half2 a_block[MATRIX_SIZE][MATRIX_SIZE];
            thread half2 b_block[MATRIX_SIZE][MATRIX_SIZE];
            
            for (uint i = 0; i < MATRIX_SIZE; i++) {
                for (uint j = 0; j < MATRIX_SIZE; j++) {
                    if (local_row + i < TILE_SIZE && k + j < TILE_SIZE) {
                        if (A_tile_valid[local_row + i][k + j] && B_tile_valid[k + i][local_col + j]) {
                            a_block[i][j] = A_tile_values[local_row + i][k + j];
                            b_block[i][j] = B_tile_values[k + i][local_col + j];
                        } else {
                            a_block[i][j] = half2(0.0h);
                            b_block[i][j] = half2(0.0h);
                        }
                    }
                }
            }
            
            // Multiply blocks
            matrix_multiply_add(
                (thread const half2*)a_block,
                (thread const half2*)b_block,
                (thread half2*)acc,
                MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE
            );
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    const uint global_row = tile_row * TILE_SIZE + local_row;
    const uint global_col = tile_col * TILE_SIZE + local_col;
    
    if (global_row < M && global_col < N) {
        float scale = 1.0f;
        float max_val = as_type<float>(atomic_load_explicit(&max_magnitude, memory_order_relaxed));
        if (max_val > MAX_MAGNITUDE) {
            scale = MAX_MAGNITUDE / max_val;
        }
        
        // Reduce accumulator
        half2 sum = half2(0.0h);
        for (uint i = 0; i < MATRIX_SIZE; i++) {
            for (uint j = 0; j < MATRIX_SIZE; j++) {
                sum += acc[i][j];
            }
        }
        
        ComplexFloat result = make_complex_float();
        result.value = sum * scale;
        result.valid = true;
        C[global_row * N + global_col] = result;
    }
}
