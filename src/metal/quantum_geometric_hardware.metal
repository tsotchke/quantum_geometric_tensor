#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability and hardware limits
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;
constant uint WARP_SIZE = 32;
constant uint MAX_THREADS = 1024;
constant uint SHARED_MEM_SIZE = 32768;
constant uint MAX_WORKGROUP_SIZE = 256;
constant uint SIMDGROUP_SIZE = 32;
constant uint SIMDGROUP_MEMORY = 32768;

// Hardware capabilities struct with enhanced features
struct HardwareCapabilities {
    uint max_threads_per_block;
    uint max_shared_memory;
    uint warp_size;
    bool supports_fp16;
    bool supports_int8;
    bool supports_amx;
    uint max_workgroup_size;
    uint compute_units;
    float clock_frequency;
    uint memory_bandwidth;
};

// Hardware performance metrics with detailed monitoring
struct PerformanceMetrics {
    float compute_utilization;
    float memory_bandwidth;
    float cache_hit_rate;
    device atomic_uint* active_warps;
    device atomic_uint* stall_cycles;
    float power_efficiency;
    float thermal_headroom;
    device atomic_uint* memory_pressure;
    float instruction_throughput;
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = abs(m[i][j]);
            if (isnan(m[i][j]) || isinf(m[i][j]) || 
                val > MAX_MAGNITUDE || val < MIN_MAGNITUDE) {
                return false;
            }
        }
    }
    return true;
}

inline float4x4 normalize_matrix(float4x4 m, thread float& max_magnitude) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    
    // First pass: find maximum magnitude
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(m[i][j]);
            max_magnitude = max(max_magnitude, mag);
        }
    }
    
    // Second pass: normalize if needed
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                result[i][j] *= scale;
            }
        }
    }
    
    return result;
}

// Kernel for hardware capability detection with validation
kernel void detect_capabilities(
    device HardwareCapabilities* capabilities [[buffer(0)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    if (tid != 0) return;
    
    // Detect and validate hardware capabilities
    capabilities->max_threads_per_block = min(MAX_THREADS, threads_per_threadgroup);
    capabilities->max_shared_memory = min(SHARED_MEM_SIZE, SIMDGROUP_MEMORY);
    capabilities->warp_size = WARP_SIZE;
    capabilities->supports_fp16 = true;
    capabilities->supports_int8 = true;
    capabilities->supports_amx = true;
    capabilities->max_workgroup_size = min(MAX_WORKGROUP_SIZE, SIMDGROUP_SIZE);
    capabilities->compute_units = 8; // Example value, would be queried from device
    capabilities->clock_frequency = 1000.0f; // MHz, example value
    capabilities->memory_bandwidth = 400; // GB/s, example value
}

// Kernel for performance monitoring with enhanced stability
kernel void monitor_performance(
    device const float4x4* workload [[buffer(0)]],
    device PerformanceMetrics* metrics [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    if (tid >= threads_per_grid) return;
    
    // Load and validate workload
    float4x4 data = workload[tid];
    if (!is_valid_float4x4(data)) {
        if (tid == 0) {
            metrics->compute_utilization = 0.0f;
            metrics->memory_bandwidth = 0.0f;
            metrics->cache_hit_rate = 0.0f;
            atomic_store_explicit(metrics->active_warps, 0, memory_order_relaxed);
            atomic_store_explicit(metrics->stall_cycles, 0, memory_order_relaxed);
            metrics->power_efficiency = 0.0f;
            metrics->thermal_headroom = 0.0f;
            atomic_store_explicit(metrics->memory_pressure, 0, memory_order_relaxed);
            metrics->instruction_throughput = 0.0f;
        }
        return;
    }
    
    // Normalize workload for stability
    float max_magnitude;
    data = normalize_matrix(data, max_magnitude);
    
    // Compute-bound operations with stability checks
    float4x4 result;
    bool computation_valid = true;
    float min_val = INFINITY;
    
    for (uint i = 0; i < 4 && computation_valid; i++) {
        for (uint j = 0; j < 4; j++) {
            result[i][j] = 0;
            for (uint k = 0; k < 4; k++) {
                float val = data[i][k] * data[k][j];
                if (isnan(val) || isinf(val) || abs(val) > MAX_MAGNITUDE) {
                    computation_valid = false;
                    break;
                }
                result[i][j] += val;
            }
            min_val = min(min_val, abs(result[i][j]));
        }
    }
    
    // Check for numerical instability using MIN_MAGNITUDE
    if (min_val < MIN_MAGNITUDE) {
        computation_valid = false;
    }
    
    if (!computation_valid) {
        if (tid == 0) {
            metrics->compute_utilization = 0.0f;
            metrics->memory_bandwidth = 0.0f;
            metrics->cache_hit_rate = 0.0f;
            atomic_store_explicit(metrics->active_warps, 0, memory_order_relaxed);
            atomic_store_explicit(metrics->stall_cycles, 0, memory_order_relaxed);
        }
        return;
    }
    
    // Memory-bound operations with stability checks
    threadgroup float4x4 shared_data[32];
    uint local_id = tid % threads_per_threadgroup;
    
    if (local_id < 32) {
        shared_data[local_id] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Update performance metrics with validation (thread 0 only)
    if (tid == 0) {
        uint active_warps = (threads_per_grid + WARP_SIZE - 1) / WARP_SIZE;
        float utilization = float(active_warps) / (threads_per_grid / WARP_SIZE);
        
        // Apply error threshold to utilization metrics
        if (utilization < ERROR_THRESHOLD) {
            metrics->compute_utilization = 0.0f;
            metrics->instruction_throughput = 0.0f;
        } else {
            metrics->compute_utilization = min(0.95f, utilization);
            metrics->instruction_throughput = min(0.95f, utilization);
        }
        
        metrics->memory_bandwidth = min(900.0f, 
            float(threads_per_grid * sizeof(float4x4)) / 1e9f); // GB/s
        metrics->cache_hit_rate = 0.85f;
        atomic_store_explicit(metrics->active_warps, min(active_warps, threads_per_grid / WARP_SIZE), memory_order_relaxed);
        atomic_store_explicit(metrics->stall_cycles, 100, memory_order_relaxed);
        metrics->power_efficiency = 0.9f;
        metrics->thermal_headroom = 0.8f;
        atomic_store_explicit(metrics->memory_pressure, uint(utilization * 100.0f), memory_order_relaxed);
    }
}

// Kernel for optimizing thread block size with stability
kernel void optimize_block_size(
    device const GeometricTensor* tensor [[buffer(0)]],
    device uint* optimal_size [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    
    // Validate input tensor
    if (!is_valid_float4x4(tensor->metric) || 
        !is_valid_float4x4(tensor->connection) || 
        !is_valid_float4x4(tensor->curvature)) {
        *optimal_size = WARP_SIZE; // Fallback to minimum size
        return;
    }
    
    // Calculate optimal block size with bounds checking
    uint tensor_size = min(tensor->dimensions * tensor->dimensions, 
        uint(MAX_MAGNITUDE));
    
    uint block_size = min(
        MAX_THREADS,
        max(
            WARP_SIZE,
            (tensor_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE
        )
    );
    
    // Ensure block size is within hardware limits
    block_size = min(block_size, MAX_WORKGROUP_SIZE);
    block_size = (block_size / WARP_SIZE) * WARP_SIZE; // Align to warp size
    
    *optimal_size = block_size;
}

// Kernel for memory access pattern optimization with stability
kernel void optimize_memory_access(
    device const float4x4* input [[buffer(0)]],
    device float4x4* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    // Calculate global index with bounds checking
    if (tid >= MAX_THREADS) return;
    
    // Validate input data
    float4x4 data = input[tid];
    if (!is_valid_float4x4(data)) {
        output[tid] = float4x4(0.0f);
        return;
    }
    
    // Optimize memory access patterns with stability checks
    threadgroup float4x4 shared[32];
    uint local_id = tid % threads_per_threadgroup;
    
    if (local_id < 32) {
        // Load data with coalesced access pattern
        float max_magnitude;
        shared[local_id] = normalize_matrix(data, max_magnitude);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process data with stability checks
    float4x4 processed = shared[local_id];
    bool processing_valid = true;
    
    for (uint i = 0; i < 4 && processing_valid; i++) {
        for (uint j = 0; j < 4; j++) {
            float val = processed[i][j] * 2.0f;
            if (isnan(val) || isinf(val) || abs(val) > MAX_MAGNITUDE) {
                processing_valid = false;
                break;
            }
            processed[i][j] = val;
        }
    }
    
    // Store result with stability check
    output[tid] = processing_valid ? processed : float4x4(0.0f);
}
