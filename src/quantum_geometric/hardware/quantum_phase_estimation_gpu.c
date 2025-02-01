#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/core/memory_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Apply quantum phase estimation on GPU
void apply_quantum_phase_gpu(GPUContext* ctx,
                           double complex* state,
                           size_t q,
                           size_t dim) {
    if (!ctx || !state) return;
    
    // Allocate device memory
    QuantumAmplitude* d_state = gpu_malloc(ctx,
        dim * sizeof(QuantumAmplitude));
    if (!d_state) return;
    
    // Convert state to GPU format
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        ((QuantumAmplitude*)d_state)[i].amplitude.x = creal(state[i]);
        ((QuantumAmplitude*)d_state)[i].amplitude.y = cimag(state[i]);
    }
    
    // Configure grid and block dimensions
    size_t block_size = get_optimal_workgroup_size();
    size_t grid_size = (dim + block_size - 1) / block_size;
    
    #ifdef __APPLE__
    id<MTLCommandBuffer> cmd_buffer = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    id<MTLComputePipelineState> pipeline = [ctx->device 
        newComputePipelineStateWithFunction:
        [ctx->library newFunctionWithName:@"quantum_phase_estimation"]
        error:nil];
    
    if (pipeline) {
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)d_state
                  offset:0
                 atIndex:0];
        [encoder setBytes:&q length:sizeof(uint) atIndex:1];
        [encoder setBytes:&dim length:sizeof(uint) atIndex:2];
        
        MTLSize grid = MTLSizeMake(grid_size * block_size, 1, 1);
        MTLSize block = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:block];
    }
    
    [encoder endEncoding];
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    #else
    dim3 grid((dim + block_size - 1) / block_size);
    dim3 block(block_size);
    
    quantum_phase_estimation<<<grid, block, 0, ctx->stream>>>(
        d_state, q, dim);
    
    cudaStreamSynchronize(ctx->stream);
    #endif
    
    // Copy result back to host
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        state[i] = ((QuantumAmplitude*)d_state)[i].amplitude.x +
                   I * ((QuantumAmplitude*)d_state)[i].amplitude.y;
    }
    
    // Cleanup
    gpu_free(ctx, d_state);
}

// Update quantum geometric tensor on GPU
void update_hmatrix_quantum_state_gpu(HierarchicalMatrix* mat,
                                    GPUContext* ctx) {
    if (!mat || !ctx) return;
    
    // Configure quantum geometric tensor computation
    quantum_tensor_config_t config = {
        .precision = 1e-10,
        .use_quantum_estimation = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Initialize or update QGT using GPU
    if (!mat->qgt) {
        // Allocate QGT structure
        mat->qgt = aligned_alloc(64, sizeof(QuantumGeometricTensor));
        if (!mat->qgt) return;
        
        size_t tensor_size = mat->rows * mat->cols;
        mat->qgt->metric = aligned_alloc(64,
            tensor_size * sizeof(double complex));
        mat->qgt->connection = aligned_alloc(64,
            tensor_size * sizeof(double complex));
        mat->qgt->curvature = aligned_alloc(64,
            tensor_size * sizeof(double complex));
        
        if (!mat->qgt->metric || !mat->qgt->connection ||
            !mat->qgt->curvature) {
            free(mat->qgt->metric);
            free(mat->qgt->connection);
            free(mat->qgt->curvature);
            free(mat->qgt);
            mat->qgt = NULL;
            return;
        }
    }
    
    // Allocate device memory
    size_t tensor_size = mat->rows * mat->cols;
    QuantumAmplitude* d_metric = gpu_malloc(ctx,
        tensor_size * sizeof(QuantumAmplitude));
    QuantumAmplitude* d_connection = gpu_malloc(ctx,
        tensor_size * sizeof(QuantumAmplitude));
    QuantumAmplitude* d_curvature = gpu_malloc(ctx,
        tensor_size * sizeof(QuantumAmplitude));
    
    if (!d_metric || !d_connection || !d_curvature) {
        gpu_free(ctx, d_metric);
        gpu_free(ctx, d_connection);
        gpu_free(ctx, d_curvature);
        return;
    }
    
    // Compute QGT components using GPU
    compute_quantum_metric_gpu(ctx, mat->data, d_metric,
        mat->rows, mat->cols, &config);
    compute_quantum_connection_gpu(ctx, mat->data, d_connection,
        mat->rows, mat->cols, &config);
    compute_quantum_curvature_gpu(ctx, mat->data, d_curvature,
        mat->rows, mat->cols, &config);
    
    // Copy results back to host
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            size_t idx = i * mat->cols + j;
            mat->qgt->metric[idx] = d_metric[idx].amplitude.x +
                                  I * d_metric[idx].amplitude.y;
            mat->qgt->connection[idx] = d_connection[idx].amplitude.x +
                                      I * d_connection[idx].amplitude.y;
            mat->qgt->curvature[idx] = d_curvature[idx].amplitude.x +
                                     I * d_curvature[idx].amplitude.y;
        }
    }
    
    // Cleanup
    gpu_free(ctx, d_metric);
    gpu_free(ctx, d_connection);
    gpu_free(ctx, d_curvature);
}

// Helper functions for QGT computation on GPU
static void compute_quantum_metric_gpu(GPUContext* ctx,
                                     const double complex* state,
                                     QuantumAmplitude* metric,
                                     size_t rows,
                                     size_t cols,
                                     const quantum_tensor_config_t* config) {
    size_t block_size = get_optimal_workgroup_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    #ifdef __APPLE__
    id<MTLCommandBuffer> cmd_buffer = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    id<MTLComputePipelineState> pipeline = [ctx->device 
        newComputePipelineStateWithFunction:
        [ctx->library newFunctionWithName:@"quantum_metric"]
        error:nil];
    
    if (pipeline) {
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)state
                  offset:0
                 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)metric
                  offset:0
                 atIndex:1];
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize grid = MTLSizeMake(grid_size * block_size, 1, 1);
        MTLSize block = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:block];
    }
    
    [encoder endEncoding];
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    #else
    dim3 grid((rows * cols + block_size - 1) / block_size);
    dim3 block(block_size);
    
    quantum_metric<<<grid, block, 0, ctx->stream>>>(
        state, metric, rows, cols);
    
    cudaStreamSynchronize(ctx->stream);
    #endif
}

static void compute_quantum_connection_gpu(GPUContext* ctx,
                                         const double complex* state,
                                         QuantumAmplitude* connection,
                                         size_t rows,
                                         size_t cols,
                                         const quantum_tensor_config_t* config) {
    size_t block_size = get_optimal_workgroup_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    #ifdef __APPLE__
    id<MTLCommandBuffer> cmd_buffer = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    id<MTLComputePipelineState> pipeline = [ctx->device 
        newComputePipelineStateWithFunction:
        [ctx->library newFunctionWithName:@"quantum_connection"]
        error:nil];
    
    if (pipeline) {
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)state
                  offset:0
                 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)connection
                  offset:0
                 atIndex:1];
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize grid = MTLSizeMake(grid_size * block_size, 1, 1);
        MTLSize block = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:block];
    }
    
    [encoder endEncoding];
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    #else
    dim3 grid((rows * cols + block_size - 1) / block_size);
    dim3 block(block_size);
    
    quantum_connection<<<grid, block, 0, ctx->stream>>>(
        state, connection, rows, cols);
    
    cudaStreamSynchronize(ctx->stream);
    #endif
}

static void compute_quantum_curvature_gpu(GPUContext* ctx,
                                        const double complex* state,
                                        QuantumAmplitude* curvature,
                                        size_t rows,
                                        size_t cols,
                                        const quantum_tensor_config_t* config) {
    size_t block_size = get_optimal_workgroup_size();
    size_t grid_size = (rows * cols + block_size - 1) / block_size;
    
    #ifdef __APPLE__
    id<MTLCommandBuffer> cmd_buffer = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    id<MTLComputePipelineState> pipeline = [ctx->device 
        newComputePipelineStateWithFunction:
        [ctx->library newFunctionWithName:@"quantum_curvature"]
        error:nil];
    
    if (pipeline) {
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)state
                  offset:0
                 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)curvature
                  offset:0
                 atIndex:1];
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize grid = MTLSizeMake(grid_size * block_size, 1, 1);
        MTLSize block = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:block];
    }
    
    [encoder endEncoding];
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    #else
    dim3 grid((rows * cols + block_size - 1) / block_size);
    dim3 block(block_size);
    
    quantum_curvature<<<grid, block, 0, ctx->stream>>>(
        state, curvature, rows, cols);
    
    cudaStreamSynchronize(ctx->stream);
    #endif
}
