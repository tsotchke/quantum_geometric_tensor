// C++ standard headers
#include <complex>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <thread>
#include <future>

// Project headers
#include "quantum_geometric/hardware/quantum_geometric_tensor_ops.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_config.h"

// Then C headers wrapped in extern "C"
extern "C" {
#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/quantum_geometric_core.h"

#ifdef ENABLE_METAL
#include "quantum_geometric/hardware/metal/quantum_geometric_metal.h"
#include "quantum_geometric/hardware/metal/mnist_metal.h"
#endif

#ifdef ENABLE_CUDA
#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#endif
}

using namespace quantum_geometric::cpu;

// Optimized pipeline structure
struct quantum_pipeline {
    // Configuration
    float input_dim;
    float latent_dim;
    float num_clusters;
    float num_classes;
    float batch_size;
    float learning_rate;
    int use_gpu;
    int num_qubits;  // Dynamic qubit count
    
    // GPU resources
#ifdef ENABLE_METAL
    float execution_time_ms;
#endif
    
    // Pipeline stages
    void* encoder_state;
    void* transformer_state;
    void* classifier_state;
    
    // Async processing queues
    std::future<void> encoder_future;
    std::future<void> transformer_future;
    std::future<void> classifier_future;
    
    // Double buffering
    void* input_buffers[2];
    void* intermediate_buffers[2];
    void* output_buffers[2];
    int current_buffer;
};

// Optimized complex conversion on GPU
static void convert_to_complex_gpu(const float* input, std::complex<double>* output,
                                 size_t size, cudaStream_t stream) {
    // Launch kernel to do conversion on GPU
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    convert_to_complex_kernel<<<grid, block, 0, stream>>>(
        input, output, size);
}

// Pipeline stage implementations
static void encode_stage(quantum_pipeline* pipeline, const float* data,
                        size_t batch_size, cudaStream_t stream) {
    // Get optimal qubit count for input dimension
    pipeline->num_qubits = (int)ceil(log2(pipeline->input_dim));
    
    // Convert and encode on GPU
    convert_to_complex_gpu(data, 
                          (std::complex<double>*)pipeline->input_buffers[pipeline->current_buffer],
                          batch_size * pipeline->input_dim,
                          stream);
                          
    // Apply quantum encoding
    QuantumGeometricState state;
    state.amplitudes = (std::complex<double>*)pipeline->input_buffers[pipeline->current_buffer];
    state.state_size = batch_size * pipeline->input_dim;
    state.num_qubits = pipeline->num_qubits;
    
    encode_quantum_state_gpu(&state, stream);
}

static void transform_stage(quantum_pipeline* pipeline, size_t batch_size,
                          cudaStream_t stream) {
    // Get input from previous stage
    QuantumGeometricState state;
    state.amplitudes = (std::complex<double>*)pipeline->input_buffers[pipeline->current_buffer];
    state.state_size = batch_size * pipeline->input_dim;
    state.num_qubits = pipeline->num_qubits;
    
    // Apply geometric transformation
    apply_geometric_transform_gpu(
        &state,
        (const float*)pipeline->transformer_state,
        batch_size,
        pipeline->latent_dim,
        stream
    );
    
    // Store result in intermediate buffer
    cudaMemcpyAsync(
        pipeline->intermediate_buffers[pipeline->current_buffer],
        state.amplitudes,
        state.state_size * sizeof(std::complex<double>),
        cudaMemcpyDeviceToDevice,
        stream
    );
}

static void classify_stage(quantum_pipeline* pipeline, size_t batch_size,
                         float* results, cudaStream_t stream) {
    // Get input from previous stage
    QuantumGeometricState state;
    state.amplitudes = (std::complex<double>*)pipeline->intermediate_buffers[pipeline->current_buffer];
    state.state_size = batch_size * pipeline->latent_dim;
    state.num_qubits = (int)ceil(log2(pipeline->latent_dim));
    
    // Apply classification
    measure_quantum_state_gpu(
        &state,
        (float*)pipeline->output_buffers[pipeline->current_buffer],
        batch_size,
        pipeline->num_classes,
        stream
    );
    
    // Copy results back to host if needed
    if (results) {
        cudaMemcpyAsync(
            results,
            pipeline->output_buffers[pipeline->current_buffer],
            batch_size * pipeline->num_classes * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
    }
}

extern "C" void* quantum_pipeline_create_impl(const float* config) {
    if (!config) return NULL;

    quantum_pipeline* pipeline = new quantum_pipeline();
    if (!pipeline) return NULL;

    // Copy configuration
    pipeline->input_dim = config[0];
    pipeline->latent_dim = config[1];
    pipeline->num_clusters = config[2];
    pipeline->num_classes = config[3];
    pipeline->batch_size = config[4];
    pipeline->learning_rate = config[5];
    pipeline->use_gpu = (int)config[6];
    pipeline->current_buffer = 0;

    // Get tensor configuration
    const QGTensorConfig* tensor_config = qgt_get_config();
    if (tensor_config->acceleration_type == QGT_ACCELERATION_METAL) {
        pipeline->use_gpu = 1;
    }

    // Validate parameters
    if (pipeline->input_dim <= 0 || pipeline->latent_dim <= 0 ||
        pipeline->num_clusters <= 0 || pipeline->num_classes <= 0 ||
        pipeline->batch_size <= 0 || pipeline->learning_rate <= 0) {
        delete pipeline;
        return NULL;
    }

    // Allocate pipeline stages
    size_t encoder_size = sizeof(float) * pipeline->input_dim * pipeline->latent_dim;
    size_t transformer_size = sizeof(float) * pipeline->latent_dim * pipeline->num_clusters;
    size_t classifier_size = sizeof(float) * pipeline->num_clusters * pipeline->num_classes;

    pipeline->encoder_state = qg_memory_pool_alloc(NULL, encoder_size);
    pipeline->transformer_state = qg_memory_pool_alloc(NULL, transformer_size);
    pipeline->classifier_state = qg_memory_pool_alloc(NULL, classifier_size);

    if (!pipeline->encoder_state || !pipeline->transformer_state || 
        !pipeline->classifier_state) {
        qg_memory_pool_free(NULL, pipeline->encoder_state);
        qg_memory_pool_free(NULL, pipeline->transformer_state);
        qg_memory_pool_free(NULL, pipeline->classifier_state);
        delete pipeline;
        return NULL;
    }

    // Allocate double buffers
    size_t buffer_size = std::max({
        pipeline->input_dim * pipeline->batch_size,
        pipeline->latent_dim * pipeline->batch_size,
        pipeline->num_classes * pipeline->batch_size
    }) * sizeof(std::complex<double>);

    for (int i = 0; i < 2; i++) {
        cudaMalloc(&pipeline->input_buffers[i], buffer_size);
        cudaMalloc(&pipeline->intermediate_buffers[i], buffer_size);
        cudaMalloc(&pipeline->output_buffers[i], buffer_size);
        
        if (!pipeline->input_buffers[i] || !pipeline->intermediate_buffers[i] ||
            !pipeline->output_buffers[i]) {
            // Cleanup on error
            for (int j = 0; j <= i; j++) {
                if (pipeline->input_buffers[j]) cudaFree(pipeline->input_buffers[j]);
                if (pipeline->intermediate_buffers[j]) cudaFree(pipeline->intermediate_buffers[j]);
                if (pipeline->output_buffers[j]) cudaFree(pipeline->output_buffers[j]);
            }
            qg_memory_pool_free(NULL, pipeline->encoder_state);
            qg_memory_pool_free(NULL, pipeline->transformer_state);
            qg_memory_pool_free(NULL, pipeline->classifier_state);
            delete pipeline;
            return NULL;
        }
    }

    return pipeline;
}

extern "C" int quantum_pipeline_train_impl(void* handle,
                                         const float* data,
                                         const int* labels,
                                         size_t num_samples) {
    quantum_pipeline* pipeline = (quantum_pipeline*)handle;
    if (!pipeline || !data || !labels || num_samples == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    clock_t start = clock();
    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process batches with pipeline parallelism
    for (size_t i = 0; i < num_samples; i += (size_t)pipeline->batch_size) {
        size_t batch_size = std::min((size_t)pipeline->batch_size, num_samples - i);
        
        // Launch pipeline stages asynchronously
        pipeline->encoder_future = std::async(std::launch::async,
            encode_stage, pipeline, &data[i * (size_t)pipeline->input_dim],
            batch_size, streams[0]);
            
        pipeline->transformer_future = std::async(std::launch::async,
            transform_stage, pipeline, batch_size, streams[1]);
            
        pipeline->classifier_future = std::async(std::launch::async,
            classify_stage, pipeline, batch_size, nullptr, streams[2]);
            
        // Wait for pipeline completion
        pipeline->encoder_future.wait();
        pipeline->transformer_future.wait();
        pipeline->classifier_future.wait();
        
        // Swap double buffers
        pipeline->current_buffer = 1 - pipeline->current_buffer;
    }

    // Cleanup
    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    pipeline->execution_time_ms = (float)(clock() - start) / CLOCKS_PER_SEC * 1000.0f;
    return QGT_SUCCESS;
}

extern "C" int quantum_pipeline_evaluate_impl(void* handle,
                                           const float* data,
                                           const int* labels,
                                           size_t num_samples,
                                           float* results) {
    quantum_pipeline* pipeline = (quantum_pipeline*)handle;
    if (!pipeline || !data || !labels || num_samples == 0 || !results) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    clock_t start = clock();
    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        cudaStreamCreate(&streams[i]);
    }

    size_t correct = 0;
    float* predictions = (float*)malloc(pipeline->num_classes * sizeof(float));
    if (!predictions) return QGT_ERROR_MEMORY_ALLOCATION;

    // Process batches
    for (size_t i = 0; i < num_samples; i += (size_t)pipeline->batch_size) {
        size_t batch_size = std::min((size_t)pipeline->batch_size, num_samples - i);
        
        // Launch pipeline stages
        pipeline->encoder_future = std::async(std::launch::async,
            encode_stage, pipeline, &data[i * (size_t)pipeline->input_dim],
            batch_size, streams[0]);
            
        pipeline->transformer_future = std::async(std::launch::async,
            transform_stage, pipeline, batch_size, streams[1]);
            
        pipeline->classifier_future = std::async(std::launch::async,
            classify_stage, pipeline, batch_size, predictions, streams[2]);
            
        // Wait for results
        pipeline->encoder_future.wait();
        pipeline->transformer_future.wait();
        pipeline->classifier_future.wait();
        
        // Count correct predictions
        for (size_t j = 0; j < batch_size; j++) {
            int predicted_class = 0;
            float max_prob = predictions[j * (size_t)pipeline->num_classes];
            
            for (int k = 1; k < (int)pipeline->num_classes; k++) {
                float prob = predictions[j * (size_t)pipeline->num_classes + k];
                if (prob > max_prob) {
                    max_prob = prob;
                    predicted_class = k;
                }
            }
            
            if (predicted_class == labels[i + j]) correct++;
        }
        
        // Swap buffers
        pipeline->current_buffer = 1 - pipeline->current_buffer;
    }

    // Cleanup
    for (int i = 0; i < 3; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(predictions);

    // Calculate metrics
    results[0] = (float)correct / num_samples;  // accuracy
    results[1] = (float)(clock() - start) / CLOCKS_PER_SEC * 1000.0f;  // time in ms
    results[2] = (float)(qg_memory_pool_get_used_size(NULL) / (1024.0 * 1024.0));  // memory in MB

    return QGT_SUCCESS;
}

extern "C" void quantum_pipeline_destroy_impl(void* handle) {
    quantum_pipeline* pipeline = (quantum_pipeline*)handle;
    if (!pipeline) return;

    // Free pipeline stages
    qg_memory_pool_free(NULL, pipeline->encoder_state);
    qg_memory_pool_free(NULL, pipeline->transformer_state);
    qg_memory_pool_free(NULL, pipeline->classifier_state);

    // Free double buffers
    for (int i = 0; i < 2; i++) {
        if (pipeline->input_buffers[i]) cudaFree(pipeline->input_buffers[i]);
        if (pipeline->intermediate_buffers[i]) cudaFree(pipeline->intermediate_buffers[i]);
        if (pipeline->output_buffers[i]) cudaFree(pipeline->output_buffers[i]);
    }

    delete pipeline;
}

extern "C" int quantum_pipeline_save_impl(void* handle, const char* filename) {
    quantum_pipeline* pipeline = (quantum_pipeline*)handle;
    if (!pipeline || !filename) return QGT_ERROR_INVALID_ARGUMENT;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return QGT_ERROR_FILE_OPERATION;

    // Write configuration
    if (fwrite(&pipeline->input_dim, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->latent_dim, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->num_clusters, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->num_classes, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->batch_size, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->learning_rate, sizeof(float), 1, fp) != 1 ||
        fwrite(&pipeline->use_gpu, sizeof(int), 1, fp) != 1 ||
        fwrite(&pipeline->num_qubits, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return QGT_ERROR_FILE_OPERATION;
    }

    // Write pipeline stages
    size_t encoder_size = sizeof(float) * pipeline->input_dim * pipeline->latent_dim;
    size_t transformer_size = sizeof(float) * pipeline->latent_dim * pipeline->num_clusters;
    size_t classifier_size = sizeof(float) * pipeline->num_clusters * pipeline->num_classes;

    if (fwrite(pipeline->encoder_state, 1, encoder_size, fp) != encoder_size ||
        fwrite(pipeline->transformer_state, 1, transformer_size, fp) != transformer_size ||
        fwrite(pipeline->classifier_state, 1, classifier_size, fp) != classifier_size) {
        fclose(fp);
        return QGT_ERROR_FILE_OPERATION;
    }

    fclose(fp);
    return QGT_SUCCESS;
}
