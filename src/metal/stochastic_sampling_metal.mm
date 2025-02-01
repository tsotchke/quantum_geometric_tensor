#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "quantum_geometric/hardware/metal/metal_common.h"

// External interface functions
extern "C" {

metal_error_t compute_stochastic_sampling(
    void* input_states_buffer,
    void* probabilities_buffer,
    void* output_states_buffer,
    uint32_t num_states,
    uint32_t num_samples
) {
    if (!input_states_buffer || !probabilities_buffer || !output_states_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }
    
    @autoreleasepool {
        // Create compute pipeline
        void* pipeline = nullptr;
        metal_error_t error = metal_create_compute_pipeline("stochastic_sampling", &pipeline);
        if (error != METAL_SUCCESS) {
            return error;
        }
        
        // Set up buffers
        void* buffers[] = {
            input_states_buffer,
            probabilities_buffer,
            output_states_buffer
        };
        
        // Set up parameters
        struct {
            uint32_t num_states;
            uint32_t num_samples;
        } params = {num_states, num_samples};
        
        // Calculate thread groups
        uint32_t thread_groups = (num_samples + 255) / 256;
        
        // Execute command
        error = metal_execute_command(
            pipeline,
            buffers,
            3,
            &params,
            sizeof(params),
            thread_groups,
            1,
            1
        );
        
        metal_destroy_compute_pipeline(pipeline);
        return error;
    }
}

metal_error_t compute_stochastic_gradient(
    void* states_buffer,
    void* gradients_buffer,
    void* output_buffer,
    uint32_t batch_size,
    float learning_rate
) {
    if (!states_buffer || !gradients_buffer || !output_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }
    
    @autoreleasepool {
        // Create compute pipeline
        void* pipeline = nullptr;
        metal_error_t error = metal_create_compute_pipeline("stochastic_gradient", &pipeline);
        if (error != METAL_SUCCESS) {
            return error;
        }
        
        // Set up buffers
        void* buffers[] = {
            states_buffer,
            gradients_buffer,
            output_buffer
        };
        
        // Set up parameters
        struct {
            uint32_t batch_size;
            float learning_rate;
        } params = {batch_size, learning_rate};
        
        // Calculate thread groups
        uint32_t thread_groups = (batch_size + 255) / 256;
        
        // Execute command
        error = metal_execute_command(
            pipeline,
            buffers,
            3,
            &params,
            sizeof(params),
            thread_groups,
            1,
            1
        );
        
        metal_destroy_compute_pipeline(pipeline);
        return error;
    }
}

metal_error_t compute_importance_sampling(
    void* states_buffer,
    void* weights_buffer,
    void* indices_buffer,
    void* resampled_weights_buffer,
    uint32_t num_particles
) {
    if (!states_buffer || !weights_buffer || !indices_buffer || !resampled_weights_buffer) {
        return METAL_ERROR_INVALID_PARAMS;
    }
    
    @autoreleasepool {
        // Create compute pipeline
        void* pipeline = nullptr;
        metal_error_t error = metal_create_compute_pipeline("importance_sampling", &pipeline);
        if (error != METAL_SUCCESS) {
            return error;
        }
        
        // Set up buffers
        void* buffers[] = {
            states_buffer,
            weights_buffer,
            indices_buffer,
            resampled_weights_buffer
        };
        
        // Set up parameters
        struct {
            uint32_t num_particles;
        } params = {num_particles};
        
        // Calculate thread groups
        uint32_t thread_groups = (num_particles + 255) / 256;
        
        // Execute command
        error = metal_execute_command(
            pipeline,
            buffers,
            4,
            &params,
            sizeof(params),
            thread_groups,
            1,
            1
        );
        
        metal_destroy_compute_pipeline(pipeline);
        return error;
    }
}

} // extern "C"
