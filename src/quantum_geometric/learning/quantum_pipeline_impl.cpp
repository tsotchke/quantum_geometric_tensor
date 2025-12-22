#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_scheduler.h"
#include "quantum_geometric/core/operation_fusion.h"
#include "quantum_geometric/learning/learning_task.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <vector>
#include <memory>
#include <stdexcept>

// Internal pipeline state
class QuantumPipelineState {
public:
    // Core components
    std::unique_ptr<computational_graph_t> model_graph;
    std::unique_ptr<computational_graph_t> optimizer_graph;
    learning_task_handle_t learning_task;  // Raw pointer as it's managed by C API
    
    // Scheduler components
    scheduler_config_t scheduler_config;
    quantum_task_t* training_task;
    
    // Pipeline configuration
    float config[QG_CONFIG_SIZE];
    
    // Performance metrics
    float gpu_utilization;
    size_t memory_usage;
    float throughput;
    
    // Training state
    size_t current_epoch;
    float current_loss;
    float current_accuracy;
    bool is_training;

    QuantumPipelineState() : 
        learning_task(nullptr),
        training_task(nullptr),
        gpu_utilization(0.0f),
        memory_usage(0),
        throughput(0.0f),
        current_epoch(0),
        current_loss(0.0f),
        current_accuracy(0.0f),
        is_training(false) {}

    ~QuantumPipelineState() {
        if (learning_task) {
            quantum_destroy_learning_task(learning_task);
        }
    }
};

extern "C" {

void* quantum_pipeline_create_impl(const float* config) {
    if (!config) return nullptr;
    
    try {
        auto state = new QuantumPipelineState();
        
        // Store configuration
        std::copy(config, config + QG_CONFIG_SIZE, state->config);
        
        // Initialize geometric processor
        geometric_processor_t* processor = create_geometric_processor(nullptr);
        if (!processor) {
            delete state;
            return nullptr;
        }
        
        // Create computational graphs
        state->model_graph.reset(create_computational_graph(processor));
        state->optimizer_graph.reset(create_computational_graph(processor));
        
        if (!state->model_graph || !state->optimizer_graph) {
            delete state;
            return nullptr;
        }
        
        // Initialize learning task
        task_config_t task_config;
        task_config.task_type = TASK_CLASSIFICATION;
        task_config.model_type = MODEL_QUANTUM_NEURAL_NETWORK;
        task_config.optimizer_type = OPTIMIZER_QUANTUM_GRADIENT_DESCENT;
        task_config.input_dim = static_cast<size_t>(config[QG_CONFIG_INPUT_DIM]);
        task_config.output_dim = static_cast<size_t>(config[QG_CONFIG_NUM_CLASSES]);
        task_config.latent_dim = static_cast<size_t>(config[QG_CONFIG_LATENT_DIM]);
        task_config.num_qubits = config[QG_CONFIG_NUM_QUBITS] > 0 ? static_cast<size_t>(config[QG_CONFIG_NUM_QUBITS]) : 8;
        task_config.num_layers = config[QG_CONFIG_NUM_LAYERS] > 0 ? static_cast<size_t>(config[QG_CONFIG_NUM_LAYERS]) : 4;
        task_config.batch_size = static_cast<size_t>(config[QG_CONFIG_BATCH_SIZE]);
        task_config.learning_rate = static_cast<double>(config[QG_CONFIG_LEARNING_RATE]);
        task_config.use_gpu = config[QG_CONFIG_USE_GPU] > 0.5f;
        task_config.enable_error_mitigation = true;
        task_config.num_shots = 1024;
        
        state->learning_task = quantum_create_learning_task(&task_config);
        if (!state->learning_task) {
            delete state;
            return nullptr;
        }
        
        // Configure scheduler
        state->scheduler_config = {
            .max_concurrent_tasks = 4,
            .max_queue_size = 1000,
            .enable_preemption = true,
            .enable_load_balancing = true,
            .scheduling_interval = 0.001,
            .min_priority = PRIORITY_LOW
        };
        
        if (!initialize_quantum_scheduler(&state->scheduler_config)) {
            delete state;
            return nullptr;
        }
        
        // Initialize fusion optimizer
        fusion_config_t fusion_config = {
            .enable_quantum_fusion = true,
            .enable_classical_fusion = true,
            .min_cost_reduction = 0.1,
            .max_group_size = 8,
            .preserve_gradients = true
        };
        
        if (!initialize_fusion_optimizer(&fusion_config)) {
            delete state;
            return nullptr;
        }
        
        return state;
    } catch (const std::exception&) {
        return nullptr;
    }
}

int quantum_pipeline_train_impl(void* pipeline, const float* data, const int* labels, size_t num_samples) {
    auto state = static_cast<QuantumPipelineState*>(pipeline);
    if (!state || !data || !labels || num_samples == 0) return QG_ERROR_INVALID_ARGUMENT;
    
    try {
        // Create training task
        state->training_task = create_task(nullptr, PRIORITY_HIGH);
        if (!state->training_task) return QG_ERROR_INITIALIZATION;
        
        // Configure resource requirements
        resource_requirement_t req = {
            .type = state->config[QG_CONFIG_USE_GPU] > 0.5f ? RESOURCE_GPU : RESOURCE_CPU,
            .quantity = 1,
            .duration = 0.0,
            .exclusive = false
        };
        
        state->training_task->requirements = new resource_requirement_t[1];
        state->training_task->requirements[0] = req;
        state->training_task->num_requirements = 1;
        
        // Submit task to scheduler
        if (!submit_task(state->training_task)) {
            delete[] state->training_task->requirements;
            delete state->training_task;
            state->training_task = nullptr;
            return QG_ERROR_RUNTIME;
        }
        
        // Start training
        state->is_training = true;
        const size_t batch_size = static_cast<size_t>(state->config[QG_CONFIG_BATCH_SIZE]);
        
        // Convert data to complex format
        const size_t input_dim = static_cast<size_t>(state->config[QG_CONFIG_INPUT_DIM]);
        std::vector<std::vector<ComplexFloat>> complex_data(num_samples);
        std::vector<const ComplexFloat*> data_ptrs(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            complex_data[i].resize(input_dim);
            for (size_t j = 0; j < input_dim; j++) {
                complex_data[i][j] = complex_float_create(data[i * input_dim + j], 0.0f);
            }
            data_ptrs[i] = complex_data[i].data();
        }
        
        // Convert labels to complex format
        std::vector<ComplexFloat> complex_labels(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            complex_labels[i] = complex_float_create(static_cast<float>(labels[i]), 0.0f);
        }
        
        // Train using learning task interface
        if (!quantum_train_task(state->learning_task, 
                              data_ptrs.data(),
                              complex_labels.data(),
                              num_samples)) {
            return QG_ERROR_RUNTIME;
        }
        
        // Get training state
        training_state_t train_state;
        if (quantum_get_training_state(state->learning_task, &train_state)) {
            state->current_epoch = train_state.current_epoch;
            state->current_loss = train_state.current_loss;
            state->is_training = !train_state.converged;
        }
        
        return QG_SUCCESS;
    } catch (const std::exception&) {
        return QG_ERROR_RUNTIME;
    }
}

int quantum_pipeline_evaluate_impl(void* pipeline, const float* data, const int* labels, 
                                 size_t num_samples, float* results) {
    auto state = static_cast<QuantumPipelineState*>(pipeline);
    if (!state || !data || !labels || num_samples == 0 || !results) return QG_ERROR_INVALID_ARGUMENT;
    
    try {
        // Create evaluation task
        quantum_task_t* eval_task = create_task(nullptr, PRIORITY_MEDIUM);
        if (!eval_task) return QG_ERROR_INITIALIZATION;
        
        // Configure resource requirements
        resource_requirement_t req = {
            .type = state->config[QG_CONFIG_USE_GPU] > 0.5f ? RESOURCE_GPU : RESOURCE_CPU,
            .quantity = 1,
            .duration = 0.0,
            .exclusive = false
        };
        
        eval_task->requirements = new resource_requirement_t[1];
        eval_task->requirements[0] = req;
        eval_task->num_requirements = 1;
        
        // Submit task to scheduler
        if (!submit_task(eval_task)) {
            delete[] eval_task->requirements;
            delete eval_task;
            return QG_ERROR_RUNTIME;
        }
        
        // Convert data to complex format
        const size_t input_dim = static_cast<size_t>(state->config[QG_CONFIG_INPUT_DIM]);
        std::vector<std::vector<ComplexFloat>> complex_data(num_samples);
        std::vector<const ComplexFloat*> data_ptrs(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            complex_data[i].resize(input_dim);
            for (size_t j = 0; j < input_dim; j++) {
                complex_data[i][j] = complex_float_create(data[i * input_dim + j], 0.0f);
            }
            data_ptrs[i] = complex_data[i].data();
        }
        
        // Convert labels to complex format
        std::vector<ComplexFloat> complex_labels(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            complex_labels[i] = complex_float_create(static_cast<float>(labels[i]), 0.0f);
        }

        // Evaluate using learning task interface
        task_metrics_t task_metrics;
        if (!quantum_evaluate_task(state->learning_task,
                                 data_ptrs.data(),
                                 complex_labels.data(),
                                 num_samples,
                                 &task_metrics)) {
            return QG_ERROR_RUNTIME;
        }
        
        // Get scheduler metrics for performance data
        scheduler_metrics_t scheduler_metrics;
        if (!get_scheduler_metrics(&scheduler_metrics)) {
            return QG_ERROR_RUNTIME;
        }
        
        // Store results
        results[0] = task_metrics.accuracy;
        results[1] = static_cast<float>(scheduler_metrics.avg_execution_time);
        results[2] = static_cast<float>(scheduler_metrics.total_tasks) / 
                    static_cast<float>(scheduler_metrics.completed_tasks + 1); // Avoid division by zero
        
        return QG_SUCCESS;
    } catch (const std::exception&) {
        return QG_ERROR_RUNTIME;
    }
}

int quantum_pipeline_save_impl(void* pipeline, const char* filename) {
    auto state = static_cast<QuantumPipelineState*>(pipeline);
    if (!state || !filename) return QG_ERROR_INVALID_ARGUMENT;

    try {
        FILE* fp = fopen(filename, "wb");
        if (!fp) return QG_ERROR_RUNTIME;

        // Write header with magic number and version
        const uint32_t MAGIC = 0x51475450; // "QGTP" - Quantum Geometric Tensor Pipeline
        const uint32_t VERSION = 1;
        fwrite(&MAGIC, sizeof(uint32_t), 1, fp);
        fwrite(&VERSION, sizeof(uint32_t), 1, fp);

        // Write configuration
        fwrite(state->config, sizeof(float), QG_CONFIG_SIZE, fp);

        // Write training state
        fwrite(&state->current_epoch, sizeof(size_t), 1, fp);
        fwrite(&state->current_loss, sizeof(float), 1, fp);
        fwrite(&state->current_accuracy, sizeof(float), 1, fp);

        // Save learning task parameters if available
        if (state->learning_task) {
            // Get model parameters from learning task
            size_t num_params = 0;
            float* params = nullptr;

            if (quantum_get_task_parameters(state->learning_task, &params, &num_params)) {
                fwrite(&num_params, sizeof(size_t), 1, fp);
                if (params && num_params > 0) {
                    fwrite(params, sizeof(float), num_params, fp);
                }
                free(params);
            } else {
                num_params = 0;
                fwrite(&num_params, sizeof(size_t), 1, fp);
            }
        } else {
            size_t num_params = 0;
            fwrite(&num_params, sizeof(size_t), 1, fp);
        }

        fclose(fp);
        return QG_SUCCESS;
    } catch (const std::exception&) {
        return QG_ERROR_RUNTIME;
    }
}

void quantum_pipeline_destroy_impl(void* pipeline) {
    auto state = static_cast<QuantumPipelineState*>(pipeline);
    if (!state) return;
    
    try {
        // Stop any running tasks
        if (state->training_task) {
            cancel_task(state->training_task);
            delete[] state->training_task->requirements;
            delete state->training_task;
        }
        
        // State cleanup will happen in destructor
        delete state;
    } catch (const std::exception&) {
        // Best effort cleanup
    }
}

} // extern "C"
