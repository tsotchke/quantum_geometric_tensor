#include "quantum_geometric/hybrid/quantum_machine_learning.h"
#include "quantum_geometric/hybrid/quantum_classical_orchestrator.h"
#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/learning/learning_task.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// OpenMP macros
#ifdef _OPENMP
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define OMP_PARALLEL_FOR_REDUCTION(op,var) _Pragma("omp parallel for reduction(" #op ":" #var ")")
#else
#define OMP_PARALLEL_FOR
#define OMP_PARALLEL_FOR_REDUCTION(op,var)
#endif

// min() is defined in numeric_utils.h (included via quantum_geometric_operations.h)

// QML parameters
#define MAX_EPOCHS 100
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001

// Forward declarations
static quantum_circuit* create_quantum_neural_circuit(size_t num_qubits, const void* layers);
static void initialize_parameters(double* parameters, size_t num_parameters);
static double compute_loss(const double* outputs, const double* targets, size_t output_size, QMLModelType type);
static void adjust_learning_rate(QMLContext* ctx, size_t epoch);
static double forward_pass(QMLContext* ctx, const double* inputs, const double* targets, size_t batch_size);
static void backward_pass(QMLContext* ctx);
static void update_parameters(QMLContext* ctx);
static double* compute_loss_gradients(QMLContext* ctx);
static void classical_backward_pass(ClassicalNetwork* network, double* gradients);
static double* classical_forward_pass(ClassicalNetwork* network, double* input);
static size_t count_classical_parameters(const NetworkArchitecture* architecture);
static void set_learning_rate(OptimizationContext* optimizer, double lr);
static void update_layer_gradients(ClassicalNetwork* network, size_t layer_idx, double* gradients);
static double* apply_layer(ClassicalNetwork* network, size_t layer_idx, double* input);
static void compute_classification_gradients(ClassicalNetwork* network, double* gradients);
static void compute_regression_gradients(ClassicalNetwork* network, double* gradients);
static void compute_reconstruction_gradients(ClassicalNetwork* network, double* gradients);
static void update_classical_parameters(ClassicalNetwork* network, OptimizationContext* optimizer);
// update_quantum_parameters is declared in quantum_circuit_types.h

// Private QMLContext implementation
struct QMLContext {
    QMLModelType type;
    quantum_circuit* quantum_circuit;
    ClassicalNetwork* classical_network;
    OptimizationContext* optimizer;
    size_t num_qubits;
    size_t num_parameters;
    double* parameters;
    double current_loss;
    bool use_gpu;
};

// Initialize QML model
QMLContext* init_qml_model(QMLModelType type,
                          size_t num_qubits,
                          const NetworkArchitecture* architecture) {
    QMLContext* ctx = malloc(sizeof(QMLContext));
    if (!ctx) return NULL;
    
    ctx->type = type;
    ctx->num_qubits = num_qubits;
    ctx->use_gpu = true;  // Default to GPU acceleration
    
    // Create quantum circuit
    ctx->quantum_circuit = create_quantum_neural_circuit(
        num_qubits,
        architecture->quantum_layers);
    
    if (!ctx->quantum_circuit) {
        free(ctx);
        return NULL;
    }
    
    // Create classical network
    ctx->classical_network = create_classical_network(architecture);
    if (!ctx->classical_network) {
        cleanup_quantum_circuit(ctx->quantum_circuit);
        free(ctx);
        return NULL;
    }
    
    // Count total parameters
    ctx->num_parameters = count_quantum_parameters(ctx->quantum_circuit) +
                         count_classical_parameters(architecture);
    
    // Initialize parameters
    ctx->parameters = aligned_alloc(64,
        ctx->num_parameters * sizeof(double));
    if (!ctx->parameters) {
        cleanup_classical_network(ctx->classical_network);
        cleanup_quantum_circuit(ctx->quantum_circuit);
        free(ctx);
        return NULL;
    }
    
    // Initialize optimizer using classical ADAM optimizer
    ctx->optimizer = init_classical_optimizer(
        OPTIMIZER_ADAM,
        ctx->num_parameters,
        ctx->use_gpu);
    
    if (!ctx->optimizer) {
        free(ctx->parameters);
        cleanup_classical_network(ctx->classical_network);
        cleanup_quantum_circuit(ctx->quantum_circuit);
        free(ctx);
        return NULL;
    }
    
    // Initialize parameters randomly
    initialize_parameters(ctx->parameters,
                        ctx->num_parameters);
    
    ctx->current_loss = INFINITY;
    
    return ctx;
}

// Create classical network
ClassicalNetwork* create_classical_network(const NetworkArchitecture* architecture) {
    ClassicalNetwork* network = malloc(sizeof(ClassicalNetwork));
    if (!network) return NULL;

    network->input_size = architecture->input_size;
    network->output_size = architecture->output_size;
    network->num_layers = architecture->num_layers;

    // Allocate and populate layer_sizes array for proper size tracking
    network->layer_sizes = malloc(network->num_layers * sizeof(size_t));
    if (!network->layer_sizes) {
        free(network);
        return NULL;
    }
    for (size_t i = 0; i < network->num_layers; i++) {
        network->layer_sizes[i] = (i == network->num_layers - 1) ?
                                  architecture->output_size : architecture->layer_sizes[i];
    }

    // Allocate weights and biases arrays
    network->weights = malloc(network->num_layers * sizeof(double*));
    network->biases = malloc(network->num_layers * sizeof(double*));
    if (!network->weights || !network->biases) {
        free(network->layer_sizes);
        free(network->weights);
        free(network->biases);
        free(network);
        return NULL;
    }

    // Initialize weights and biases for each layer
    size_t prev_size = network->input_size;
    for (size_t i = 0; i < network->num_layers; i++) {
        size_t curr_size = network->layer_sizes[i];
        
        // Allocate weights
        network->weights[i] = malloc(prev_size * curr_size * sizeof(double));
        if (!network->weights[i]) {
            for (size_t j = 0; j < i; j++) {
                free(network->weights[j]);
                free(network->biases[j]);
            }
            free(network->weights);
            free(network->biases);
            free(network);
            return NULL;
        }
        
        // Allocate biases
        network->biases[i] = malloc(curr_size * sizeof(double));
        if (!network->biases[i]) {
            free(network->weights[i]);
            for (size_t j = 0; j < i; j++) {
                free(network->weights[j]);
                free(network->biases[j]);
            }
            free(network->weights);
            free(network->biases);
            free(network);
            return NULL;
        }
        
        prev_size = curr_size;
    }

    // Allocate and initialize activation functions for each layer
    network->activation_functions = malloc(network->num_layers * sizeof(ActivationType));
    if (!network->activation_functions) {
        for (size_t j = 0; j < network->num_layers; j++) {
            free(network->weights[j]);
            free(network->biases[j]);
        }
        free(network->weights);
        free(network->biases);
        free(network->layer_sizes);
        free(network);
        return NULL;
    }

    // Set default activations: ReLU for hidden layers, linear for output layer
    for (size_t i = 0; i < network->num_layers; i++) {
        if (i < network->num_layers - 1) {
            network->activation_functions[i] = ACTIVATION_RELU;
        } else {
            network->activation_functions[i] = ACTIVATION_NONE;  // Linear output
        }
    }

    // Allocate activation cache for backpropagation
    network->activations = malloc(network->num_layers * sizeof(double*));
    if (!network->activations) {
        for (size_t j = 0; j < network->num_layers; j++) {
            free(network->weights[j]);
            free(network->biases[j]);
        }
        free(network->weights);
        free(network->biases);
        free(network->layer_sizes);
        free(network->activation_functions);
        free(network);
        return NULL;
    }
    for (size_t i = 0; i < network->num_layers; i++) {
        network->activations[i] = malloc(network->layer_sizes[i] * sizeof(double));
        if (!network->activations[i]) {
            for (size_t j = 0; j < i; j++) {
                free(network->activations[j]);
            }
            for (size_t j = 0; j < network->num_layers; j++) {
                free(network->weights[j]);
                free(network->biases[j]);
            }
            free(network->activations);
            free(network->weights);
            free(network->biases);
            free(network->layer_sizes);
            free(network->activation_functions);
            free(network);
            return NULL;
        }
    }

    // Allocate space for caching input
    network->last_input = malloc(network->input_size * sizeof(double));
    if (!network->last_input) {
        for (size_t j = 0; j < network->num_layers; j++) {
            free(network->activations[j]);
            free(network->weights[j]);
            free(network->biases[j]);
        }
        free(network->activations);
        free(network->weights);
        free(network->biases);
        free(network->layer_sizes);
        free(network->activation_functions);
        free(network);
        return NULL;
    }

    return network;
}

// Clean up classical network
void cleanup_classical_network(ClassicalNetwork* network) {
    if (!network) return;

    if (network->weights) {
        for (size_t i = 0; i < network->num_layers; i++) {
            free(network->weights[i]);
        }
        free(network->weights);
    }

    if (network->biases) {
        for (size_t i = 0; i < network->num_layers; i++) {
            free(network->biases[i]);
        }
        free(network->biases);
    }

    if (network->activations) {
        for (size_t i = 0; i < network->num_layers; i++) {
            free(network->activations[i]);
        }
        free(network->activations);
    }

    free(network->last_input);
    free(network->layer_sizes);
    free(network->activation_functions);
    free(network);
}

// Count classical parameters
static size_t count_classical_parameters(const NetworkArchitecture* architecture) {
    size_t count = 0;
    size_t prev_size = architecture->input_size;
    
    // Count weights and biases for each layer
    for (size_t i = 0; i < architecture->num_layers; i++) {
        size_t curr_size = (i == architecture->num_layers - 1) ?
                          architecture->output_size : architecture->layer_sizes[i];
        
        count += prev_size * curr_size;  // Weights
        count += curr_size;              // Biases
        
        prev_size = curr_size;
    }
    
    return count;
}

// Train QML model
int train_qml_model(QMLContext* ctx,
                   const DataSet* training_data,
                   const DataSet* validation_data,
                   TrainingConfig* config) {
    if (!ctx || !training_data || !config) return -1;
    
    size_t num_batches = (training_data->size + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Training loop
    for (size_t epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        
        // Process mini-batches
        OMP_PARALLEL_FOR_REDUCTION(+,epoch_loss)
        for (size_t batch = 0; batch < num_batches; batch++) {
            // Get batch data
            size_t start_idx = batch * BATCH_SIZE;
            size_t end_idx = min(start_idx + BATCH_SIZE,
                               training_data->size);
            
            // Forward pass
            double batch_loss = forward_pass(ctx,
                training_data->inputs + start_idx,
                training_data->targets + start_idx,
                end_idx - start_idx);
            
            // Backward pass
            backward_pass(ctx);
            
            // Update parameters
            update_parameters(ctx);
            
            epoch_loss += batch_loss;
        }
        
        epoch_loss /= num_batches;
        ctx->current_loss = epoch_loss;
        
        // Validation
        if (validation_data) {
            double val_loss = evaluate_model(ctx,
                validation_data);
            
            // Early stopping
            if (check_early_stopping(val_loss, config)) {
                break;
            }
        }
        
        // Learning rate scheduling
        adjust_learning_rate(ctx, epoch);
    }
    
    return 0;
}

// Forward pass
static double forward_pass(QMLContext* ctx,
                         const double* inputs,
                         const double* targets,
                         size_t batch_size) {
    double batch_loss = 0.0;
    
    // Process each sample in batch
    for (size_t i = 0; i < batch_size; i++) {
        // Quantum forward pass
        QuantumState* quantum_state = encode_input(
            inputs + i * ctx->num_qubits,
            (const struct QuantumCircuit*)ctx->quantum_circuit);

        apply_quantum_layers(ctx->quantum_circuit,
                           quantum_state);

        double* quantum_output = measure_quantum_state(
            quantum_state);
        
        // Classical forward pass
        double* final_output = classical_forward_pass(
            ctx->classical_network,
            quantum_output);
        
        // Compute loss
        batch_loss += compute_loss(final_output,
            targets + i * ctx->classical_network->output_size,
            ctx->classical_network->output_size,
            ctx->type);
        
        free(quantum_output);
        free(final_output);
        cleanup_quantum_state(quantum_state);
    }
    
    return batch_loss / batch_size;
}

// Backward pass
static void backward_pass(QMLContext* ctx) {
    // Compute gradients
    double* gradients = compute_loss_gradients(ctx);
    
    // Classical backward pass
    classical_backward_pass(ctx->classical_network,
                          gradients);
    
    // Quantum backward pass
    quantum_backward_pass(ctx->quantum_circuit,
                        gradients);
    
    free(gradients);
}

// Update parameters
static void update_parameters(QMLContext* ctx) {
    // Update classical parameters
    update_classical_parameters(ctx->classical_network,
                              ctx->optimizer);
    
    // Update quantum parameters
    update_quantum_parameters(ctx->quantum_circuit,
                            ctx->optimizer);
}

// Helper functions
static quantum_circuit* create_quantum_neural_circuit(
    size_t num_qubits,
    const void* layers) {
    
    const QuantumLayerConfig* quantum_layers = (const QuantumLayerConfig*)layers;
    quantum_circuit* circuit = init_quantum_circuit(num_qubits);
    if (!circuit) return NULL;
    
    // Add quantum layers
    for (size_t i = 0; i < quantum_layers->num_layers; i++) {
        switch (quantum_layers->types[i]) {
            case LAYER_QUANTUM_CONV:
                add_quantum_conv_layer(circuit,
                    quantum_layers->params[i]);
                break;
                
            case LAYER_QUANTUM_POOL:
                add_quantum_pool_layer(circuit,
                    quantum_layers->params[i]);
                break;
                
            case LAYER_QUANTUM_DENSE:
                add_quantum_dense_layer(circuit,
                    quantum_layers->params[i]);
                break;

            case LAYER_QUANTUM_GATE:
            case LAYER_QUANTUM_MEASURE:
            default:
                // These layer types are handled elsewhere
                break;
        }
    }

    return circuit;
}

static void initialize_parameters(double* parameters,
                               size_t num_parameters) {
    // Xavier initialization
    double scale = sqrt(2.0 / num_parameters);
    
    OMP_PARALLEL_FOR
    for (size_t i = 0; i < num_parameters; i++) {
        parameters[i] = ((double)rand() / RAND_MAX - 0.5) * scale;
    }
}

static double compute_loss(const double* outputs,
                         const double* targets,
                         size_t output_size,
                         QMLModelType type) {
    switch (type) {
        case QML_CLASSIFIER:
            return compute_cross_entropy_loss(outputs, targets, output_size);

        case QML_REGRESSOR:
            return compute_mse_loss(outputs, targets, output_size);

        case QML_AUTOENCODER:
            return compute_reconstruction_loss(outputs, targets, output_size);

        default:
            return INFINITY;
    }
}

static void adjust_learning_rate(QMLContext* ctx,
                               size_t epoch) {
    // Cosine annealing
    double progress = (double)epoch / MAX_EPOCHS;
    double new_lr = LEARNING_RATE *
        (1.0 + cos(M_PI * progress)) * 0.5;
    
    set_learning_rate(ctx->optimizer, new_lr);
}

// Additional helper function implementations
static double* compute_loss_gradients(QMLContext* ctx) {
    double* gradients = calloc(ctx->num_parameters, sizeof(double));
    if (!gradients) return NULL;
    
    // Compute gradients based on loss type
    switch (ctx->type) {
        case QML_CLASSIFIER:
            compute_classification_gradients(ctx->classical_network, gradients);
            break;
            
        case QML_REGRESSOR:
            compute_regression_gradients(ctx->classical_network, gradients);
            break;
            
        case QML_AUTOENCODER:
            compute_reconstruction_gradients(ctx->classical_network, gradients);
            break;
    }
    
    return gradients;
}

static void classical_backward_pass(ClassicalNetwork* network, double* gradients) {
    // Backpropagate through classical layers
    for (size_t i = network->num_layers; i > 0; i--) {
        update_layer_gradients(network, i - 1, gradients);
    }
}

static double* classical_forward_pass(ClassicalNetwork* network, double* input) {
    double* current = input;
    double* next = NULL;
    
    // Forward through classical layers
    for (size_t i = 0; i < network->num_layers; i++) {
        next = apply_layer(network, i, current);
        if (i > 0) free(current);
        current = next;
    }
    
    return current;
}

static void set_learning_rate(OptimizationContext* optimizer, double lr) {
    if (optimizer) {
        optimizer->learning_rate = lr;
    }
}

// Clean up QML model
void cleanup_qml_model(QMLContext* ctx) {
    if (!ctx) return;
    
    cleanup_quantum_circuit(ctx->quantum_circuit);
    cleanup_classical_network(ctx->classical_network);
    cleanup_classical_optimizer(ctx->optimizer);
    free(ctx->parameters);
    free(ctx);
}

// ============================================================================
// Production implementations of neural network helper functions
// ============================================================================

// ReLU activation function
static double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// ReLU derivative
static double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Sigmoid activation function
static double sigmoid_activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid derivative
static double sigmoid_derivative(double x) {
    double s = sigmoid_activation(x);
    return s * (1.0 - s);
}

// Tanh activation function
static double tanh_activation(double x) {
    return tanh(x);
}

// Tanh derivative
static double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

// Softmax for classification output (in-place on output array)
static void softmax(double* output, size_t size) {
    double max_val = output[0];
    for (size_t i = 1; i < size; i++) {
        if (output[i] > max_val) max_val = output[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        output[i] = exp(output[i] - max_val);
        sum += output[i];
    }

    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Apply activation function based on type
static double apply_activation(double x, ActivationType activation_type) {
    switch (activation_type) {
        case ACTIVATION_RELU:
            return relu(x);
        case ACTIVATION_SIGMOID:
            return sigmoid_activation(x);
        case ACTIVATION_TANH:
            return tanh_activation(x);
        case ACTIVATION_NONE:
        case ACTIVATION_SOFTMAX:  // Softmax is applied separately to entire layer
        default:
            return x;  // Linear (no activation)
    }
}

// Get derivative of activation function
static double activation_derivative(double x, ActivationType activation_type) {
    switch (activation_type) {
        case ACTIVATION_RELU:
            return relu_derivative(x);
        case ACTIVATION_SIGMOID:
            return sigmoid_derivative(x);
        case ACTIVATION_TANH:
            return tanh_derivative(x);
        case ACTIVATION_NONE:
        case ACTIVATION_SOFTMAX:
        default:
            return 1.0;  // Linear derivative
    }
}

// Apply a single layer forward pass
// Returns newly allocated output array (caller must free)
static double* apply_layer(ClassicalNetwork* network, size_t layer_idx, double* input) {
    if (!network || !input || layer_idx >= network->num_layers) {
        return NULL;
    }

    // Determine input and output sizes using the stored layer_sizes
    size_t input_size;
    size_t output_size;

    if (layer_idx == 0) {
        input_size = network->input_size;
    } else {
        // Previous layer's output size from stored dimensions
        input_size = network->layer_sizes[layer_idx - 1];
    }

    // Output size from stored layer dimensions
    output_size = network->layer_sizes[layer_idx];

    // Allocate output
    double* output = aligned_alloc(64, output_size * sizeof(double));
    if (!output) return NULL;

    // Compute output = weights * input + bias
    double* weights = network->weights[layer_idx];
    double* bias = network->biases[layer_idx];

    // Get the configured activation type for this layer
    ActivationType activation = network->activation_functions ?
                                network->activation_functions[layer_idx] : ACTIVATION_RELU;

    OMP_PARALLEL_FOR
    for (size_t i = 0; i < output_size; i++) {
        double sum = bias[i];
        for (size_t j = 0; j < input_size; j++) {
            sum += weights[j * output_size + i] * input[j];
        }

        // Apply configured activation function
        output[i] = apply_activation(sum, activation);
    }

    // Apply softmax if specified (needs entire layer output)
    if (activation == ACTIVATION_SOFTMAX) {
        softmax(output, output_size);
    }

    // Cache activations for backpropagation
    if (network->activations && network->activations[layer_idx]) {
        memcpy(network->activations[layer_idx], output, output_size * sizeof(double));
    }

    // Cache input for first layer (needed for gradient computation)
    if (layer_idx == 0 && network->last_input) {
        memcpy(network->last_input, input, input_size * sizeof(double));
    }

    return output;
}

// Update gradients for a specific layer (backpropagation)
static void update_layer_gradients(ClassicalNetwork* network, size_t layer_idx, double* gradients) {
    if (!network || !gradients || layer_idx >= network->num_layers) {
        return;
    }

    // Get layer dimensions using stored layer sizes
    size_t input_size;
    size_t output_size;

    if (layer_idx == 0) {
        input_size = network->input_size;
    } else {
        input_size = network->layer_sizes[layer_idx - 1];
    }
    output_size = network->layer_sizes[layer_idx];

    double* weights = network->weights[layer_idx];
    double learning_rate = 0.001;  // Default learning rate

    // Get the input activations for this layer
    // For layer 0, use cached input; for other layers, use previous layer's cached activations
    double* layer_input = NULL;
    if (layer_idx == 0) {
        layer_input = network->last_input;
    } else if (network->activations && network->activations[layer_idx - 1]) {
        layer_input = network->activations[layer_idx - 1];
    }

    // Update weights using gradient descent: dW = input^T * gradient
    if (layer_input) {
        OMP_PARALLEL_FOR
        for (size_t i = 0; i < input_size; i++) {
            for (size_t j = 0; j < output_size; j++) {
                // Weight gradient = input_activation * output_gradient
                weights[i * output_size + j] -= learning_rate * gradients[j] * layer_input[i];
            }
        }
    } else {
        // Fallback if activations not cached (shouldn't happen in normal operation)
        OMP_PARALLEL_FOR
        for (size_t i = 0; i < input_size; i++) {
            for (size_t j = 0; j < output_size; j++) {
                weights[i * output_size + j] -= learning_rate * gradients[j];
            }
        }
    }

    // Update biases: dB = gradient
    double* bias = network->biases[layer_idx];
    for (size_t i = 0; i < output_size; i++) {
        bias[i] -= learning_rate * gradients[i];
    }
}

// Compute gradients for classification (cross-entropy loss)
static void compute_classification_gradients(ClassicalNetwork* network, double* gradients) {
    if (!network || !gradients) return;

    size_t output_size = network->output_size;

    // For cross-entropy loss with softmax output:
    // gradient = predicted - target
    // This assumes gradients already contains (predicted - target) after loss computation

    // Apply softmax derivative (for softmax + cross-entropy, gradient is already correct)
    // Scale by batch size for proper gradient averaging
    double scale = 1.0;

    OMP_PARALLEL_FOR
    for (size_t i = 0; i < output_size; i++) {
        gradients[i] *= scale;
    }

    // Backpropagate through layers (from output to input)
    for (size_t layer = network->num_layers; layer > 0; layer--) {
        size_t layer_idx = layer - 1;

        // Get layer dimensions
        size_t curr_size = (layer_idx == network->num_layers - 1) ?
                          network->output_size : network->input_size;
        size_t prev_size = (layer_idx == 0) ?
                          network->input_size : network->input_size;

        // Compute gradient for previous layer
        double* prev_gradients = aligned_alloc(64, prev_size * sizeof(double));
        if (!prev_gradients) return;

        memset(prev_gradients, 0, prev_size * sizeof(double));

        double* weights = network->weights[layer_idx];

        // Get the configured activation type for the previous layer
        ActivationType prev_activation = (layer_idx > 0 && network->activation_functions) ?
                                         network->activation_functions[layer_idx - 1] : ACTIVATION_RELU;

        OMP_PARALLEL_FOR
        for (size_t i = 0; i < prev_size; i++) {
            for (size_t j = 0; j < curr_size; j++) {
                prev_gradients[i] += weights[i * curr_size + j] * gradients[j];
            }
            // Apply activation derivative for hidden layers
            if (layer_idx > 0) {
                prev_gradients[i] *= activation_derivative(prev_gradients[i], prev_activation);
            }
        }

        // Update layer parameters
        update_layer_gradients(network, layer_idx, gradients);

        // Move to previous layer
        if (layer_idx > 0) {
            memcpy(gradients, prev_gradients, prev_size * sizeof(double));
        }

        free(prev_gradients);
    }
}

// Compute gradients for regression (MSE loss)
static void compute_regression_gradients(ClassicalNetwork* network, double* gradients) {
    if (!network || !gradients) return;

    size_t output_size = network->output_size;

    // For MSE loss: gradient = 2 * (predicted - target) / n
    // Assuming gradients contains (predicted - target) after loss computation
    double scale = 2.0 / (double)output_size;

    OMP_PARALLEL_FOR
    for (size_t i = 0; i < output_size; i++) {
        gradients[i] *= scale;
    }

    // Backpropagate through layers
    for (size_t layer = network->num_layers; layer > 0; layer--) {
        size_t layer_idx = layer - 1;

        size_t curr_size = (layer_idx == network->num_layers - 1) ?
                          network->output_size : network->input_size;
        size_t prev_size = (layer_idx == 0) ?
                          network->input_size : network->input_size;

        double* prev_gradients = aligned_alloc(64, prev_size * sizeof(double));
        if (!prev_gradients) return;

        memset(prev_gradients, 0, prev_size * sizeof(double));

        double* weights = network->weights[layer_idx];

        // Get the configured activation type for the previous layer
        ActivationType prev_activation = (layer_idx > 0 && network->activation_functions) ?
                                         network->activation_functions[layer_idx - 1] : ACTIVATION_RELU;

        OMP_PARALLEL_FOR
        for (size_t i = 0; i < prev_size; i++) {
            for (size_t j = 0; j < curr_size; j++) {
                prev_gradients[i] += weights[i * curr_size + j] * gradients[j];
            }
            if (layer_idx > 0) {
                prev_gradients[i] *= activation_derivative(prev_gradients[i], prev_activation);
            }
        }

        update_layer_gradients(network, layer_idx, gradients);

        if (layer_idx > 0) {
            memcpy(gradients, prev_gradients, prev_size * sizeof(double));
        }

        free(prev_gradients);
    }
}

// Compute gradients for reconstruction (autoencoder, VAE)
static void compute_reconstruction_gradients(ClassicalNetwork* network, double* gradients) {
    if (!network || !gradients) return;

    size_t output_size = network->output_size;

    // For reconstruction loss (typically MSE or binary cross-entropy):
    // gradient = predicted - target for MSE
    // gradient = predicted - target for BCE with sigmoid output

    // Scale gradients
    double scale = 1.0 / (double)output_size;

    OMP_PARALLEL_FOR
    for (size_t i = 0; i < output_size; i++) {
        gradients[i] *= scale;
    }

    // Backpropagate through decoder layers first, then encoder
    // For simplicity, treating as a single network (symmetric autoencoder)
    for (size_t layer = network->num_layers; layer > 0; layer--) {
        size_t layer_idx = layer - 1;

        size_t curr_size = (layer_idx == network->num_layers - 1) ?
                          network->output_size : network->input_size;
        size_t prev_size = (layer_idx == 0) ?
                          network->input_size : network->input_size;

        double* prev_gradients = aligned_alloc(64, prev_size * sizeof(double));
        if (!prev_gradients) return;

        memset(prev_gradients, 0, prev_size * sizeof(double));

        double* weights = network->weights[layer_idx];

        // Get the configured activation type for the previous layer
        // Default to SIGMOID for reconstruction networks if not specified
        ActivationType prev_activation = (layer_idx > 0 && network->activation_functions) ?
                                         network->activation_functions[layer_idx - 1] : ACTIVATION_SIGMOID;

        // Compute gradient w.r.t. previous layer
        OMP_PARALLEL_FOR
        for (size_t i = 0; i < prev_size; i++) {
            for (size_t j = 0; j < curr_size; j++) {
                prev_gradients[i] += weights[i * curr_size + j] * gradients[j];
            }

            // Apply configured activation derivative
            if (layer_idx > 0) {
                prev_gradients[i] *= activation_derivative(prev_gradients[i], prev_activation);
            }
        }

        // Update layer parameters
        update_layer_gradients(network, layer_idx, gradients);

        if (layer_idx > 0) {
            memcpy(gradients, prev_gradients, prev_size * sizeof(double));
        }

        free(prev_gradients);
    }
}

// Update classical network parameters using optimizer
static void update_classical_parameters(ClassicalNetwork* network, OptimizationContext* optimizer) {
    if (!network || !optimizer) return;

    size_t param_offset = 0;

    // Apply optimizer update to each layer's parameters
    for (size_t layer = 0; layer < network->num_layers; layer++) {
        if (network->weights && network->weights[layer]) {
            size_t layer_size = network->layer_sizes[layer];
            size_t prev_size = (layer == 0) ? network->input_size
                                            : network->layer_sizes[layer - 1];

            // Apply Adam update to weights
            for (size_t i = 0; i < layer_size * prev_size; i++) {
                size_t idx = param_offset + i;
                if (optimizer->gradients && idx < optimizer->num_parameters) {
                    double grad = optimizer->gradients[idx];

                    // Adam optimizer update
                    if (optimizer->momentum) {
                        optimizer->momentum[idx] = optimizer->beta1 * optimizer->momentum[idx]
                                                 + (1.0 - optimizer->beta1) * grad;
                    }
                    if (optimizer->velocity) {
                        optimizer->velocity[idx] = optimizer->beta2 * optimizer->velocity[idx]
                                                 + (1.0 - optimizer->beta2) * grad * grad;
                    }

                    double m_hat = optimizer->momentum ? optimizer->momentum[idx] / (1.0 - optimizer->beta1) : grad;
                    double v_hat = optimizer->velocity ? optimizer->velocity[idx] / (1.0 - optimizer->beta2) : 1.0;

                    network->weights[layer][i] -= optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon);
                }
            }
            param_offset += layer_size * prev_size;
        }
    }
}

// Implementation of update_quantum_parameters declared in quantum_circuit_types.h
void update_quantum_parameters(quantum_circuit* circuit, void* optimizer_ptr) {
    if (!circuit || !optimizer_ptr) return;

    OptimizationContext* optimizer = (OptimizationContext*)optimizer_ptr;

    // Update parameters stored in the circuit's gates
    // Each gate may have rotation parameters that need to be optimized
    if (circuit->num_gates > 0 && circuit->gates && optimizer->gradients) {
        size_t param_idx = 0;
        for (size_t g = 0; g < circuit->num_gates && param_idx < optimizer->num_parameters; g++) {
            quantum_gate_t* gate = circuit->gates[g];
            if (gate && gate->num_parameters > 0 && gate->parameters) {
                for (size_t p = 0; p < gate->num_parameters && param_idx < optimizer->num_parameters; p++) {
                    double grad = optimizer->gradients[param_idx];

                    // Apply Adam-style update
                    double m_hat = grad;
                    double v_hat = 1.0;

                    if (optimizer->momentum) {
                        optimizer->momentum[param_idx] = optimizer->beta1 * optimizer->momentum[param_idx]
                                                       + (1.0 - optimizer->beta1) * grad;
                        m_hat = optimizer->momentum[param_idx] / (1.0 - optimizer->beta1);
                    }
                    if (optimizer->velocity) {
                        optimizer->velocity[param_idx] = optimizer->beta2 * optimizer->velocity[param_idx]
                                                       + (1.0 - optimizer->beta2) * grad * grad;
                        v_hat = optimizer->velocity[param_idx] / (1.0 - optimizer->beta2);
                    }

                    gate->parameters[p] -= optimizer->learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon);
                    param_idx++;
                }
            }
        }
    }
}

// =============================================================================
// Loss Functions
// =============================================================================

/**
 * @brief Compute cross-entropy loss for classification
 *
 * Computes the categorical cross-entropy loss between predicted probabilities
 * and target one-hot encoded labels.
 *
 * L = -sum(target_i * log(output_i))
 *
 * @param outputs Predicted probabilities (should sum to 1 after softmax)
 * @param targets One-hot encoded target labels
 * @param size Number of classes/output dimensions
 * @return Cross-entropy loss value
 */
double compute_cross_entropy_loss(const double* outputs,
                                 const double* targets,
                                 size_t size) {
    if (!outputs || !targets || size == 0) return INFINITY;

    double loss = 0.0;
    const double epsilon = 1e-15;  // Numerical stability to prevent log(0)

    // Accumulate cross-entropy loss across all classes
    for (size_t i = 0; i < size; i++) {
        // Clamp outputs to prevent log(0) and log(1) numerical issues
        double clamped = outputs[i];
        if (clamped < epsilon) clamped = epsilon;
        if (clamped > 1.0 - epsilon) clamped = 1.0 - epsilon;

        // Cross-entropy: -sum(target * log(output))
        // Only non-zero targets contribute (one-hot encoding)
        if (targets[i] > 0.0) {
            loss -= targets[i] * log(clamped);
        }
    }

    return loss;
}

/**
 * @brief Compute Mean Squared Error loss for regression
 *
 * Computes the mean of squared differences between outputs and targets.
 *
 * L = (1/n) * sum((output_i - target_i)^2)
 *
 * @param outputs Predicted values
 * @param targets Target values
 * @param size Number of output dimensions
 * @return MSE loss value
 */
double compute_mse_loss(const double* outputs,
                       const double* targets,
                       size_t size) {
    if (!outputs || !targets || size == 0) return INFINITY;

    double loss = 0.0;

    // Sum squared differences
    for (size_t i = 0; i < size; i++) {
        double diff = outputs[i] - targets[i];
        loss += diff * diff;
    }

    // Return mean
    return loss / (double)size;
}

/**
 * @brief Compute reconstruction loss for autoencoders
 *
 * Uses binary cross-entropy suitable for reconstructing normalized inputs
 * in the [0, 1] range (e.g., normalized images).
 *
 * L = -(1/n) * sum(target_i * log(output_i) + (1 - target_i) * log(1 - output_i))
 *
 * @param outputs Reconstructed outputs (should be in [0, 1])
 * @param targets Original input targets (should be in [0, 1])
 * @param size Number of dimensions
 * @return Reconstruction loss value
 */
double compute_reconstruction_loss(const double* outputs,
                                  const double* targets,
                                  size_t size) {
    if (!outputs || !targets || size == 0) return INFINITY;

    double loss = 0.0;
    const double epsilon = 1e-15;  // Numerical stability

    for (size_t i = 0; i < size; i++) {
        // Clamp outputs to valid probability range
        double out = outputs[i];
        if (out < epsilon) out = epsilon;
        if (out > 1.0 - epsilon) out = 1.0 - epsilon;

        // Clamp targets similarly for numerical stability
        double target = targets[i];
        if (target < 0.0) target = 0.0;
        if (target > 1.0) target = 1.0;

        // Binary cross-entropy per element
        loss -= target * log(out) + (1.0 - target) * log(1.0 - out);
    }

    // Return mean loss
    return loss / (double)size;
}

// =============================================================================
// Early Stopping
// =============================================================================

// State for early stopping
static struct {
    double best_loss;
    size_t patience_counter;
    bool initialized;
} early_stopping_state = { INFINITY, 0, false };

bool check_early_stopping(double validation_loss, const TrainingConfig* config) {
    if (!config) return false;

    // Initialize on first call
    if (!early_stopping_state.initialized) {
        early_stopping_state.best_loss = INFINITY;
        early_stopping_state.patience_counter = 0;
        early_stopping_state.initialized = true;
    }

    // Check if validation loss improved
    double improvement = early_stopping_state.best_loss - validation_loss;

    if (improvement > config->early_stopping_threshold) {
        // Loss improved significantly
        early_stopping_state.best_loss = validation_loss;
        early_stopping_state.patience_counter = 0;
        return false;  // Continue training
    } else {
        // Loss did not improve enough
        early_stopping_state.patience_counter++;

        if (early_stopping_state.patience_counter >= config->patience) {
            // Reset state for next training run
            early_stopping_state.initialized = false;
            return true;  // Stop training
        }
        return false;  // Continue training
    }
}

// =============================================================================
// Model Evaluation
// =============================================================================

double evaluate_model(const QMLContext* ctx, const DataSet* data) {
    if (!ctx || !data) return INFINITY;

    double total_loss = 0.0;
    size_t num_samples = data->size;

    // Evaluate on all samples
    for (size_t i = 0; i < num_samples; i++) {
        // Get input and target pointers for this sample
        const double* input = data->inputs + i * data->input_dim;
        const double* target = data->targets + i * data->target_dim;

        // Forward pass through quantum circuit
        QuantumState* quantum_state = encode_input(input, (const struct QuantumCircuit*)ctx->quantum_circuit);
        if (!quantum_state) continue;

        apply_quantum_layers(ctx->quantum_circuit, quantum_state);

        double* quantum_output = measure_quantum_state(quantum_state);
        if (!quantum_output) {
            cleanup_quantum_state(quantum_state);
            continue;
        }

        // Forward pass through classical network
        double* final_output = classical_forward_pass(ctx->classical_network, quantum_output);
        if (!final_output) {
            free(quantum_output);
            cleanup_quantum_state(quantum_state);
            continue;
        }

        // Compute loss based on model type
        double sample_loss = 0.0;
        switch (ctx->type) {
            case QML_CLASSIFIER:
                sample_loss = compute_cross_entropy_loss(final_output, target, data->target_dim);
                break;
            case QML_REGRESSOR:
                sample_loss = compute_mse_loss(final_output, target, data->target_dim);
                break;
            case QML_AUTOENCODER:
                sample_loss = compute_reconstruction_loss(final_output, target, data->target_dim);
                break;
        }

        total_loss += sample_loss;

        // Cleanup
        free(final_output);
        free(quantum_output);
        cleanup_quantum_state(quantum_state);
    }

    return total_loss / (double)num_samples;
}
