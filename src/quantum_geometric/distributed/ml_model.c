/**
 * @file ml_model.c
 * @brief Neural network-based ML model for bottleneck prediction
 *
 * Implements a multi-layer perceptron for bottleneck type prediction
 * with momentum-based gradient descent, adaptive learning rates,
 * and online learning support.
 */

#include "quantum_geometric/distributed/ml_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Network architecture
#define NUM_CLASSES 6
#define HIDDEN_SIZE 32
#define INPUT_SIZE MAX_METRICS

// Training hyperparameters
#define MOMENTUM 0.9
#define BETA1 0.9           // Adam beta1
#define BETA2 0.999         // Adam beta2
#define EPSILON 1e-8        // Numerical stability
#define DROPOUT_RATE 0.2    // Dropout for regularization
#define GRADIENT_CLIP 5.0   // Gradient clipping threshold

// Internal layer structure
typedef struct {
    double* weights;           // Weight matrix
    double* biases;           // Bias vector
    double* weight_momentum;  // Momentum for weights
    double* bias_momentum;    // Momentum for biases
    double* weight_velocity;  // Adam velocity for weights
    double* bias_velocity;    // Adam velocity for biases
    double* activations;      // Layer activations
    double* gradients;        // Gradient storage
    size_t input_size;
    size_t output_size;
} Layer;

// Extended ML model with hidden layers
typedef struct {
    // Network layers
    Layer hidden_layer;
    Layer output_layer;

    // Training state
    size_t training_samples;
    double learning_rate;
    double current_loss;
    size_t update_step;

    // Feature statistics for normalization
    double* feature_mean;
    double* feature_std;
    size_t normalization_samples;

    // Performance tracking
    double accuracy_sum;
    size_t accuracy_count;

    // Dropout mask
    double* dropout_mask;
    bool training_mode;
} MLModelInternal;

// Map external MLModel to internal representation
static MLModelInternal* internal_models[256] = {NULL};
static size_t next_model_id = 0;

// Forward declarations
static void layer_init(Layer* layer, size_t input_size, size_t output_size);
static void layer_cleanup(Layer* layer);
static void layer_forward(Layer* layer, const double* input, bool apply_relu);
static void layer_backward(Layer* layer, const double* input,
                          const double* output_grad, double* input_grad,
                          double learning_rate, size_t step);
static double relu(double x);
static double relu_derivative(double x);
static void softmax(double* output, size_t n);
static void apply_dropout(double* activations, double* mask, size_t n, double rate);
static double clip_gradient(double grad);
static void xavier_init(double* weights, size_t fan_in, size_t fan_out);

// Initialize ML model
MLModel* init_ml_model(void) {
    MLModel* model = calloc(1, sizeof(MLModel));
    if (!model) return NULL;

    MLModelInternal* internal = calloc(1, sizeof(MLModelInternal));
    if (!internal) {
        free(model);
        return NULL;
    }

    // Initialize network layers
    layer_init(&internal->hidden_layer, INPUT_SIZE, HIDDEN_SIZE);
    layer_init(&internal->output_layer, HIDDEN_SIZE, NUM_CLASSES);

    // Initialize feature normalization
    internal->feature_mean = calloc(INPUT_SIZE, sizeof(double));
    internal->feature_std = calloc(INPUT_SIZE, sizeof(double));
    internal->dropout_mask = calloc(HIDDEN_SIZE, sizeof(double));

    if (!internal->feature_mean || !internal->feature_std || !internal->dropout_mask) {
        layer_cleanup(&internal->hidden_layer);
        layer_cleanup(&internal->output_layer);
        free(internal->feature_mean);
        free(internal->feature_std);
        free(internal->dropout_mask);
        free(internal);
        free(model);
        return NULL;
    }

    // Initialize standard deviations to 1.0
    for (size_t i = 0; i < INPUT_SIZE; i++) {
        internal->feature_std[i] = 1.0;
    }

    // Set training parameters
    internal->learning_rate = ML_LEARNING_RATE;
    internal->training_samples = 0;
    internal->update_step = 0;
    internal->normalization_samples = 0;
    internal->current_loss = 0.0;
    internal->accuracy_sum = 0.0;
    internal->accuracy_count = 0;
    internal->training_mode = true;

    // Store internal model reference
    size_t model_id = next_model_id++;
    if (model_id < 256) {
        internal_models[model_id] = internal;
    }

    // Setup external model fields
    model->num_features = INPUT_SIZE;
    model->learning_rate = ML_LEARNING_RATE;
    model->training_samples = 0;
    model->bias = (double)model_id;  // Store model ID in bias field
    model->weights = calloc(INPUT_SIZE, sizeof(double));  // For feature importance

    if (!model->weights) {
        layer_cleanup(&internal->hidden_layer);
        layer_cleanup(&internal->output_layer);
        free(internal->feature_mean);
        free(internal->feature_std);
        free(internal->dropout_mask);
        free(internal);
        free(model);
        return NULL;
    }

    return model;
}

// Initialize a layer
static void layer_init(Layer* layer, size_t input_size, size_t output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    size_t weight_count = input_size * output_size;

    layer->weights = calloc(weight_count, sizeof(double));
    layer->biases = calloc(output_size, sizeof(double));
    layer->weight_momentum = calloc(weight_count, sizeof(double));
    layer->bias_momentum = calloc(output_size, sizeof(double));
    layer->weight_velocity = calloc(weight_count, sizeof(double));
    layer->bias_velocity = calloc(output_size, sizeof(double));
    layer->activations = calloc(output_size, sizeof(double));
    layer->gradients = calloc(output_size, sizeof(double));

    // Xavier initialization
    xavier_init(layer->weights, input_size, output_size);
}

// Xavier/Glorot weight initialization
static void xavier_init(double* weights, size_t fan_in, size_t fan_out) {
    double scale = sqrt(2.0 / (fan_in + fan_out));
    for (size_t i = 0; i < fan_in * fan_out; i++) {
        // Box-Muller transform for normal distribution
        double u1 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        weights[i] = z * scale;
    }
}

// Cleanup a layer
static void layer_cleanup(Layer* layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->weight_momentum);
    free(layer->bias_momentum);
    free(layer->weight_velocity);
    free(layer->bias_velocity);
    free(layer->activations);
    free(layer->gradients);
}

// ReLU activation
static double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// ReLU derivative
static double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Softmax normalization with numerical stability
static void softmax(double* output, size_t n) {
    if (!output || n == 0) return;

    // Find max for numerical stability
    double max_val = output[0];
    for (size_t i = 1; i < n; i++) {
        if (output[i] > max_val) max_val = output[i];
    }

    // Compute exp and sum
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double val = output[i] - max_val;
        if (val < -500.0) val = -500.0;  // Prevent underflow
        output[i] = exp(val);
        sum += output[i];
    }

    // Normalize
    if (sum > 0.0) {
        for (size_t i = 0; i < n; i++) {
            output[i] /= sum;
        }
    }
}

// Apply dropout during training
static void apply_dropout(double* activations, double* mask, size_t n, double rate) {
    double scale = 1.0 / (1.0 - rate);
    for (size_t i = 0; i < n; i++) {
        mask[i] = ((double)rand() / RAND_MAX) > rate ? scale : 0.0;
        activations[i] *= mask[i];
    }
}

// Gradient clipping
static double clip_gradient(double grad) {
    if (grad > GRADIENT_CLIP) return GRADIENT_CLIP;
    if (grad < -GRADIENT_CLIP) return -GRADIENT_CLIP;
    return grad;
}

// Forward pass through a layer
static void layer_forward(Layer* layer, const double* input, bool apply_relu_activation) {
    for (size_t o = 0; o < layer->output_size; o++) {
        double sum = layer->biases[o];
        for (size_t i = 0; i < layer->input_size; i++) {
            sum += layer->weights[o * layer->input_size + i] * input[i];
        }
        layer->activations[o] = apply_relu_activation ? relu(sum) : sum;
    }
}

// Backward pass through a layer with Adam optimizer
static void layer_backward(Layer* layer, const double* input,
                          const double* output_grad, double* input_grad,
                          double learning_rate, size_t step) {
    // Bias-corrected learning rate for Adam
    double lr = learning_rate * sqrt(1.0 - pow(BETA2, step + 1)) /
                (1.0 - pow(BETA1, step + 1));

    // Compute gradients and update weights
    for (size_t o = 0; o < layer->output_size; o++) {
        double grad_o = output_grad[o];
        layer->gradients[o] = grad_o;

        // Update biases with Adam
        layer->bias_momentum[o] = BETA1 * layer->bias_momentum[o] +
                                  (1.0 - BETA1) * grad_o;
        layer->bias_velocity[o] = BETA2 * layer->bias_velocity[o] +
                                  (1.0 - BETA2) * grad_o * grad_o;
        layer->biases[o] -= lr * layer->bias_momentum[o] /
                           (sqrt(layer->bias_velocity[o]) + EPSILON);

        for (size_t i = 0; i < layer->input_size; i++) {
            size_t idx = o * layer->input_size + i;
            double weight_grad = clip_gradient(grad_o * input[i]);

            // L2 regularization
            weight_grad += ML_REGULARIZATION * layer->weights[idx];

            // Adam update
            layer->weight_momentum[idx] = BETA1 * layer->weight_momentum[idx] +
                                         (1.0 - BETA1) * weight_grad;
            layer->weight_velocity[idx] = BETA2 * layer->weight_velocity[idx] +
                                         (1.0 - BETA2) * weight_grad * weight_grad;
            layer->weights[idx] -= lr * layer->weight_momentum[idx] /
                                  (sqrt(layer->weight_velocity[idx]) + EPSILON);

            // Accumulate input gradient
            if (input_grad) {
                input_grad[i] += grad_o * layer->weights[idx];
            }
        }
    }
}

// Get internal model from external model
static MLModelInternal* get_internal(const MLModel* model) {
    if (!model) return NULL;
    size_t model_id = (size_t)model->bias;
    if (model_id >= 256) return NULL;
    return internal_models[model_id];
}

// Normalize features
static void normalize_features(const MLModelInternal* internal,
                              const double* input, double* normalized) {
    for (size_t i = 0; i < INPUT_SIZE; i++) {
        if (internal->feature_std[i] > EPSILON) {
            normalized[i] = (input[i] - internal->feature_mean[i]) /
                           internal->feature_std[i];
        } else {
            normalized[i] = input[i] - internal->feature_mean[i];
        }
    }
}

// Update running statistics for normalization
static void update_normalization_stats(MLModelInternal* internal, const double* input) {
    internal->normalization_samples++;
    double n = (double)internal->normalization_samples;

    for (size_t i = 0; i < INPUT_SIZE; i++) {
        double delta = input[i] - internal->feature_mean[i];
        internal->feature_mean[i] += delta / n;
        double delta2 = input[i] - internal->feature_mean[i];
        internal->feature_std[i] += delta * delta2;
    }

    // Compute standard deviation
    if (internal->normalization_samples > 1) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            internal->feature_std[i] = sqrt(internal->feature_std[i] / (n - 1));
            if (internal->feature_std[i] < EPSILON) {
                internal->feature_std[i] = 1.0;
            }
        }
    }
}

// Run ML model prediction
MLPrediction run_ml_model(MLModel* model, const double* features) {
    MLPrediction prediction = {0};
    prediction.bottleneck_type = NO_BOTTLENECK;
    prediction.confidence = 0.0;
    prediction.feature_contributions = NULL;

    if (!model || !features) return prediction;

    MLModelInternal* internal = get_internal(model);
    if (!internal) return prediction;

    // Normalize input features
    double normalized[INPUT_SIZE];
    normalize_features(internal, features, normalized);

    // Forward pass through hidden layer
    layer_forward(&internal->hidden_layer, normalized, true);

    // Forward pass through output layer (no activation, softmax applied separately)
    layer_forward(&internal->output_layer, internal->hidden_layer.activations, false);

    // Apply softmax to get probabilities
    double probs[NUM_CLASSES];
    memcpy(probs, internal->output_layer.activations, NUM_CLASSES * sizeof(double));
    softmax(probs, NUM_CLASSES);

    // Find best class
    int best_class = 0;
    double best_prob = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > best_prob) {
            best_prob = probs[c];
            best_class = c;
        }
    }

    prediction.bottleneck_type = (BottleneckType)best_class;
    prediction.confidence = best_prob;

    // Compute feature contributions using gradient-based saliency
    prediction.feature_contributions = calloc(model->num_features, sizeof(double));
    if (prediction.feature_contributions) {
        for (size_t f = 0; f < model->num_features; f++) {
            double contribution = 0.0;
            // Sum absolute gradients through the network for this feature
            for (size_t h = 0; h < HIDDEN_SIZE; h++) {
                double hidden_weight = internal->hidden_layer.weights[h * INPUT_SIZE + f];
                double hidden_activation = relu_derivative(
                    internal->hidden_layer.activations[h]);

                for (int c = 0; c < NUM_CLASSES; c++) {
                    double output_weight = internal->output_layer.weights[
                        c * HIDDEN_SIZE + h];
                    contribution += fabs(hidden_weight * hidden_activation *
                                       output_weight * probs[c]);
                }
            }
            prediction.feature_contributions[f] = contribution / NUM_CLASSES;
        }

        // Normalize contributions
        double max_contrib = 0.0;
        for (size_t f = 0; f < model->num_features; f++) {
            if (prediction.feature_contributions[f] > max_contrib) {
                max_contrib = prediction.feature_contributions[f];
            }
        }
        if (max_contrib > 0.0) {
            for (size_t f = 0; f < model->num_features; f++) {
                prediction.feature_contributions[f] /= max_contrib;
            }
        }
    }

    return prediction;
}

// Train model on a single sample
void ml_model_train_sample(MLModel* model, const double* features,
                           BottleneckType label) {
    if (!model || !features || (int)label >= NUM_CLASSES) return;

    MLModelInternal* internal = get_internal(model);
    if (!internal) return;

    internal->training_mode = true;

    // Update normalization statistics
    update_normalization_stats(internal, features);

    // Normalize input
    double normalized[INPUT_SIZE];
    normalize_features(internal, features, normalized);

    // Forward pass through hidden layer
    layer_forward(&internal->hidden_layer, normalized, true);

    // Apply dropout during training
    apply_dropout(internal->hidden_layer.activations, internal->dropout_mask,
                 HIDDEN_SIZE, DROPOUT_RATE);

    // Forward pass through output layer
    layer_forward(&internal->output_layer, internal->hidden_layer.activations, false);

    // Compute softmax probabilities
    double probs[NUM_CLASSES];
    memcpy(probs, internal->output_layer.activations, NUM_CLASSES * sizeof(double));
    softmax(probs, NUM_CLASSES);

    // Compute cross-entropy loss
    double loss = -log(probs[(int)label] + EPSILON);
    internal->current_loss = 0.9 * internal->current_loss + 0.1 * loss;

    // Track accuracy
    int predicted = 0;
    double max_prob = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > max_prob) {
            max_prob = probs[c];
            predicted = c;
        }
    }
    internal->accuracy_sum += (predicted == (int)label) ? 1.0 : 0.0;
    internal->accuracy_count++;

    // Compute output layer gradients (softmax + cross-entropy)
    double output_grad[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        output_grad[c] = probs[c] - ((c == (int)label) ? 1.0 : 0.0);
    }

    // Backward pass through output layer
    double hidden_grad[HIDDEN_SIZE] = {0};
    layer_backward(&internal->output_layer, internal->hidden_layer.activations,
                  output_grad, hidden_grad, internal->learning_rate,
                  internal->update_step);

    // Apply ReLU derivative and dropout mask to hidden gradients
    for (size_t h = 0; h < HIDDEN_SIZE; h++) {
        hidden_grad[h] *= relu_derivative(internal->hidden_layer.activations[h] /
                                         (internal->dropout_mask[h] + EPSILON));
        hidden_grad[h] *= internal->dropout_mask[h];
    }

    // Backward pass through hidden layer
    layer_backward(&internal->hidden_layer, normalized, hidden_grad, NULL,
                  internal->learning_rate, internal->update_step);

    internal->update_step++;
    internal->training_samples++;
    model->training_samples = internal->training_samples;
}

// Update feature importance based on model weights
void update_feature_importance(MLModel* model, double* importance) {
    if (!model || !importance) return;

    MLModelInternal* internal = get_internal(model);
    if (!internal) return;

    // Compute importance as sum of absolute weights from input to all hidden units
    for (size_t f = 0; f < model->num_features; f++) {
        double total = 0.0;
        for (size_t h = 0; h < HIDDEN_SIZE; h++) {
            total += fabs(internal->hidden_layer.weights[h * INPUT_SIZE + f]);
        }
        importance[f] = total / HIDDEN_SIZE;

        // Also store in model weights for external access
        if (model->weights) {
            model->weights[f] = importance[f];
        }
    }

    // Normalize
    double max_val = 0.0;
    for (size_t f = 0; f < model->num_features; f++) {
        if (importance[f] > max_val) max_val = importance[f];
    }
    if (max_val > 0.0) {
        for (size_t f = 0; f < model->num_features; f++) {
            importance[f] /= max_val;
        }
    }
}

// Get model accuracy estimate
double ml_model_get_accuracy(const MLModel* model) {
    if (!model) return 0.0;

    MLModelInternal* internal = get_internal(model);
    if (!internal || internal->accuracy_count == 0) return 0.5;

    return internal->accuracy_sum / (double)internal->accuracy_count;
}

// Reset model to initial state
void ml_model_reset(MLModel* model) {
    if (!model) return;

    MLModelInternal* internal = get_internal(model);
    if (!internal) return;

    // Re-initialize weights
    xavier_init(internal->hidden_layer.weights, INPUT_SIZE, HIDDEN_SIZE);
    xavier_init(internal->output_layer.weights, HIDDEN_SIZE, NUM_CLASSES);

    // Reset biases
    memset(internal->hidden_layer.biases, 0, HIDDEN_SIZE * sizeof(double));
    memset(internal->output_layer.biases, 0, NUM_CLASSES * sizeof(double));

    // Reset momentum and velocity
    memset(internal->hidden_layer.weight_momentum, 0,
           INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    memset(internal->hidden_layer.bias_momentum, 0, HIDDEN_SIZE * sizeof(double));
    memset(internal->output_layer.weight_momentum, 0,
           HIDDEN_SIZE * NUM_CLASSES * sizeof(double));
    memset(internal->output_layer.bias_momentum, 0, NUM_CLASSES * sizeof(double));

    // Reset statistics
    memset(internal->feature_mean, 0, INPUT_SIZE * sizeof(double));
    for (size_t i = 0; i < INPUT_SIZE; i++) {
        internal->feature_std[i] = 1.0;
    }

    internal->training_samples = 0;
    internal->update_step = 0;
    internal->normalization_samples = 0;
    internal->current_loss = 0.0;
    internal->accuracy_sum = 0.0;
    internal->accuracy_count = 0;

    model->training_samples = 0;
}

// Get number of training samples
size_t ml_model_get_sample_count(const MLModel* model) {
    if (!model) return 0;
    MLModelInternal* internal = get_internal(model);
    return internal ? internal->training_samples : 0;
}

// Clean up ML model
void cleanup_ml_model(MLModel* model) {
    if (!model) return;

    MLModelInternal* internal = get_internal(model);
    if (internal) {
        size_t model_id = (size_t)model->bias;
        if (model_id < 256) {
            internal_models[model_id] = NULL;
        }

        layer_cleanup(&internal->hidden_layer);
        layer_cleanup(&internal->output_layer);
        free(internal->feature_mean);
        free(internal->feature_std);
        free(internal->dropout_mask);
        free(internal);
    }

    free(model->weights);
    free(model);
}
