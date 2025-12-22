/**
 * @file prediction_model.c
 * @brief Neural network prediction model implementation
 *
 * Implements a multi-layer neural network for performance prediction
 * with online learning, batch training, and feature normalization.
 */

#include "quantum_geometric/distributed/prediction_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Internal constants
#define NUM_LAYERS 3
#define LEARNING_RATE_DEFAULT 0.001
#define BATCH_SIZE_DEFAULT 16
#define MIN_SAMPLES_DEFAULT 50
#define EPSILON 1e-8

// Training sample - internal structure
typedef struct {
    double* features;
    double* targets;
    time_t timestamp;
    double weight;
} TrainingSample;

// Model layer - internal structure
typedef struct {
    double* weights;
    double* bias;
    double* output;
    double* gradients;
    double* bias_gradients;
    size_t input_dim;
    size_t output_dim;
} ModelLayer;

// Prediction model - internal structure
struct PredictionModelImpl {
    // Neural network layers
    ModelLayer* layers;
    size_t num_layers;

    // Training state
    TrainingSample** training_buffer;
    size_t buffer_size;
    size_t buffer_capacity;
    size_t samples_seen;

    // Feature normalization
    double* feature_means;
    double* feature_stds;
    bool is_normalized;
    size_t norm_samples;

    // Performance tracking
    double train_loss;
    double validation_loss;
    double prediction_error;

    // Configuration
    PredictionConfig config;
};

// Forward declarations
static void initialize_layer(ModelLayer* layer, size_t input_dim, size_t output_dim);
static void cleanup_layer(ModelLayer* layer);
static void xavier_init(double* weights, size_t input_dim, size_t output_dim);
static void forward_pass(PredictionModel* model, const double* input, double* output);
static void apply_relu(double* x, size_t n);
static void apply_relu_derivative(const double* x, double* dx, size_t n);
static double* normalize_features(PredictionModel* model, const double* features);
static void update_feature_statistics(PredictionModel* model, const double* features);
static void add_training_sample(PredictionModel* model, const double* features, const double* targets);
static void train_on_batch(PredictionModel* model);
static void backward_pass(PredictionModel* model, const double* targets);
static void update_weights(PredictionModel* model);
static double compute_loss(const double* predictions, const double* targets, size_t n);
static void cleanup_training_sample(TrainingSample* sample);

// Initialize prediction model
PredictionModel* init_prediction_model(const PredictionConfig* config) {
    PredictionModel* model = calloc(1, sizeof(PredictionModel));
    if (!model) return NULL;

    // Store configuration
    if (config) {
        model->config = *config;
    } else {
        // Default configuration
        model->config.input_dim = PREDICTION_INPUT_DIM;
        model->config.hidden_dim = PREDICTION_HIDDEN_DIM;
        model->config.output_dim = PREDICTION_OUTPUT_DIM;
        model->config.learning_rate = LEARNING_RATE_DEFAULT;
        model->config.batch_size = BATCH_SIZE_DEFAULT;
        model->config.min_samples = MIN_SAMPLES_DEFAULT;
        model->config.enable_online_learning = true;
        model->config.enable_normalization = true;
    }

    // Initialize layers
    model->num_layers = NUM_LAYERS;
    model->layers = calloc(NUM_LAYERS, sizeof(ModelLayer));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    // Input -> Hidden
    initialize_layer(&model->layers[0], model->config.input_dim, model->config.hidden_dim);
    // Hidden -> Hidden
    initialize_layer(&model->layers[1], model->config.hidden_dim, model->config.hidden_dim);
    // Hidden -> Output
    initialize_layer(&model->layers[2], model->config.hidden_dim, model->config.output_dim);

    // Initialize training buffer
    model->buffer_capacity = PREDICTION_MAX_SAMPLES;
    model->training_buffer = calloc(model->buffer_capacity, sizeof(TrainingSample*));
    if (!model->training_buffer) {
        for (size_t i = 0; i < NUM_LAYERS; i++) {
            cleanup_layer(&model->layers[i]);
        }
        free(model->layers);
        free(model);
        return NULL;
    }
    model->buffer_size = 0;
    model->samples_seen = 0;

    // Initialize feature normalization
    model->feature_means = calloc(model->config.input_dim, sizeof(double));
    model->feature_stds = calloc(model->config.input_dim, sizeof(double));
    if (!model->feature_means || !model->feature_stds) {
        free(model->training_buffer);
        for (size_t i = 0; i < NUM_LAYERS; i++) {
            cleanup_layer(&model->layers[i]);
        }
        free(model->layers);
        free(model->feature_means);
        free(model->feature_stds);
        free(model);
        return NULL;
    }

    // Initialize stds to 1 to avoid division by zero
    for (size_t i = 0; i < model->config.input_dim; i++) {
        model->feature_stds[i] = 1.0;
    }
    model->is_normalized = false;
    model->norm_samples = 0;

    // Initialize performance tracking
    model->train_loss = 0.0;
    model->validation_loss = 0.0;
    model->prediction_error = 0.0;

    return model;
}

// Initialize a layer
static void initialize_layer(ModelLayer* layer, size_t input_dim, size_t output_dim) {
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;

    layer->weights = calloc(input_dim * output_dim, sizeof(double));
    layer->bias = calloc(output_dim, sizeof(double));
    layer->output = calloc(output_dim, sizeof(double));
    layer->gradients = calloc(input_dim * output_dim, sizeof(double));
    layer->bias_gradients = calloc(output_dim, sizeof(double));

    // Xavier initialization
    if (layer->weights) {
        xavier_init(layer->weights, input_dim, output_dim);
    }
}

// Cleanup a layer
static void cleanup_layer(ModelLayer* layer) {
    if (!layer) return;
    free(layer->weights);
    free(layer->bias);
    free(layer->output);
    free(layer->gradients);
    free(layer->bias_gradients);
}

// Xavier weight initialization
static void xavier_init(double* weights, size_t input_dim, size_t output_dim) {
    double scale = sqrt(2.0 / (double)(input_dim + output_dim));
    size_t n = input_dim * output_dim;

    for (size_t i = 0; i < n; i++) {
        // Simple uniform random in [-scale, scale]
        double u = (double)rand() / RAND_MAX;
        weights[i] = (2.0 * u - 1.0) * scale;
    }
}

// ReLU activation
static void apply_relu(double* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (x[i] < 0.0) x[i] = 0.0;
    }
}

// ReLU derivative
static void apply_relu_derivative(const double* x, double* dx, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dx[i] = (x[i] > 0.0) ? 1.0 : 0.0;
    }
}

// Normalize features using running statistics
static double* normalize_features(PredictionModel* model, const double* features) {
    if (!model || !features) return NULL;

    double* normalized = calloc(model->config.input_dim, sizeof(double));
    if (!normalized) return NULL;

    if (model->config.enable_normalization && model->is_normalized) {
        for (size_t i = 0; i < model->config.input_dim; i++) {
            double std = model->feature_stds[i];
            if (std < EPSILON) std = 1.0;
            normalized[i] = (features[i] - model->feature_means[i]) / std;
        }
    } else {
        memcpy(normalized, features, model->config.input_dim * sizeof(double));
    }

    return normalized;
}

// Update running feature statistics (online mean and std)
static void update_feature_statistics(PredictionModel* model, const double* features) {
    if (!model || !features) return;

    model->norm_samples++;
    double n = (double)model->norm_samples;

    for (size_t i = 0; i < model->config.input_dim; i++) {
        double delta = features[i] - model->feature_means[i];
        model->feature_means[i] += delta / n;

        // Welford's algorithm for variance
        double delta2 = features[i] - model->feature_means[i];
        double variance = model->feature_stds[i] * model->feature_stds[i];
        variance = ((n - 1.0) * variance + delta * delta2) / n;
        model->feature_stds[i] = sqrt(variance + EPSILON);
    }

    if (model->norm_samples >= model->config.min_samples) {
        model->is_normalized = true;
    }
}

// Forward pass through the network
static void forward_pass(PredictionModel* model, const double* input, double* output) {
    if (!model || !input) return;

    const double* current_input = input;

    for (size_t l = 0; l < model->num_layers; l++) {
        ModelLayer* layer = &model->layers[l];

        // Initialize output with bias
        memcpy(layer->output, layer->bias, layer->output_dim * sizeof(double));

        // Matrix multiplication: output = weights^T * input + bias
        // Using row-major: output[j] = sum_i(weights[i*output_dim + j] * input[i]) + bias[j]
        for (size_t j = 0; j < layer->output_dim; j++) {
            for (size_t i = 0; i < layer->input_dim; i++) {
                layer->output[j] += layer->weights[i * layer->output_dim + j] * current_input[i];
            }
        }

        // Apply activation (ReLU for hidden layers, linear for output)
        if (l < model->num_layers - 1) {
            apply_relu(layer->output, layer->output_dim);
        }

        current_input = layer->output;
    }

    // Copy final output
    if (output) {
        memcpy(output, model->layers[model->num_layers - 1].output,
               model->config.output_dim * sizeof(double));
    }
}

// Add training sample to buffer
static void add_training_sample(PredictionModel* model, const double* features, const double* targets) {
    if (!model || !features || !targets) return;

    // Create sample
    TrainingSample* sample = calloc(1, sizeof(TrainingSample));
    if (!sample) return;

    sample->features = calloc(model->config.input_dim, sizeof(double));
    sample->targets = calloc(model->config.output_dim, sizeof(double));

    if (!sample->features || !sample->targets) {
        free(sample->features);
        free(sample->targets);
        free(sample);
        return;
    }

    memcpy(sample->features, features, model->config.input_dim * sizeof(double));
    memcpy(sample->targets, targets, model->config.output_dim * sizeof(double));
    sample->timestamp = time(NULL);
    sample->weight = 1.0;

    // Add to buffer (circular)
    if (model->buffer_size >= model->buffer_capacity) {
        // Remove oldest sample
        cleanup_training_sample(model->training_buffer[0]);
        memmove(model->training_buffer, model->training_buffer + 1,
                (model->buffer_capacity - 1) * sizeof(TrainingSample*));
        model->buffer_size--;
    }

    model->training_buffer[model->buffer_size++] = sample;
}

// Cleanup training sample
static void cleanup_training_sample(TrainingSample* sample) {
    if (!sample) return;
    free(sample->features);
    free(sample->targets);
    free(sample);
}

// Compute MSE loss
static double compute_loss(const double* predictions, const double* targets, size_t n) {
    double loss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    return loss / (double)n;
}

// Backward pass (backpropagation)
static void backward_pass(PredictionModel* model, const double* targets) {
    if (!model || !targets) return;

    ModelLayer* output_layer = &model->layers[model->num_layers - 1];

    // Output layer gradients (MSE derivative)
    double* delta = calloc(output_layer->output_dim, sizeof(double));
    if (!delta) return;

    for (size_t i = 0; i < output_layer->output_dim; i++) {
        delta[i] = 2.0 * (output_layer->output[i] - targets[i]) / (double)output_layer->output_dim;
    }

    // Backpropagate through layers
    for (int l = (int)model->num_layers - 1; l >= 0; l--) {
        ModelLayer* layer = &model->layers[l];
        const double* input = (l > 0) ? model->layers[l - 1].output : NULL;

        // Compute gradients for weights and biases
        for (size_t j = 0; j < layer->output_dim; j++) {
            layer->bias_gradients[j] = delta[j];

            for (size_t i = 0; i < layer->input_dim; i++) {
                double input_val = input ? input[i] : 0.0;
                layer->gradients[i * layer->output_dim + j] = delta[j] * input_val;
            }
        }

        // Compute delta for previous layer
        if (l > 0) {
            ModelLayer* prev_layer = &model->layers[l - 1];
            double* new_delta = calloc(layer->input_dim, sizeof(double));

            if (new_delta) {
                // delta_prev = W^T * delta
                for (size_t i = 0; i < layer->input_dim; i++) {
                    new_delta[i] = 0.0;
                    for (size_t j = 0; j < layer->output_dim; j++) {
                        new_delta[i] += layer->weights[i * layer->output_dim + j] * delta[j];
                    }

                    // Apply ReLU derivative
                    if (prev_layer->output[i] <= 0.0) {
                        new_delta[i] = 0.0;
                    }
                }

                free(delta);
                delta = new_delta;
            }
        }
    }

    free(delta);
}

// Update weights using gradients
static void update_weights(PredictionModel* model) {
    if (!model) return;

    double lr = model->config.learning_rate;

    for (size_t l = 0; l < model->num_layers; l++) {
        ModelLayer* layer = &model->layers[l];

        // Update weights
        size_t weight_count = layer->input_dim * layer->output_dim;
        for (size_t i = 0; i < weight_count; i++) {
            layer->weights[i] -= lr * layer->gradients[i];
        }

        // Update biases
        for (size_t i = 0; i < layer->output_dim; i++) {
            layer->bias[i] -= lr * layer->bias_gradients[i];
        }
    }
}

// Train on mini-batch
static void train_on_batch(PredictionModel* model) {
    if (!model || model->buffer_size < model->config.batch_size) return;

    double total_loss = 0.0;
    size_t batch_start = model->buffer_size - model->config.batch_size;

    for (size_t b = 0; b < model->config.batch_size; b++) {
        TrainingSample* sample = model->training_buffer[batch_start + b];
        if (!sample) continue;

        // Normalize features
        double* normalized = normalize_features(model, sample->features);
        if (!normalized) continue;

        // Forward pass
        double output[PREDICTION_OUTPUT_DIM];
        forward_pass(model, normalized, output);

        // Compute loss
        total_loss += compute_loss(output, sample->targets, model->config.output_dim);

        // Backward pass
        backward_pass(model, sample->targets);

        // Update weights (SGD)
        update_weights(model);

        free(normalized);
    }

    model->train_loss = total_loss / (double)model->config.batch_size;
}

// Make prediction
PredictionResult* prediction_model_predict(PredictionModel* model, const double* features, size_t num_features) {
    if (!model || !features) return NULL;
    (void)num_features;  // Use config.input_dim instead

    PredictionResult* result = calloc(1, sizeof(PredictionResult));
    if (!result) return NULL;

    result->values = calloc(model->config.output_dim, sizeof(double));
    result->confidences = calloc(model->config.output_dim, sizeof(double));
    if (!result->values || !result->confidences) {
        free(result->values);
        free(result->confidences);
        free(result);
        return NULL;
    }

    // Normalize features
    double* normalized = normalize_features(model, features);
    if (!normalized) {
        free(result->values);
        free(result->confidences);
        free(result);
        return NULL;
    }

    // Forward pass
    forward_pass(model, normalized, result->values);

    // Compute confidence based on output magnitudes and training
    result->num_values = model->config.output_dim;
    double sum_confidence = 0.0;

    for (size_t i = 0; i < result->num_values; i++) {
        // Sigmoid-like confidence based on output magnitude
        double magnitude = fabs(result->values[i]);
        result->confidences[i] = 1.0 / (1.0 + exp(-magnitude));
        sum_confidence += result->confidences[i];
    }

    result->confidence = sum_confidence / (double)result->num_values;

    // Uncertainty based on training samples and loss
    double training_factor = fmin(1.0, (double)model->samples_seen / (double)model->config.min_samples);
    double loss_factor = exp(-model->train_loss);
    result->uncertainty = 1.0 - (training_factor * loss_factor * result->confidence);

    free(normalized);
    return result;
}

// Update model with new sample
void prediction_model_update(PredictionModel* model, const double* features, const double* targets, size_t num_features) {
    if (!model || !features || !targets) return;
    (void)num_features;

    // Update feature statistics
    update_feature_statistics(model, features);

    // Add to training buffer
    add_training_sample(model, features, targets);
    model->samples_seen++;

    // Train if enough samples and online learning enabled
    if (model->config.enable_online_learning &&
        model->samples_seen >= model->config.min_samples &&
        model->buffer_size >= model->config.batch_size) {

        // Train periodically
        if (model->samples_seen % PREDICTION_UPDATE_INTERVAL == 0) {
            train_on_batch(model);
        }
    }
}

// Get training loss
double prediction_model_get_train_loss(const PredictionModel* model) {
    return model ? model->train_loss : 0.0;
}

// Get validation loss
double prediction_model_get_validation_loss(const PredictionModel* model) {
    return model ? model->validation_loss : 0.0;
}

// Get prediction error
double prediction_model_get_error(const PredictionModel* model) {
    return model ? model->prediction_error : 0.0;
}

// Get number of samples seen
size_t prediction_model_get_samples_seen(const PredictionModel* model) {
    return model ? model->samples_seen : 0;
}

// Reset model state
void prediction_model_reset(PredictionModel* model) {
    if (!model) return;

    // Clear training buffer
    for (size_t i = 0; i < model->buffer_size; i++) {
        cleanup_training_sample(model->training_buffer[i]);
        model->training_buffer[i] = NULL;
    }
    model->buffer_size = 0;
    model->samples_seen = 0;

    // Reset normalization
    memset(model->feature_means, 0, model->config.input_dim * sizeof(double));
    for (size_t i = 0; i < model->config.input_dim; i++) {
        model->feature_stds[i] = 1.0;
    }
    model->is_normalized = false;
    model->norm_samples = 0;

    // Reinitialize weights
    for (size_t l = 0; l < model->num_layers; l++) {
        ModelLayer* layer = &model->layers[l];
        xavier_init(layer->weights, layer->input_dim, layer->output_dim);
        memset(layer->bias, 0, layer->output_dim * sizeof(double));
    }

    // Reset metrics
    model->train_loss = 0.0;
    model->validation_loss = 0.0;
    model->prediction_error = 0.0;
}

// Free prediction result
void cleanup_prediction_result(PredictionResult* result) {
    if (!result) return;
    free(result->values);
    free(result->confidences);
    free(result);
}

// Clean up prediction model
void cleanup_prediction_model(PredictionModel* model) {
    if (!model) return;

    // Clean up layers
    for (size_t i = 0; i < model->num_layers; i++) {
        cleanup_layer(&model->layers[i]);
    }
    free(model->layers);

    // Clean up training buffer
    for (size_t i = 0; i < model->buffer_size; i++) {
        cleanup_training_sample(model->training_buffer[i]);
    }
    free(model->training_buffer);

    // Clean up feature normalization
    free(model->feature_means);
    free(model->feature_stds);

    free(model);
}
