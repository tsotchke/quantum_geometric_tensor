#ifndef PREDICTION_MODEL_H
#define PREDICTION_MODEL_H

/**
 * @file prediction_model.h
 * @brief Neural network prediction model for performance forecasting
 *
 * Provides a multi-layer neural network for predicting performance
 * metrics, resource usage, and completion times in distributed training.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define PREDICTION_INPUT_DIM 32
#define PREDICTION_HIDDEN_DIM 64
#define PREDICTION_OUTPUT_DIM 8
#define PREDICTION_MAX_SAMPLES 1000
#define PREDICTION_UPDATE_INTERVAL 100

// Feature types for prediction
typedef enum {
    FEATURE_PROGRESS_RATE,
    FEATURE_RESOURCE_USAGE,
    FEATURE_DEPENDENCY_STATUS,
    FEATURE_BLOCKING_FACTOR,
    FEATURE_EFFICIENCY_METRIC,
    FEATURE_QUANTUM_UTILIZATION,
    FEATURE_NETWORK_SATURATION,
    FEATURE_MEMORY_PRESSURE
} PredictionFeatureType;

// Prediction configuration
typedef struct {
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;
    double learning_rate;
    size_t batch_size;
    size_t min_samples;
    bool enable_online_learning;
    bool enable_normalization;
} PredictionConfig;

// Prediction result
typedef struct {
    double* values;             // Predicted values
    size_t num_values;          // Number of predictions
    double confidence;          // Overall confidence
    double* confidences;        // Per-value confidences
    double uncertainty;         // Prediction uncertainty
} PredictionResult;

// Prediction model (opaque)
typedef struct PredictionModelImpl PredictionModel;

// Initialize prediction model
PredictionModel* init_prediction_model(const PredictionConfig* config);

// Make prediction
PredictionResult* prediction_model_predict(
    PredictionModel* model,
    const double* features,
    size_t num_features);

// Update model with new sample (online learning)
void prediction_model_update(
    PredictionModel* model,
    const double* features,
    const double* targets,
    size_t num_features);

// Get training loss
double prediction_model_get_train_loss(const PredictionModel* model);

// Get validation loss
double prediction_model_get_validation_loss(const PredictionModel* model);

// Get prediction error
double prediction_model_get_error(const PredictionModel* model);

// Get number of samples seen
size_t prediction_model_get_samples_seen(const PredictionModel* model);

// Reset model state
void prediction_model_reset(PredictionModel* model);

// Free prediction result
void cleanup_prediction_result(PredictionResult* result);

// Clean up prediction model
void cleanup_prediction_model(PredictionModel* model);

#ifdef __cplusplus
}
#endif

#endif // PREDICTION_MODEL_H
