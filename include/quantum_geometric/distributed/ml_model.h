#ifndef ML_MODEL_H
#define ML_MODEL_H

/**
 * @file ml_model.h
 * @brief Simple ML model for bottleneck prediction and feature analysis
 *
 * This header provides the ML model functions used by bottleneck_detector.
 * The main types (MLModel, MLPrediction) are defined in bottleneck_detector.h.
 */

#include "quantum_geometric/distributed/bottleneck_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Training parameters
#define ML_LEARNING_RATE 0.01
#define ML_REGULARIZATION 0.001
#define ML_MIN_SAMPLES 10

// Additional ML model utilities

/**
 * @brief Train ML model on a single sample
 */
void ml_model_train_sample(MLModel* model, const double* features,
                           BottleneckType label);

/**
 * @brief Get model accuracy estimate
 */
double ml_model_get_accuracy(const MLModel* model);

/**
 * @brief Reset model to initial state
 */
void ml_model_reset(MLModel* model);

/**
 * @brief Get number of training samples seen
 */
size_t ml_model_get_sample_count(const MLModel* model);

#ifdef __cplusplus
}
#endif

#endif // ML_MODEL_H
