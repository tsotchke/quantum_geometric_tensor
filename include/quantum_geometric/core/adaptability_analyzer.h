#ifndef ADAPTABILITY_ANALYZER_H
#define ADAPTABILITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>

// Adaptation modes
typedef enum {
    ADAPT_CONTINUOUS,         // Continuous adaptation
    ADAPT_PERIODIC,          // Periodic adaptation
    ADAPT_THRESHOLD,         // Threshold-based adaptation
    ADAPT_EVENT_DRIVEN       // Event-driven adaptation
} adaptation_mode_t;

// Optimization targets
typedef enum {
    TARGET_THROUGHPUT,       // Optimize for throughput
    TARGET_LATENCY,         // Optimize for latency
    TARGET_EFFICIENCY,      // Optimize for efficiency
    TARGET_RESOURCE_USAGE   // Optimize for resource usage
} optimization_target_t;

// Learning strategies
typedef enum {
    LEARN_REINFORCEMENT,     // Reinforcement learning
    LEARN_BAYESIAN,         // Bayesian optimization
    LEARN_EVOLUTIONARY,     // Evolutionary algorithms
    LEARN_HYBRID           // Hybrid learning approach
} learning_strategy_t;

// Prediction models
typedef enum {
    PRED_NEURAL_NETWORK,     // Neural network model
    PRED_STATISTICAL,       // Statistical model
    PRED_QUANTUM,          // Quantum prediction model
    PRED_ENSEMBLE         // Ensemble model
} prediction_model_t;

// Analyzer configuration
typedef struct {
    adaptation_mode_t mode;           // Adaptation mode
    optimization_target_t target;     // Optimization target
    learning_strategy_t strategy;     // Learning strategy
    prediction_model_t model;         // Prediction model
    double adaptation_rate;           // Rate of adaptation
    double learning_rate;             // Learning rate
    size_t history_window;           // History window size
    bool enable_quantum;             // Enable quantum optimization
} analyzer_config_t;

// Performance metrics
typedef struct {
    double throughput;               // System throughput
    double latency;                  // System latency
    double efficiency;               // System efficiency
    double resource_usage;           // Resource usage
    double adaptation_overhead;      // Adaptation overhead
    double prediction_accuracy;      // Prediction accuracy
} analyzer_metrics_t;

// System state
typedef struct {
    double load_level;               // Current load level
    double resource_availability;    // Resource availability
    double performance_level;        // Performance level
    double stability_index;          // System stability
    double adaptation_progress;      // Adaptation progress
    bool requires_adaptation;        // Adaptation requirement flag
} system_state_t;

// Adaptation parameters
typedef struct {
    double throughput_threshold;     // Throughput threshold
    double latency_threshold;        // Latency threshold
    double efficiency_threshold;     // Efficiency threshold
    double resource_threshold;       // Resource usage threshold
    double stability_threshold;      // Stability threshold
    double confidence_level;         // Confidence level
} adaptation_params_t;

// Prediction results
typedef struct {
    double predicted_throughput;     // Predicted throughput
    double predicted_latency;        // Predicted latency
    double predicted_efficiency;     // Predicted efficiency
    double prediction_confidence;    // Prediction confidence
    double prediction_horizon;       // Prediction horizon
    bool requires_action;           // Action requirement flag
} prediction_results_t;

// Opaque analyzer handle
typedef struct adaptability_analyzer_t adaptability_analyzer_t;

// Core functions
adaptability_analyzer_t* create_adaptability_analyzer(const analyzer_config_t* config);
void destroy_adaptability_analyzer(adaptability_analyzer_t* analyzer);

// Analysis functions
bool analyze_system_state(adaptability_analyzer_t* analyzer,
                         system_state_t* state);
bool analyze_adaptation_needs(adaptability_analyzer_t* analyzer,
                            const system_state_t* state,
                            adaptation_params_t* params);
bool analyze_performance_trends(adaptability_analyzer_t* analyzer,
                              analyzer_metrics_t* metrics);

// Prediction functions
bool predict_performance(adaptability_analyzer_t* analyzer,
                        const system_state_t* state,
                        prediction_results_t* results);
bool validate_predictions(adaptability_analyzer_t* analyzer,
                         const prediction_results_t* predictions,
                         const analyzer_metrics_t* actual);
bool update_prediction_model(adaptability_analyzer_t* analyzer,
                           const system_state_t* state,
                           const analyzer_metrics_t* metrics);

// Adaptation functions
bool adapt_system_parameters(adaptability_analyzer_t* analyzer,
                           const system_state_t* state,
                           const adaptation_params_t* params);
bool validate_adaptation(adaptability_analyzer_t* analyzer,
                        const system_state_t* before,
                        const system_state_t* after);
bool rollback_adaptation(adaptability_analyzer_t* analyzer,
                        const system_state_t* target_state);

// Learning functions
bool update_learning_model(adaptability_analyzer_t* analyzer,
                          const system_state_t* state,
                          const analyzer_metrics_t* metrics);
bool optimize_learning_parameters(adaptability_analyzer_t* analyzer,
                                const analyzer_metrics_t* metrics);
bool validate_learning_progress(adaptability_analyzer_t* analyzer,
                              const analyzer_metrics_t* metrics);

// Monitoring functions
bool get_analyzer_metrics(const adaptability_analyzer_t* analyzer,
                         analyzer_metrics_t* metrics);
bool reset_analyzer_metrics(adaptability_analyzer_t* analyzer);
bool log_adaptation_event(adaptability_analyzer_t* analyzer,
                         const system_state_t* state,
                         const analyzer_metrics_t* metrics);

#endif // ADAPTABILITY_ANALYZER_H
