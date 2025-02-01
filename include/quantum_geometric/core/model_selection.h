#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Model types
typedef enum {
    MODEL_QUANTUM,         // Quantum model
    MODEL_CLASSICAL,       // Classical model
    MODEL_HYBRID,          // Hybrid model
    MODEL_GEOMETRIC,       // Geometric model
    MODEL_ENSEMBLE        // Ensemble model
} model_type_t;

// Selection criteria
typedef enum {
    CRITERIA_ACCURACY,     // Accuracy-based
    CRITERIA_PERFORMANCE,  // Performance-based
    CRITERIA_COMPLEXITY,   // Complexity-based
    CRITERIA_RESOURCE,     // Resource-based
    CRITERIA_HYBRID       // Hybrid criteria
} selection_criteria_t;

// Validation methods
typedef enum {
    VALIDATE_CROSS,       // Cross-validation
    VALIDATE_HOLDOUT,     // Holdout validation
    VALIDATE_BOOTSTRAP,   // Bootstrap validation
    VALIDATE_QUANTUM     // Quantum validation
} validation_method_t;

// Search strategies
typedef enum {
    SEARCH_GRID,          // Grid search
    SEARCH_RANDOM,        // Random search
    SEARCH_BAYESIAN,      // Bayesian optimization
    SEARCH_QUANTUM       // Quantum search
} search_strategy_t;

// Selector configuration
typedef struct {
    selection_criteria_t criteria;  // Selection criteria
    validation_method_t method;     // Validation method
    search_strategy_t strategy;     // Search strategy
    size_t num_folds;              // Number of folds
    double split_ratio;            // Train/test split ratio
    bool enable_early_stopping;    // Early stopping flag
} selector_config_t;

// Model metrics
typedef struct {
    model_type_t type;             // Model type
    double accuracy;               // Model accuracy
    double loss;                   // Model loss
    double complexity;             // Model complexity
    double resource_usage;         // Resource usage
    void* model_data;            // Additional data
} model_metrics_t;

// Validation results
typedef struct {
    validation_method_t method;    // Validation method
    double* scores;                // Validation scores
    size_t num_scores;             // Number of scores
    double mean_score;             // Mean score
    double std_deviation;          // Standard deviation
    void* validation_data;        // Additional data
} validation_results_t;

// Search results
typedef struct {
    search_strategy_t strategy;    // Search strategy
    model_type_t best_model;       // Best model type
    double best_score;             // Best score
    size_t iterations;             // Search iterations
    struct timespec search_time;   // Search time
    void* search_data;           // Additional data
} search_results_t;

// Opaque selector handle
typedef struct model_selector_t model_selector_t;

// Core functions
model_selector_t* create_model_selector(const selector_config_t* config);
void destroy_model_selector(model_selector_t* selector);

// Selection functions
bool select_model(model_selector_t* selector,
                 model_type_t* models,
                 size_t num_models,
                 model_type_t* best_model);
bool evaluate_model(model_selector_t* selector,
                   model_type_t type,
                   model_metrics_t* metrics);
bool compare_models(model_selector_t* selector,
                   const model_metrics_t* model1,
                   const model_metrics_t* model2,
                   int* comparison);

// Validation functions
bool validate_model(model_selector_t* selector,
                   model_type_t type,
                   validation_results_t* results);
bool cross_validate(model_selector_t* selector,
                   model_type_t type,
                   validation_results_t* results);
bool validate_ensemble(model_selector_t* selector,
                      model_type_t* models,
                      size_t num_models,
                      validation_results_t* results);

// Search functions
bool search_models(model_selector_t* selector,
                  model_type_t* models,
                  size_t num_models,
                  search_results_t* results);
bool optimize_search(model_selector_t* selector,
                    search_strategy_t strategy,
                    search_results_t* results);
bool validate_search(model_selector_t* selector,
                    const search_results_t* results);

// Ensemble functions
bool create_ensemble(model_selector_t* selector,
                    model_type_t* models,
                    size_t num_models,
                    model_type_t* ensemble);
bool evaluate_ensemble(model_selector_t* selector,
                      const model_type_t* ensemble,
                      model_metrics_t* metrics);
bool optimize_ensemble(model_selector_t* selector,
                      model_type_t* ensemble,
                      model_metrics_t* metrics);

// Quantum-specific functions
bool select_quantum_model(model_selector_t* selector,
                        model_metrics_t* metrics);
bool validate_quantum_model(model_selector_t* selector,
                          validation_results_t* results);
bool optimize_quantum_ensemble(model_selector_t* selector,
                             model_type_t* ensemble,
                             model_metrics_t* metrics);

// Utility functions
bool export_selector_data(const model_selector_t* selector,
                         const char* filename);
bool import_selector_data(model_selector_t* selector,
                         const char* filename);
void free_validation_results(validation_results_t* results);

#endif // MODEL_SELECTION_H
