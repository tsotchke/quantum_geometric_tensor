#ifndef ACTION_PLAN_GENERATOR_H
#define ACTION_PLAN_GENERATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * @brief Configuration for action plan generator
 */
typedef struct ActionConfig {
    double min_confidence;          // Minimum confidence threshold
    size_t max_active_plans;        // Maximum concurrent plans
    bool enable_ml_prediction;      // Enable ML-based impact prediction
    bool enable_validation;         // Enable impact validation
    size_t history_size;            // Size of action history
    double learning_rate;           // ML model learning rate
} ActionConfig;

/**
 * @brief Insight from performance analysis
 */
typedef struct Insight {
    char* description;              // Insight description
    double severity;                // Severity (0.0-1.0)
    double confidence;              // Confidence in insight
    char* category;                 // Category (memory, compute, etc.)
    void* context_data;             // Additional context
    size_t context_size;            // Size of context data
} Insight;

// ============================================================================
// Internal Types (opaque pointers for external use)
// ============================================================================

/**
 * @brief Action history tracker
 */
typedef struct ActionHistoryEntry {
    time_t timestamp;
    double impact;
    double predicted_impact;
    bool was_successful;
} ActionHistoryEntry;

typedef struct ActionHistory {
    ActionHistoryEntry* entries;
    size_t num_entries;
    size_t capacity;
} ActionHistory;

/**
 * @brief Simple ML model for impact prediction
 */
typedef struct MLModel {
    double* weights;
    size_t num_features;
    double bias;
    double learning_rate;
    size_t training_samples;
} MLModel;

/**
 * @brief Progress tracking for actions
 */
typedef struct ProgressTracker {
    double* step_progress;
    size_t num_steps;
    time_t last_update;
    bool is_stalled;
} ProgressTracker;

// ============================================================================
// Action Types and Structures
// ============================================================================

/**
 * @brief Action type enumeration
 */
typedef enum {
    MEMORY_ACTION,
    COMPUTE_ACTION,
    NETWORK_ACTION,
    QUANTUM_ACTION,
    CONFIGURATION_ACTION,
    HARDWARE_ACTION
} ActionType;

/**
 * @brief Implementation step
 */
typedef struct {
    char* description;
    double estimated_effort;
    double completion_percentage;
    bool is_blocking;
    bool requires_validation;
} ImplementationStep;

/**
 * @brief Action dependency
 */
typedef struct {
    ActionType dependent_type;
    char* dependency_reason;
    bool is_hard_dependency;
    double impact_factor;
} ActionDependency;

/**
 * @brief Action plan structure
 */
typedef struct ActionPlan {
    // Basic info
    ActionType type;
    char* description;
    double estimated_impact;
    double confidence;

    // Implementation details
    ImplementationStep** steps;
    size_t num_steps;

    // Dependencies
    ActionDependency* dependencies;
    size_t num_dependencies;

    // Progress tracking
    double progress;
    bool is_completed;
    time_t start_time;
    time_t completion_time;

    // Effectiveness
    double measured_impact;
    bool impact_validated;
} ActionPlan;

/**
 * @brief Action plan generator
 */
typedef struct ActionPlanGenerator {
    // Current plans
    ActionPlan** active_plans;
    size_t num_active;

    // History tracking
    ActionHistory* history;

    // Impact prediction
    MLModel* impact_predictor;

    // Progress monitoring
    ProgressTracker* progress_tracker;

    // Configuration
    ActionConfig config;
} ActionPlanGenerator;

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Initialize action plan generator
 */
ActionPlanGenerator* init_action_plan_generator(const ActionConfig* config);

/**
 * @brief Create action plan from insight
 */
ActionPlan* create_action_plan(
    ActionPlanGenerator* generator,
    const Insight* insight);

/**
 * @brief Update action plan progress
 */
void update_plan_progress(
    ActionPlanGenerator* generator,
    ActionPlan* plan,
    size_t step_index,
    double progress);

/**
 * @brief Validate action impact
 */
void validate_action_impact(
    ActionPlanGenerator* generator,
    ActionPlan* plan);

/**
 * @brief Clean up action plan generator
 */
void cleanup_action_plan_generator(ActionPlanGenerator* generator);

// ============================================================================
// Internal Helper Functions
// ============================================================================

// History management
ActionHistory* create_action_history(void);
void update_action_history(ActionHistory* history, const ActionPlan* plan);
void cleanup_action_history(ActionHistory* history);

// ML model
MLModel* init_impact_prediction_model(void);
double predict_action_impact(MLModel* model, const ActionPlan* plan);
double compute_plan_confidence(MLModel* model, const ActionPlan* plan);
void update_impact_predictor(MLModel* model, const ActionPlan* plan, double accuracy);
void cleanup_ml_model(MLModel* model);

// Progress tracking
ProgressTracker* init_progress_tracker(void);
void update_progress_tracking(ProgressTracker* tracker, const ActionPlan* plan);
void cleanup_progress_tracker(ProgressTracker* tracker);

// Action type determination
int determine_action_type(const Insight* insight);

// Step generation (internal)
void generate_implementation_steps(ActionPlanGenerator* generator,
                                  ActionPlan* plan,
                                  const Insight* insight);
void identify_dependencies(ActionPlanGenerator* generator, ActionPlan* plan);

// Step type generators
void generate_memory_steps(ActionPlan* plan, const Insight* insight);
void generate_compute_steps(ActionPlan* plan, const Insight* insight);
void generate_network_steps(ActionPlan* plan, const Insight* insight);
void generate_quantum_steps(ActionPlan* plan, const Insight* insight);
void generate_config_steps(ActionPlan* plan, const Insight* insight);
void generate_hardware_steps(ActionPlan* plan, const Insight* insight);

// Effort estimation
void estimate_step_effort(void* step);

// Progress helpers
void update_overall_progress(ActionPlan* plan);
bool should_validate_impact(const ActionPlan* plan);
double measure_actual_impact(const ActionPlan* plan);
double compute_prediction_accuracy(double estimated, double measured);

// Cleanup helpers
void cleanup_action_plan(ActionPlan* plan);

#ifdef __cplusplus
}
#endif

#endif // ACTION_PLAN_GENERATOR_H
