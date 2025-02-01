#ifndef IMPORTANCE_ANALYZER_H
#define IMPORTANCE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Importance types
typedef enum {
    IMPORTANCE_CRITICAL,     // Critical importance
    IMPORTANCE_HIGH,         // High importance
    IMPORTANCE_MEDIUM,       // Medium importance
    IMPORTANCE_LOW,          // Low importance
    IMPORTANCE_NEGLIGIBLE   // Negligible importance
} importance_type_t;

// Analysis modes
typedef enum {
    ANALYZE_STATIC,         // Static analysis
    ANALYZE_DYNAMIC,        // Dynamic analysis
    ANALYZE_CONTEXTUAL,     // Contextual analysis
    ANALYZE_ADAPTIVE       // Adaptive analysis
} analysis_mode_t;

// Component types
typedef enum {
    COMPONENT_QUANTUM,      // Quantum component
    COMPONENT_CLASSICAL,    // Classical component
    COMPONENT_HYBRID,       // Hybrid component
    COMPONENT_SYSTEM       // System component
} component_type_t;

// Context types
typedef enum {
    CONTEXT_OPERATIONAL,    // Operational context
    CONTEXT_PERFORMANCE,    // Performance context
    CONTEXT_RESOURCE,       // Resource context
    CONTEXT_QUANTUM        // Quantum context
} context_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;          // Analysis mode
    bool track_history;            // Track history
    bool enable_learning;          // Enable learning
    bool monitor_changes;          // Monitor changes
    size_t window_size;           // Analysis window
    double threshold;             // Importance threshold
} analyzer_config_t;

// Importance metrics
typedef struct {
    importance_type_t type;        // Importance type
    double score;                  // Importance score
    double confidence;             // Confidence level
    double impact_factor;          // Impact factor
    size_t dependencies;          // Number of dependencies
    double criticality;           // Criticality score
} importance_metrics_t;

// Component metrics
typedef struct {
    component_type_t type;         // Component type
    importance_type_t importance;  // Component importance
    double utilization;            // Component utilization
    double reliability;            // Component reliability
    size_t interactions;          // Interaction count
    void* component_data;         // Additional data
} component_metrics_t;

// Context analysis
typedef struct {
    context_type_t type;          // Context type
    double relevance;              // Context relevance
    double weight;                 // Context weight
    struct timespec timestamp;     // Analysis timestamp
    char* description;            // Context description
    void* context_data;          // Additional data
} context_analysis_t;

// Opaque analyzer handle
typedef struct importance_analyzer_t importance_analyzer_t;

// Core functions
importance_analyzer_t* create_importance_analyzer(const analyzer_config_t* config);
void destroy_importance_analyzer(importance_analyzer_t* analyzer);

// Analysis functions
bool analyze_importance(importance_analyzer_t* analyzer,
                       component_type_t type,
                       importance_metrics_t* metrics);
bool analyze_component(importance_analyzer_t* analyzer,
                      const component_metrics_t* component,
                      importance_metrics_t* metrics);
bool analyze_context(importance_analyzer_t* analyzer,
                    context_type_t type,
                    context_analysis_t* analysis);

// Evaluation functions
bool evaluate_importance(importance_analyzer_t* analyzer,
                        const importance_metrics_t* metrics,
                        importance_type_t* type);
bool evaluate_component(importance_analyzer_t* analyzer,
                       const component_metrics_t* component,
                       importance_type_t* type);
bool evaluate_context(importance_analyzer_t* analyzer,
                     const context_analysis_t* context,
                     double* relevance);

// Monitoring functions
bool monitor_importance(importance_analyzer_t* analyzer,
                       component_type_t type,
                       importance_metrics_t* metrics);
bool track_changes(importance_analyzer_t* analyzer,
                  const importance_metrics_t* metrics,
                  bool* significant_change);
bool get_importance_history(const importance_analyzer_t* analyzer,
                          importance_metrics_t* history,
                          size_t* num_entries);

// Learning functions
bool learn_patterns(importance_analyzer_t* analyzer,
                   const importance_metrics_t* metrics);
bool update_model(importance_analyzer_t* analyzer,
                 const importance_metrics_t* metrics);
bool validate_learning(importance_analyzer_t* analyzer,
                      const importance_metrics_t* metrics);

// Dependency analysis
bool analyze_dependencies(importance_analyzer_t* analyzer,
                         const component_metrics_t* component,
                         component_metrics_t* dependencies,
                         size_t* num_dependencies);
bool evaluate_dependency_chain(importance_analyzer_t* analyzer,
                             const component_metrics_t* component,
                             importance_metrics_t* chain_metrics);
bool validate_dependencies(importance_analyzer_t* analyzer,
                         const component_metrics_t* component);

// Quantum-specific functions
bool analyze_quantum_importance(importance_analyzer_t* analyzer,
                              importance_metrics_t* metrics);
bool evaluate_quantum_context(importance_analyzer_t* analyzer,
                            context_analysis_t* analysis);
bool validate_quantum_importance(importance_analyzer_t* analyzer,
                               const importance_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const importance_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(importance_analyzer_t* analyzer,
                         const char* filename);
void free_context_analysis(context_analysis_t* analysis);

#endif // IMPORTANCE_ANALYZER_H
