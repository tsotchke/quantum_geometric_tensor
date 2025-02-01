#ifndef COMPLEXITY_ANALYZER_H
#define COMPLEXITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Complexity types
typedef enum {
    COMPLEXITY_TIME,          // Time complexity
    COMPLEXITY_SPACE,         // Space complexity
    COMPLEXITY_CIRCUIT,       // Circuit complexity
    COMPLEXITY_QUANTUM,       // Quantum complexity
    COMPLEXITY_GEOMETRIC     // Geometric complexity
} complexity_type_t;

// Analysis modes
typedef enum {
    ANALYSIS_STATIC,          // Static analysis
    ANALYSIS_DYNAMIC,         // Dynamic analysis
    ANALYSIS_HYBRID,          // Hybrid analysis
    ANALYSIS_QUANTUM         // Quantum-specific analysis
} analysis_mode_t;

// Complexity classes
typedef enum {
    CLASS_CONSTANT,           // O(1)
    CLASS_LOGARITHMIC,        // O(log n)
    CLASS_LINEAR,             // O(n)
    CLASS_POLYNOMIAL,         // O(n^k)
    CLASS_EXPONENTIAL,        // O(2^n)
    CLASS_QUANTUM            // Quantum speedup
} complexity_class_t;

// Resource types
typedef enum {
    RESOURCE_GATES,           // Quantum gates
    RESOURCE_QUBITS,          // Number of qubits
    RESOURCE_DEPTH,           // Circuit depth
    RESOURCE_MEMORY,          // Memory usage
    RESOURCE_TIME            // Execution time
} resource_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;            // Analysis mode
    complexity_type_t type;          // Complexity type
    bool enable_optimization;        // Enable optimization
    bool track_resources;           // Track resource usage
    bool validate_bounds;           // Validate complexity bounds
    size_t sample_size;             // Analysis sample size
} analyzer_config_t;

// Complexity metrics
typedef struct {
    complexity_type_t type;          // Complexity type
    complexity_class_t class;        // Complexity class
    double coefficient;              // Leading coefficient
    size_t degree;                   // Polynomial degree
    double overhead;                 // Constant overhead
    bool is_tight_bound;            // Tight bound flag
} complexity_metrics_t;

// Resource usage
typedef struct {
    resource_type_t type;            // Resource type
    size_t quantity;                 // Resource quantity
    double utilization;              // Resource utilization
    double efficiency;               // Resource efficiency
    size_t peak_usage;              // Peak resource usage
    double scaling_factor;           // Scaling factor
} resource_usage_t;

// Performance bounds
typedef struct {
    double lower_bound;              // Lower bound
    double upper_bound;              // Upper bound
    bool is_asymptotic;             // Asymptotic bounds
    bool is_amortized;              // Amortized bounds
    double average_case;             // Average case
    double worst_case;              // Worst case
} performance_bounds_t;

// Optimization strategy
typedef struct {
    complexity_type_t target;        // Target complexity
    double improvement_factor;       // Improvement factor
    bool quantum_speedup;           // Quantum speedup
    bool geometric_optimization;    // Geometric optimization
    char* strategy_description;     // Strategy description
    void* strategy_data;           // Strategy-specific data
} optimization_strategy_t;

// Opaque analyzer handle
typedef struct complexity_analyzer_t complexity_analyzer_t;

// Core functions
complexity_analyzer_t* create_complexity_analyzer(const analyzer_config_t* config);
void destroy_complexity_analyzer(complexity_analyzer_t* analyzer);

// Analysis functions
bool analyze_complexity(complexity_analyzer_t* analyzer,
                       complexity_metrics_t* metrics);
bool analyze_resource_usage(complexity_analyzer_t* analyzer,
                          resource_usage_t* usage);
bool analyze_performance_bounds(complexity_analyzer_t* analyzer,
                              performance_bounds_t* bounds);

// Resource tracking
bool track_resource_usage(complexity_analyzer_t* analyzer,
                         resource_type_t resource,
                         resource_usage_t* usage);
bool update_resource_metrics(complexity_analyzer_t* analyzer,
                           const resource_usage_t* usage);
bool get_resource_metrics(const complexity_analyzer_t* analyzer,
                         resource_type_t resource,
                         resource_usage_t* metrics);

// Optimization functions
bool optimize_complexity(complexity_analyzer_t* analyzer,
                        complexity_type_t type,
                        optimization_strategy_t* strategy);
bool validate_optimization(complexity_analyzer_t* analyzer,
                         const optimization_strategy_t* strategy,
                         complexity_metrics_t* result);
bool apply_optimization(complexity_analyzer_t* analyzer,
                       const optimization_strategy_t* strategy);

// Quantum-specific functions
bool analyze_quantum_circuit(complexity_analyzer_t* analyzer,
                           complexity_metrics_t* metrics);
bool optimize_quantum_resources(complexity_analyzer_t* analyzer,
                              resource_type_t resource,
                              optimization_strategy_t* strategy);
bool validate_quantum_bounds(complexity_analyzer_t* analyzer,
                           const performance_bounds_t* bounds);

// Geometric analysis
bool analyze_geometric_complexity(complexity_analyzer_t* analyzer,
                                complexity_metrics_t* metrics);
bool optimize_geometric_operations(complexity_analyzer_t* analyzer,
                                 optimization_strategy_t* strategy);
bool validate_geometric_bounds(complexity_analyzer_t* analyzer,
                             const performance_bounds_t* bounds);

// Utility functions
bool export_complexity_data(const complexity_analyzer_t* analyzer,
                          const char* filename);
bool import_complexity_data(complexity_analyzer_t* analyzer,
                          const char* filename);
void free_complexity_metrics(complexity_metrics_t* metrics);

#endif // COMPLEXITY_ANALYZER_H
