#ifndef DISTRIBUTION_OPTIMIZER_H
#define DISTRIBUTION_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include "distribution_analyzer.h"

// Optimization strategies
typedef enum {
    OPT_STRATEGY_BALANCED,     // Balanced distribution
    OPT_STRATEGY_PERFORMANCE,  // Performance-focused
    OPT_STRATEGY_EFFICIENCY,   // Efficiency-focused
    OPT_STRATEGY_HYBRID       // Hybrid optimization
} optimization_strategy_t;

// Optimization goals
typedef enum {
    GOAL_THROUGHPUT,          // Maximize throughput
    GOAL_LATENCY,             // Minimize latency
    GOAL_RESOURCE_USAGE,      // Optimize resource usage
    GOAL_ENERGY_EFFICIENCY   // Maximize energy efficiency
} optimization_goal_t;

// Cost models
typedef enum {
    COST_MODEL_LINEAR,        // Linear cost model
    COST_MODEL_QUADRATIC,     // Quadratic cost model
    COST_MODEL_EXPONENTIAL,   // Exponential cost model
    COST_MODEL_CUSTOM        // Custom cost model
} cost_model_t;

// Optimization phases
typedef enum {
    PHASE_ANALYSIS,           // Analysis phase
    PHASE_PLANNING,           // Planning phase
    PHASE_EXECUTION,          // Execution phase
    PHASE_VALIDATION        // Validation phase
} optimization_phase_t;

// Optimizer configuration
typedef struct {
    optimization_strategy_t strategy;  // Optimization strategy
    optimization_goal_t goal;          // Primary goal
    cost_model_t cost_model;           // Cost model
    double convergence_threshold;       // Convergence threshold
    size_t max_iterations;             // Maximum iterations
    bool enable_backtracking;          // Enable backtracking
} optimizer_config_t;

// Optimization constraints
typedef struct {
    double max_load_imbalance;         // Maximum load imbalance
    double min_efficiency;             // Minimum efficiency
    size_t max_migrations;             // Maximum migrations
    double resource_limit;             // Resource usage limit
    double time_limit;                 // Time limit
    bool preserve_locality;            // Preserve data locality
} optimization_constraints_t;

// Optimization metrics
typedef struct {
    double load_balance;               // Load balance score
    double resource_efficiency;        // Resource efficiency
    double energy_efficiency;          // Energy efficiency
    double performance_gain;           // Performance improvement
    size_t migrations;                 // Number of migrations
    double convergence_rate;           // Convergence rate
} optimization_metrics_t;

// Migration plan
typedef struct {
    size_t source_node;                // Source node
    size_t target_node;                // Target node
    size_t workload_size;              // Workload size
    double estimated_cost;             // Migration cost
    double expected_benefit;           // Expected benefit
    struct timespec scheduled_time;    // Scheduled time
} migration_plan_t;

// Optimization result
typedef struct {
    bool success;                      // Optimization success
    optimization_metrics_t metrics;     // Result metrics
    size_t iterations;                 // Iterations taken
    double optimization_time;          // Time taken
    char* description;                 // Result description
    void* result_data;                // Additional data
} optimization_result_t;

// Opaque optimizer handle
typedef struct distribution_optimizer_t distribution_optimizer_t;

// Core functions
distribution_optimizer_t* create_distribution_optimizer(const optimizer_config_t* config);
void destroy_distribution_optimizer(distribution_optimizer_t* optimizer);

// Optimization functions
bool optimize_distribution(distribution_optimizer_t* optimizer,
                         const distribution_metrics_t* current,
                         optimization_result_t* result);
bool optimize_workload(distribution_optimizer_t* optimizer,
                      const workload_stats_t* workload,
                      optimization_result_t* result);
bool optimize_resource_usage(distribution_optimizer_t* optimizer,
                           const node_stats_t* nodes,
                           size_t num_nodes,
                           optimization_result_t* result);

// Planning functions
bool generate_migration_plan(distribution_optimizer_t* optimizer,
                           migration_plan_t* plan);
bool validate_migration_plan(distribution_optimizer_t* optimizer,
                           const migration_plan_t* plan);
bool execute_migration_plan(distribution_optimizer_t* optimizer,
                          const migration_plan_t* plan);

// Constraint management
bool set_optimization_constraints(distribution_optimizer_t* optimizer,
                                const optimization_constraints_t* constraints);
bool validate_constraints(distribution_optimizer_t* optimizer,
                         const optimization_result_t* result);
bool adjust_constraints(distribution_optimizer_t* optimizer,
                       const optimization_metrics_t* metrics);

// Cost analysis
bool analyze_migration_cost(distribution_optimizer_t* optimizer,
                          const migration_plan_t* plan,
                          double* cost);
bool estimate_optimization_cost(distribution_optimizer_t* optimizer,
                              const distribution_metrics_t* current,
                              double* cost);
bool validate_cost_model(distribution_optimizer_t* optimizer,
                        const optimization_metrics_t* metrics);

// Quantum-specific functions
bool optimize_quantum_distribution(distribution_optimizer_t* optimizer,
                                 const distribution_metrics_t* current,
                                 optimization_result_t* result);
bool generate_quantum_migration_plan(distribution_optimizer_t* optimizer,
                                   migration_plan_t* plan);
bool validate_quantum_optimization(distribution_optimizer_t* optimizer,
                                 const optimization_result_t* result);

// Utility functions
bool export_optimizer_data(const distribution_optimizer_t* optimizer,
                          const char* filename);
bool import_optimizer_data(distribution_optimizer_t* optimizer,
                          const char* filename);
void free_optimization_result(optimization_result_t* result);

#endif // DISTRIBUTION_OPTIMIZER_H
