#ifndef OPERATION_FUSION_H
#define OPERATION_FUSION_H

#include "computational_graph.h"
#include <stdbool.h>

// Fusion patterns
typedef enum {
    FUSION_SEQUENTIAL,    // Sequential operations fusion
    FUSION_PARALLEL,      // Parallel operations fusion
    FUSION_QUANTUM,       // Quantum operations fusion
    FUSION_HYBRID        // Hybrid operations fusion
} fusion_pattern_t;

// Fusion rules
typedef struct {
    operation_type_t op1;         // First operation type
    operation_type_t op2;         // Second operation type
    fusion_pattern_t pattern;     // Fusion pattern
    double cost_reduction;        // Expected cost reduction
    bool (*compatibility_check)(computation_node_t*, computation_node_t*);
} fusion_rule_t;

// Fusion group
typedef struct {
    computation_node_t** nodes;   // Nodes in fusion group
    size_t num_nodes;            // Number of nodes
    fusion_pattern_t pattern;    // Fusion pattern
    double cost_reduction;       // Total cost reduction
} fusion_group_t;

// Fusion optimizer configuration
typedef struct {
    bool enable_quantum_fusion;   // Enable quantum operation fusion
    bool enable_classical_fusion; // Enable classical operation fusion
    double min_cost_reduction;    // Minimum cost reduction threshold
    size_t max_group_size;       // Maximum fusion group size
    bool preserve_gradients;      // Preserve gradient computation capability
} fusion_config_t;

// Core functions
bool initialize_fusion_optimizer(fusion_config_t* config);
bool shutdown_fusion_optimizer(void);

// Fusion operations
bool analyze_fusion_opportunities(computational_graph_t* graph);
bool apply_fusion_rules(computational_graph_t* graph);
fusion_group_t* identify_fusion_groups(computational_graph_t* graph,
                                     size_t* num_groups);

// Rule management
bool register_fusion_rule(fusion_rule_t* rule);
bool remove_fusion_rule(operation_type_t op1, operation_type_t op2);
bool modify_fusion_rule(fusion_rule_t* rule);

// Pattern management
bool register_fusion_pattern(fusion_pattern_t pattern,
                           bool (*pattern_check)(computation_node_t**, size_t));
bool remove_fusion_pattern(fusion_pattern_t pattern);

// Cost analysis
typedef struct {
    double computation_cost;     // Computational cost
    double memory_cost;         // Memory usage cost
    double communication_cost;   // Communication cost
    double quantum_cost;        // Quantum resource cost
} fusion_cost_t;

bool analyze_fusion_cost(fusion_group_t* group, fusion_cost_t* cost);
bool estimate_cost_reduction(fusion_group_t* group, double* reduction);

// Optimization strategies
typedef enum {
    STRATEGY_GREEDY,           // Greedy optimization
    STRATEGY_EXHAUSTIVE,       // Exhaustive search
    STRATEGY_HEURISTIC,        // Heuristic-based
    STRATEGY_QUANTUM          // Quantum-assisted
} optimization_strategy_t;

bool set_optimization_strategy(optimization_strategy_t strategy);
bool optimize_fusion_groups(computational_graph_t* graph,
                          optimization_strategy_t strategy);

// Quantum-specific fusion
typedef struct {
    bool requires_coherence;    // Requires quantum coherence
    bool allows_measurement;    // Allows intermediate measurements
    size_t qubit_count;        // Required number of qubits
    double fidelity_threshold; // Minimum required fidelity
} quantum_fusion_constraints_t;

bool register_quantum_fusion_constraints(quantum_fusion_constraints_t* constraints);
bool validate_quantum_fusion(fusion_group_t* group);

// Utility functions
void print_fusion_statistics(computational_graph_t* graph);
bool export_fusion_rules(const char* filename);
bool import_fusion_rules(const char* filename);

// Performance monitoring
typedef struct {
    size_t fused_operations;    // Number of fused operations
    double total_cost_reduction; // Total cost reduction
    double fusion_overhead;      // Fusion overhead
    double effective_speedup;    // Effective speedup
} fusion_metrics_t;

bool collect_fusion_metrics(computational_graph_t* graph,
                          fusion_metrics_t* metrics);

#endif // OPERATION_FUSION_H
