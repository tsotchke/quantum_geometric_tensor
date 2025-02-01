#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Balancing strategies
typedef enum {
    STRATEGY_ROUND_ROBIN,   // Round-robin balancing
    STRATEGY_LEAST_LOADED,  // Least loaded first
    STRATEGY_WEIGHTED,      // Weighted distribution
    STRATEGY_ADAPTIVE      // Adaptive balancing
} balancing_strategy_t;

// Resource types
typedef enum {
    RESOURCE_CPU,           // CPU resources
    RESOURCE_MEMORY,        // Memory resources
    RESOURCE_QUANTUM,       // Quantum resources
    RESOURCE_NETWORK       // Network resources
} resource_type_t;

// Operation modes
typedef enum {
    MODE_STATIC,           // Static balancing
    MODE_DYNAMIC,          // Dynamic balancing
    MODE_PREDICTIVE,       // Predictive balancing
    MODE_HYBRID           // Hybrid balancing
} operation_mode_t;

// Migration types
typedef enum {
    MIGRATE_TASK,          // Task migration
    MIGRATE_PROCESS,       // Process migration
    MIGRATE_STATE,         // State migration
    MIGRATE_RESOURCE      // Resource migration
} migration_type_t;

// Balancer configuration
typedef struct {
    balancing_strategy_t strategy;  // Balancing strategy
    operation_mode_t mode;          // Operation mode
    bool enable_migration;          // Enable migration
    bool monitor_health;            // Monitor health
    size_t check_interval;         // Check interval
    double threshold;              // Balance threshold
} balancer_config_t;

// Node metrics
typedef struct {
    size_t node_id;                // Node identifier
    double load_factor;            // Load factor
    double capacity;               // Node capacity
    double utilization;            // Resource utilization
    size_t active_tasks;          // Active tasks
    bool is_available;            // Availability flag
} node_metrics_t;

// Load distribution
typedef struct {
    resource_type_t resource;      // Resource type
    double* distribution;          // Load distribution
    size_t num_nodes;             // Number of nodes
    double imbalance;             // Imbalance factor
    bool requires_rebalancing;    // Rebalancing flag
    void* distribution_data;      // Additional data
} load_distribution_t;

// Migration plan
typedef struct {
    migration_type_t type;         // Migration type
    size_t source_node;            // Source node
    size_t target_node;            // Target node
    double estimated_cost;         // Migration cost
    struct timespec planned_time;  // Planned time
    void* migration_data;         // Additional data
} migration_plan_t;

// Opaque balancer handle
typedef struct load_balancer_t load_balancer_t;

// Core functions
load_balancer_t* create_load_balancer(const balancer_config_t* config);
void destroy_load_balancer(load_balancer_t* balancer);

// Balancing functions
bool balance_load(load_balancer_t* balancer,
                 resource_type_t resource,
                 load_distribution_t* distribution);
bool rebalance_nodes(load_balancer_t* balancer,
                    const node_metrics_t* nodes,
                    size_t num_nodes);
bool validate_balance(load_balancer_t* balancer,
                     const load_distribution_t* distribution);

// Node management
bool register_node(load_balancer_t* balancer,
                  const node_metrics_t* node);
bool unregister_node(load_balancer_t* balancer,
                    size_t node_id);
bool update_node_metrics(load_balancer_t* balancer,
                        const node_metrics_t* metrics);

// Distribution functions
bool get_load_distribution(load_balancer_t* balancer,
                         resource_type_t resource,
                         load_distribution_t* distribution);
bool optimize_distribution(load_balancer_t* balancer,
                         load_distribution_t* distribution);
bool validate_distribution(load_balancer_t* balancer,
                         const load_distribution_t* distribution);

// Migration functions
bool plan_migration(load_balancer_t* balancer,
                   const load_distribution_t* distribution,
                   migration_plan_t* plan);
bool execute_migration(load_balancer_t* balancer,
                      const migration_plan_t* plan);
bool validate_migration(load_balancer_t* balancer,
                       const migration_plan_t* plan);

// Monitoring functions
bool monitor_balance(load_balancer_t* balancer,
                    resource_type_t resource,
                    load_distribution_t* distribution);
bool check_health(load_balancer_t* balancer,
                 node_metrics_t* nodes,
                 size_t* num_nodes);
bool get_balance_metrics(const load_balancer_t* balancer,
                        resource_type_t resource,
                        double* metrics);

// Quantum-specific functions
bool balance_quantum_load(load_balancer_t* balancer,
                        load_distribution_t* distribution);
bool migrate_quantum_state(load_balancer_t* balancer,
                         const migration_plan_t* plan);
bool validate_quantum_balance(load_balancer_t* balancer,
                            const load_distribution_t* distribution);

// Utility functions
bool export_balancer_data(const load_balancer_t* balancer,
                         const char* filename);
bool import_balancer_data(load_balancer_t* balancer,
                         const char* filename);
void free_distribution(load_distribution_t* distribution);

#endif // LOAD_BALANCER_H
