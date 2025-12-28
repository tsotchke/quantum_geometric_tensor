/**
 * @file utilization_optimizer.h
 * @brief Resource Utilization Optimization for Quantum Geometric Operations
 *
 * Provides resource utilization optimization including:
 * - Resource allocation optimization
 * - Utilization balancing
 * - Capacity planning
 * - Resource pooling strategies
 * - Waste reduction analysis
 * - Scaling recommendations
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef UTILIZATION_OPTIMIZER_H
#define UTILIZATION_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define UTILIZATION_OPT_MAX_NAME_LENGTH 128
#define UTILIZATION_OPT_MAX_RESOURCES 256
#define UTILIZATION_OPT_MAX_POOLS 64
#define UTILIZATION_OPT_HISTORY_SIZE 10000

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Resource types for optimization
 */
typedef enum {
    UTILIZATION_OPT_RESOURCE_CPU,
    UTILIZATION_OPT_RESOURCE_GPU,
    UTILIZATION_OPT_RESOURCE_MEMORY,
    UTILIZATION_OPT_RESOURCE_GPU_MEMORY,
    UTILIZATION_OPT_RESOURCE_QUBIT,
    UTILIZATION_OPT_RESOURCE_NETWORK,
    UTILIZATION_OPT_RESOURCE_STORAGE,
    UTILIZATION_OPT_RESOURCE_CUSTOM
} utilization_opt_resource_type_t;

/**
 * Optimization strategies
 */
typedef enum {
    UTILIZATION_OPT_STRATEGY_CONSOLIDATE,    // Consolidate workloads
    UTILIZATION_OPT_STRATEGY_DISTRIBUTE,     // Distribute workloads
    UTILIZATION_OPT_STRATEGY_SCALE_UP,       // Increase resource capacity
    UTILIZATION_OPT_STRATEGY_SCALE_DOWN,     // Decrease resource capacity
    UTILIZATION_OPT_STRATEGY_REBALANCE,      // Rebalance across resources
    UTILIZATION_OPT_STRATEGY_POOL,           // Use resource pooling
    UTILIZATION_OPT_STRATEGY_CACHE,          // Add caching
    UTILIZATION_OPT_STRATEGY_THROTTLE,       // Throttle usage
    UTILIZATION_OPT_STRATEGY_CUSTOM          // Custom strategy
} utilization_opt_strategy_t;

/**
 * Utilization zones
 */
typedef enum {
    UTILIZATION_ZONE_UNDERUTILIZED,   // < 30%
    UTILIZATION_ZONE_OPTIMAL,         // 30-70%
    UTILIZATION_ZONE_HIGH,            // 70-85%
    UTILIZATION_ZONE_OVERUTILIZED,    // 85-95%
    UTILIZATION_ZONE_CRITICAL         // > 95%
} utilization_zone_t;

/**
 * Scaling direction
 */
typedef enum {
    SCALING_NONE,
    SCALING_UP,
    SCALING_DOWN,
    SCALING_OUT,
    SCALING_IN
} scaling_direction_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Resource utilization snapshot
 */
typedef struct {
    char name[UTILIZATION_OPT_MAX_NAME_LENGTH];
    utilization_opt_resource_type_t type;
    double current_utilization;       // 0.0 to 1.0
    double capacity;
    double used;
    double available;
    utilization_zone_t zone;
    uint64_t timestamp_ns;
} utilization_snapshot_t;

/**
 * Resource allocation
 */
typedef struct {
    uint32_t resource_id;
    uint32_t consumer_id;
    double allocated;
    double used;
    double efficiency;                // used/allocated
    uint64_t allocation_time_ns;
} utilization_allocation_t;

/**
 * Resource pool configuration
 */
typedef struct {
    char name[UTILIZATION_OPT_MAX_NAME_LENGTH];
    utilization_opt_resource_type_t type;
    double total_capacity;
    double min_allocation;
    double max_allocation;
    double target_utilization;        // Target zone (e.g., 0.6 for 60%)
    bool elastic;                     // Can grow/shrink
    double elasticity_factor;         // How much it can scale
} utilization_pool_config_t;

/**
 * Pool metrics
 */
typedef struct {
    char name[UTILIZATION_OPT_MAX_NAME_LENGTH];
    double total_capacity;
    double allocated;
    double used;
    double available;
    double utilization;
    double efficiency;
    size_t num_allocations;
    double fragmentation;             // 0.0 (none) to 1.0 (severe)
    utilization_zone_t zone;
} utilization_pool_metrics_t;

/**
 * Capacity planning result
 */
typedef struct {
    utilization_opt_resource_type_t resource_type;
    double current_capacity;
    double recommended_capacity;
    double growth_rate;               // Per time unit
    uint64_t time_to_exhaustion_ns;
    scaling_direction_t recommended_scaling;
    double scaling_factor;
    char rationale[256];
} utilization_capacity_plan_t;

/**
 * Waste analysis result
 */
typedef struct {
    char resource_name[UTILIZATION_OPT_MAX_NAME_LENGTH];
    double allocated_unused;          // Allocated but unused
    double waste_percent;
    double cost_impact;               // Relative cost of waste
    char recommendation[256];
    utilization_opt_strategy_t suggested_strategy;
} utilization_waste_analysis_t;

/**
 * Balancing recommendation
 */
typedef struct {
    uint32_t from_resource_id;
    uint32_t to_resource_id;
    double amount_to_move;
    double expected_improvement;
    char description[256];
} utilization_balance_rec_t;

/**
 * Optimization action
 */
typedef struct {
    utilization_opt_strategy_t strategy;
    char target_resource[UTILIZATION_OPT_MAX_NAME_LENGTH];
    double current_value;
    double target_value;
    double expected_improvement;
    double implementation_cost;
    uint32_t priority;                // 1 = highest
    char description[256];
    char prerequisites[256];
} utilization_action_t;

/**
 * Overall optimization metrics
 */
typedef struct {
    double avg_utilization;
    double utilization_std_dev;
    double overall_efficiency;
    double total_waste_percent;
    size_t resources_underutilized;
    size_t resources_optimal;
    size_t resources_overutilized;
    size_t resources_critical;
    double balance_score;             // 0 (imbalanced) to 1 (balanced)
    uint64_t analysis_timestamp_ns;
} utilization_opt_metrics_t;

/**
 * Historical trend
 */
typedef struct {
    char resource_name[UTILIZATION_OPT_MAX_NAME_LENGTH];
    double current_utilization;
    double avg_7day;
    double avg_30day;
    double trend_slope;               // Positive = increasing
    double predicted_utilization;     // At horizon
    uint64_t prediction_horizon_ns;
    bool capacity_risk;               // Risk of capacity issue
} utilization_trend_t;

/**
 * Optimizer configuration
 */
typedef struct {
    double underutilized_threshold;   // Below this = underutilized
    double optimal_min_threshold;     // Above this = optimal zone
    double optimal_max_threshold;     // Below this = still optimal
    double overutilized_threshold;    // Above this = overutilized
    double critical_threshold;        // Above this = critical
    bool enable_auto_rebalancing;
    bool enable_capacity_planning;
    bool enable_waste_detection;
    uint64_t analysis_interval_ns;
    double rebalance_threshold;       // Min imbalance to trigger
} utilization_optimizer_config_t;

/**
 * Opaque optimizer handle
 */
typedef struct utilization_optimizer utilization_optimizer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create utilization optimizer with default configuration
 */
utilization_optimizer_t* utilization_optimizer_create(void);

/**
 * Create with custom configuration
 */
utilization_optimizer_t* utilization_optimizer_create_with_config(
    const utilization_optimizer_config_t* config);

/**
 * Get default configuration
 */
utilization_optimizer_config_t utilization_optimizer_default_config(void);

/**
 * Destroy utilization optimizer
 */
void utilization_optimizer_destroy(utilization_optimizer_t* optimizer);

/**
 * Reset all state
 */
bool utilization_optimizer_reset(utilization_optimizer_t* optimizer);

// ============================================================================
// Resource Registration
// ============================================================================

/**
 * Register resource for optimization
 */
uint32_t utilization_opt_register_resource(
    utilization_optimizer_t* optimizer,
    const char* name,
    utilization_opt_resource_type_t type,
    double capacity);

/**
 * Update resource capacity
 */
bool utilization_opt_update_capacity(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    double new_capacity);

/**
 * Remove resource
 */
bool utilization_opt_remove_resource(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id);

// ============================================================================
// Pool Management
// ============================================================================

/**
 * Create resource pool
 */
bool utilization_opt_create_pool(
    utilization_optimizer_t* optimizer,
    const utilization_pool_config_t* config);

/**
 * Add resource to pool
 */
bool utilization_opt_add_to_pool(
    utilization_optimizer_t* optimizer,
    const char* pool_name,
    uint32_t resource_id);

/**
 * Remove resource from pool
 */
bool utilization_opt_remove_from_pool(
    utilization_optimizer_t* optimizer,
    const char* pool_name,
    uint32_t resource_id);

/**
 * Get pool metrics
 */
bool utilization_opt_get_pool_metrics(
    utilization_optimizer_t* optimizer,
    const char* pool_name,
    utilization_pool_metrics_t* metrics);

// ============================================================================
// Utilization Recording
// ============================================================================

/**
 * Record utilization measurement
 */
bool utilization_opt_record(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    double used,
    double allocated);

/**
 * Record utilization as percentage
 */
bool utilization_opt_record_percent(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    double utilization_percent);

/**
 * Record allocation
 */
bool utilization_opt_record_allocation(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    uint32_t consumer_id,
    double amount);

/**
 * Record deallocation
 */
bool utilization_opt_record_deallocation(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    uint32_t consumer_id);

// ============================================================================
// Analysis
// ============================================================================

/**
 * Get current utilization snapshot
 */
bool utilization_opt_get_snapshot(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    utilization_snapshot_t* snapshot);

/**
 * Get all snapshots
 */
bool utilization_opt_get_all_snapshots(
    utilization_optimizer_t* optimizer,
    utilization_snapshot_t** snapshots,
    size_t* count);

/**
 * Get overall metrics
 */
bool utilization_opt_get_metrics(
    utilization_optimizer_t* optimizer,
    utilization_opt_metrics_t* metrics);

/**
 * Get utilization trend
 */
bool utilization_opt_get_trend(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    utilization_trend_t* trend);

/**
 * Detect waste
 */
bool utilization_opt_detect_waste(
    utilization_optimizer_t* optimizer,
    utilization_waste_analysis_t** analyses,
    size_t* count);

/**
 * Get imbalanced resources
 */
bool utilization_opt_get_imbalanced(
    utilization_optimizer_t* optimizer,
    utilization_snapshot_t** snapshots,
    size_t* count);

// ============================================================================
// Optimization
// ============================================================================

/**
 * Generate optimization actions
 */
bool utilization_opt_generate_actions(
    utilization_optimizer_t* optimizer,
    utilization_action_t** actions,
    size_t* count);

/**
 * Get balancing recommendations
 */
bool utilization_opt_get_balance_recommendations(
    utilization_optimizer_t* optimizer,
    utilization_balance_rec_t** recommendations,
    size_t* count);

/**
 * Generate capacity plan
 */
bool utilization_opt_generate_capacity_plan(
    utilization_optimizer_t* optimizer,
    uint64_t planning_horizon_ns,
    utilization_capacity_plan_t** plans,
    size_t* count);

/**
 * Apply optimization strategy
 */
bool utilization_opt_apply_strategy(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    utilization_opt_strategy_t strategy);

/**
 * Auto-optimize (if enabled)
 */
bool utilization_opt_auto_optimize(utilization_optimizer_t* optimizer);

/**
 * Simulate scaling
 */
bool utilization_opt_simulate_scaling(
    utilization_optimizer_t* optimizer,
    uint32_t resource_id,
    scaling_direction_t direction,
    double factor,
    utilization_snapshot_t* predicted_snapshot);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate optimization report
 */
char* utilization_opt_generate_report(utilization_optimizer_t* optimizer);

/**
 * Export to JSON
 */
char* utilization_opt_export_json(utilization_optimizer_t* optimizer);

/**
 * Export to file
 */
bool utilization_opt_export_to_file(
    utilization_optimizer_t* optimizer,
    const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get resource type name
 */
const char* utilization_opt_resource_type_name(utilization_opt_resource_type_t type);

/**
 * Get strategy name
 */
const char* utilization_opt_strategy_name(utilization_opt_strategy_t strategy);

/**
 * Get zone name
 */
const char* utilization_opt_zone_name(utilization_zone_t zone);

/**
 * Get scaling direction name
 */
const char* utilization_opt_scaling_name(scaling_direction_t direction);

/**
 * Classify utilization zone
 */
utilization_zone_t utilization_opt_classify_zone(
    const utilization_optimizer_config_t* config,
    double utilization);

/**
 * Free snapshots array
 */
void utilization_opt_free_snapshots(utilization_snapshot_t* snapshots, size_t count);

/**
 * Free actions array
 */
void utilization_opt_free_actions(utilization_action_t* actions, size_t count);

/**
 * Free balance recommendations array
 */
void utilization_opt_free_balance_recs(utilization_balance_rec_t* recs, size_t count);

/**
 * Free capacity plans array
 */
void utilization_opt_free_capacity_plans(utilization_capacity_plan_t* plans, size_t count);

/**
 * Free waste analyses array
 */
void utilization_opt_free_waste_analyses(utilization_waste_analysis_t* analyses, size_t count);

/**
 * Get last error message
 */
const char* utilization_opt_get_last_error(utilization_optimizer_t* optimizer);

#ifdef __cplusplus
}
#endif

#endif // UTILIZATION_OPTIMIZER_H
