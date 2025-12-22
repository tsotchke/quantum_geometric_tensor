#ifndef SYSTEM_ANALYZER_H
#define SYSTEM_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// System Analyzer Types
// ============================================================================

// Health status levels
typedef enum {
    HEALTH_OPTIMAL,       // System performing optimally
    HEALTH_GOOD,          // Minor issues, acceptable performance
    HEALTH_DEGRADED,      // Noticeable degradation
    HEALTH_WARNING,       // Significant issues requiring attention
    HEALTH_CRITICAL,      // Critical issues, intervention required
    HEALTH_FAILING        // System failing or failed
} health_status_t;

// Component types for health tracking
typedef enum {
    COMPONENT_CPU,
    COMPONENT_MEMORY,
    COMPONENT_DISK,
    COMPONENT_NETWORK,
    COMPONENT_GPU,
    COMPONENT_QUANTUM_BACKEND,
    COMPONENT_MPI,
    COMPONENT_LIBRARY_CORE,
    COMPONENT_COUNT
} component_type_t;

// Anomaly types
typedef enum {
    ANOMALY_SPIKE,           // Sudden spike in metric
    ANOMALY_DROP,            // Sudden drop in metric
    ANOMALY_TREND_CHANGE,    // Change in trend direction
    ANOMALY_VARIANCE_CHANGE, // Change in variance
    ANOMALY_OUTLIER,         // Statistical outlier
    ANOMALY_PERIODIC,        // Unusual periodic behavior
    ANOMALY_CORRELATION      // Unusual correlation between metrics
} anomaly_type_t;

// Trend direction
typedef enum {
    TREND_STABLE,
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_OSCILLATING,
    TREND_UNKNOWN
} trend_direction_t;

// Incident status
typedef enum {
    INCIDENT_OPEN,
    INCIDENT_ACKNOWLEDGED,
    INCIDENT_INVESTIGATING,
    INCIDENT_MITIGATING,
    INCIDENT_RESOLVED,
    INCIDENT_CLOSED
} incident_status_t;

// Health score for a component
typedef struct {
    component_type_t component;
    double score;                  // 0.0 (failing) to 100.0 (optimal)
    health_status_t status;
    char status_message[256];
    uint64_t last_check_ns;
    size_t issue_count;
} component_health_t;

// Overall system health
typedef struct {
    double overall_score;          // Weighted average of all components
    health_status_t overall_status;
    component_health_t components[COMPONENT_COUNT];
    size_t total_issues;
    size_t critical_issues;
    uint64_t uptime_seconds;
    uint64_t last_incident_ns;
} system_health_t;

// Detected anomaly
typedef struct {
    anomaly_type_t type;
    component_type_t component;
    char metric_name[64];
    double expected_value;
    double actual_value;
    double deviation_sigma;        // Number of standard deviations
    double confidence;             // Confidence in detection (0.0-1.0)
    uint64_t detection_time_ns;
    char description[256];
    bool is_critical;
} detected_anomaly_t;

// Trend analysis result
typedef struct {
    component_type_t component;
    char metric_name[64];
    trend_direction_t direction;
    double slope;                  // Rate of change per second
    double r_squared;              // Goodness of fit
    double predicted_value_1h;     // Predicted value in 1 hour
    double predicted_value_24h;    // Predicted value in 24 hours
    double time_to_threshold;      // Seconds until threshold breach (-1 if N/A)
    bool is_concerning;            // Whether trend is concerning
} trend_analysis_t;

// Capacity planning data
typedef struct {
    component_type_t component;
    double current_usage_percent;
    double peak_usage_percent;
    double avg_usage_percent;
    double growth_rate_per_day;    // Percent growth per day
    double days_until_80_percent;  // Days until 80% capacity
    double days_until_full;        // Days until 100% capacity
    size_t recommended_capacity;   // Recommended capacity increase
    char recommendation[512];
} capacity_plan_t;

// SLO (Service Level Objective) definition
typedef struct {
    char name[64];
    component_type_t component;
    char metric_name[64];
    double target_value;           // Target value for SLO
    bool is_upper_bound;           // true: <= target, false: >= target
    double error_budget_percent;   // Allowed error budget (e.g., 0.1 for 99.9%)
    double current_compliance;     // Current compliance rate (0.0-1.0)
    uint64_t violations_count;     // Number of violations
    double remaining_budget;       // Remaining error budget (0.0-1.0)
} slo_definition_t;

// Incident record
typedef struct {
    uint64_t id;
    incident_status_t status;
    component_type_t component;
    health_status_t severity;
    char title[128];
    char description[512];
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint64_t duration_seconds;
    size_t related_anomalies;
    double impact_score;           // Estimated impact (0.0-100.0)
    char root_cause[256];
    char resolution[256];
} incident_record_t;

// Correlation between metrics
typedef struct {
    char metric1_name[64];
    char metric2_name[64];
    component_type_t component1;
    component_type_t component2;
    double correlation;            // Pearson correlation (-1.0 to 1.0)
    double lag_seconds;            // Time lag between metrics
    bool is_significant;           // Statistical significance
    double p_value;                // P-value for significance test
} metric_correlation_t;

// System analyzer configuration
typedef struct {
    double health_check_interval_ms;
    double anomaly_detection_window_ms;
    double trend_analysis_window_ms;
    size_t max_anomaly_history;
    size_t max_incident_history;
    double anomaly_threshold_sigma;      // Standard deviations for anomaly
    bool enable_predictive_alerts;
    bool enable_correlation_analysis;
    double correlation_significance_threshold;
    size_t capacity_planning_horizon_days;
} system_analyzer_config_t;

// Opaque handle
typedef struct system_analyzer system_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

// Create system analyzer with default configuration
system_analyzer_t* system_analyzer_create(void);

// Create with custom configuration
system_analyzer_t* system_analyzer_create_with_config(
    const system_analyzer_config_t* config);

// Get default configuration
system_analyzer_config_t system_analyzer_default_config(void);

// Destroy system analyzer
void system_analyzer_destroy(system_analyzer_t* analyzer);

// Reset all data
bool system_analyzer_reset(system_analyzer_t* analyzer);

// ============================================================================
// Health Score Computation
// ============================================================================

// Compute overall system health
bool system_compute_health(system_analyzer_t* analyzer,
                          system_health_t* health);

// Get health for specific component
bool system_get_component_health(system_analyzer_t* analyzer,
                                 component_type_t component,
                                 component_health_t* health);

// Update component health manually
void system_update_component_health(system_analyzer_t* analyzer,
                                    component_type_t component,
                                    double score,
                                    const char* message);

// Set component weight for overall score
void system_set_component_weight(system_analyzer_t* analyzer,
                                 component_type_t component,
                                 double weight);

// Get health history
bool system_get_health_history(system_analyzer_t* analyzer,
                               system_health_t** history,
                               size_t* count);

// ============================================================================
// Anomaly Detection
// ============================================================================

// Run anomaly detection on recent data
bool system_detect_anomalies(system_analyzer_t* analyzer,
                             detected_anomaly_t** anomalies,
                             size_t* count);

// Record a metric value for anomaly detection
void system_record_metric(system_analyzer_t* analyzer,
                          component_type_t component,
                          const char* metric_name,
                          double value);

// Get anomaly history
bool system_get_anomaly_history(system_analyzer_t* analyzer,
                                detected_anomaly_t** anomalies,
                                size_t* count);

// Clear anomaly history
void system_clear_anomaly_history(system_analyzer_t* analyzer);

// Set anomaly detection sensitivity
void system_set_anomaly_sensitivity(system_analyzer_t* analyzer,
                                    double sigma_threshold);

// Register callback for anomaly detection
typedef void (*anomaly_callback_t)(const detected_anomaly_t* anomaly,
                                   void* user_data);
void system_set_anomaly_callback(system_analyzer_t* analyzer,
                                 anomaly_callback_t callback,
                                 void* user_data);

// ============================================================================
// Trend Analysis
// ============================================================================

// Analyze trend for a metric
bool system_analyze_trend(system_analyzer_t* analyzer,
                          component_type_t component,
                          const char* metric_name,
                          trend_analysis_t* trend);

// Analyze all trends
bool system_analyze_all_trends(system_analyzer_t* analyzer,
                               trend_analysis_t** trends,
                               size_t* count);

// Predict future value
double system_predict_value(system_analyzer_t* analyzer,
                            component_type_t component,
                            const char* metric_name,
                            double seconds_ahead);

// Get time until threshold breach
double system_time_to_threshold(system_analyzer_t* analyzer,
                                component_type_t component,
                                const char* metric_name,
                                double threshold);

// ============================================================================
// Capacity Planning
// ============================================================================

// Generate capacity plan for a component
bool system_generate_capacity_plan(system_analyzer_t* analyzer,
                                   component_type_t component,
                                   capacity_plan_t* plan);

// Generate capacity plans for all components
bool system_generate_all_capacity_plans(system_analyzer_t* analyzer,
                                        capacity_plan_t** plans,
                                        size_t* count);

// Set capacity threshold
void system_set_capacity_threshold(system_analyzer_t* analyzer,
                                   component_type_t component,
                                   double warning_percent,
                                   double critical_percent);

// Get capacity recommendations
char* system_get_capacity_recommendations(system_analyzer_t* analyzer);

// ============================================================================
// Service Level Objectives (SLO)
// ============================================================================

// Define an SLO
bool system_define_slo(system_analyzer_t* analyzer,
                       const slo_definition_t* slo);

// Get SLO status
bool system_get_slo_status(system_analyzer_t* analyzer,
                           const char* slo_name,
                           slo_definition_t* slo);

// Get all SLO statuses
bool system_get_all_slo_statuses(system_analyzer_t* analyzer,
                                 slo_definition_t** slos,
                                 size_t* count);

// Record SLO metric value
void system_record_slo_metric(system_analyzer_t* analyzer,
                              const char* slo_name,
                              double value,
                              bool is_compliant);

// Get SLO compliance over time
bool system_get_slo_compliance_history(system_analyzer_t* analyzer,
                                       const char* slo_name,
                                       double** compliance,
                                       uint64_t** timestamps,
                                       size_t* count);

// Get error budget burn rate
double system_get_error_budget_burn_rate(system_analyzer_t* analyzer,
                                         const char* slo_name);

// ============================================================================
// Incident Management
// ============================================================================

// Create a new incident
uint64_t system_create_incident(system_analyzer_t* analyzer,
                                component_type_t component,
                                health_status_t severity,
                                const char* title,
                                const char* description);

// Update incident status
bool system_update_incident(system_analyzer_t* analyzer,
                            uint64_t incident_id,
                            incident_status_t status,
                            const char* update_message);

// Resolve incident
bool system_resolve_incident(system_analyzer_t* analyzer,
                             uint64_t incident_id,
                             const char* root_cause,
                             const char* resolution);

// Get incident by ID
bool system_get_incident(system_analyzer_t* analyzer,
                         uint64_t incident_id,
                         incident_record_t* incident);

// Get all open incidents
bool system_get_open_incidents(system_analyzer_t* analyzer,
                               incident_record_t** incidents,
                               size_t* count);

// Get incident history
bool system_get_incident_history(system_analyzer_t* analyzer,
                                 incident_record_t** incidents,
                                 size_t* count);

// Correlate incident with anomalies
bool system_correlate_incident(system_analyzer_t* analyzer,
                               uint64_t incident_id,
                               detected_anomaly_t** related_anomalies,
                               size_t* count);

// ============================================================================
// Correlation Analysis
// ============================================================================

// Compute correlations between all metrics
bool system_compute_correlations(system_analyzer_t* analyzer,
                                 metric_correlation_t** correlations,
                                 size_t* count);

// Get correlation between two metrics
bool system_get_correlation(system_analyzer_t* analyzer,
                            component_type_t comp1,
                            const char* metric1,
                            component_type_t comp2,
                            const char* metric2,
                            metric_correlation_t* correlation);

// Find metrics correlated with a given metric
bool system_find_correlated_metrics(system_analyzer_t* analyzer,
                                    component_type_t component,
                                    const char* metric_name,
                                    double min_correlation,
                                    metric_correlation_t** correlations,
                                    size_t* count);

// ============================================================================
// Reporting and Export
// ============================================================================

// Generate system health report
char* system_generate_health_report(system_analyzer_t* analyzer);

// Generate incident report
char* system_generate_incident_report(system_analyzer_t* analyzer,
                                      uint64_t start_time_ns,
                                      uint64_t end_time_ns);

// Generate capacity planning report
char* system_generate_capacity_report(system_analyzer_t* analyzer);

// Generate SLO report
char* system_generate_slo_report(system_analyzer_t* analyzer);

// Export all data to JSON
char* system_export_json(system_analyzer_t* analyzer);

// Export all data to file
bool system_export_to_file(system_analyzer_t* analyzer,
                           const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

// Get health status name
const char* system_health_status_name(health_status_t status);

// Get component type name
const char* system_component_name(component_type_t component);

// Get anomaly type name
const char* system_anomaly_type_name(anomaly_type_t type);

// Get trend direction name
const char* system_trend_direction_name(trend_direction_t direction);

// Get incident status name
const char* system_incident_status_name(incident_status_t status);

// Get current timestamp in nanoseconds
uint64_t system_get_timestamp_ns(void);

// Get last error message
const char* system_get_last_error(void);

// Free allocated data
void system_free_anomalies(detected_anomaly_t* anomalies);
void system_free_trends(trend_analysis_t* trends);
void system_free_capacity_plans(capacity_plan_t* plans);
void system_free_slos(slo_definition_t* slos);
void system_free_incidents(incident_record_t* incidents);
void system_free_correlations(metric_correlation_t* correlations);
void system_free_health_history(system_health_t* history);

#ifdef __cplusplus
}
#endif

#endif // SYSTEM_ANALYZER_H
