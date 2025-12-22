#include "quantum_geometric/core/system_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_METRICS 1000
#define MAX_SAMPLES_PER_METRIC 10000
#define MAX_ANOMALIES 1000
#define MAX_INCIDENTS 500
#define MAX_SLOS 100
#define MAX_CORRELATIONS 1000
#define METRIC_HASH_SIZE 256

// ============================================================================
// Internal Structures
// ============================================================================

// Metric sample
typedef struct {
    double value;
    uint64_t timestamp_ns;
} metric_sample_t;

// Metric storage
typedef struct metric_storage {
    char name[64];
    component_type_t component;
    metric_sample_t* samples;
    size_t sample_count;
    size_t sample_capacity;
    double mean;
    double variance;
    double std_dev;
    struct metric_storage* next;
} metric_storage_t;

// Main analyzer structure
struct system_analyzer {
    system_analyzer_config_t config;

    // Component health
    component_health_t components[COMPONENT_COUNT];
    double component_weights[COMPONENT_COUNT];

    // Health history
    system_health_t* health_history;
    size_t health_history_count;
    size_t health_history_capacity;

    // Metric storage (hash table)
    metric_storage_t* metrics[METRIC_HASH_SIZE];
    size_t metric_count;

    // Anomaly detection
    detected_anomaly_t* anomaly_history;
    size_t anomaly_count;
    size_t anomaly_capacity;
    anomaly_callback_t anomaly_callback;
    void* anomaly_user_data;
    double anomaly_sigma_threshold;

    // Incidents
    incident_record_t* incidents;
    size_t incident_count;
    size_t incident_capacity;
    uint64_t next_incident_id;

    // SLOs
    slo_definition_t* slos;
    size_t slo_count;
    size_t slo_capacity;

    // Correlations
    metric_correlation_t* correlations;
    size_t correlation_count;
    size_t correlation_capacity;

    // Capacity thresholds
    double capacity_warning[COMPONENT_COUNT];
    double capacity_critical[COMPONENT_COUNT];

    // Start time for uptime calculation
    uint64_t start_time_ns;

    // Thread safety
    pthread_mutex_t mutex;

    // Error message
    char last_error[256];
};

// Thread-local error
static __thread char tls_error[256] = {0};

static void set_error(system_analyzer_t* analyzer, const char* msg) {
    if (analyzer) {
        strncpy(analyzer->last_error, msg, sizeof(analyzer->last_error) - 1);
    }
    strncpy(tls_error, msg, sizeof(tls_error) - 1);
}

// ============================================================================
// Timestamp
// ============================================================================

uint64_t system_get_timestamp_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return (mach_absolute_time() * timebase.numer) / timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// ============================================================================
// Hash Function
// ============================================================================

static size_t hash_metric_name(component_type_t comp, const char* name) {
    uint32_t hash = (uint32_t)comp * 31;
    while (*name) {
        hash = ((hash << 5) + hash) + (unsigned char)*name++;
    }
    return hash % METRIC_HASH_SIZE;
}

// ============================================================================
// Initialization
// ============================================================================

system_analyzer_config_t system_analyzer_default_config(void) {
    return (system_analyzer_config_t){
        .health_check_interval_ms = 1000.0,
        .anomaly_detection_window_ms = 60000.0,
        .trend_analysis_window_ms = 300000.0,
        .max_anomaly_history = MAX_ANOMALIES,
        .max_incident_history = MAX_INCIDENTS,
        .anomaly_threshold_sigma = 3.0,
        .enable_predictive_alerts = true,
        .enable_correlation_analysis = true,
        .correlation_significance_threshold = 0.05,
        .capacity_planning_horizon_days = 30
    };
}

system_analyzer_t* system_analyzer_create(void) {
    system_analyzer_config_t config = system_analyzer_default_config();
    return system_analyzer_create_with_config(&config);
}

system_analyzer_t* system_analyzer_create_with_config(
    const system_analyzer_config_t* config) {

    if (!config) return NULL;

    system_analyzer_t* analyzer = calloc(1, sizeof(system_analyzer_t));
    if (!analyzer) return NULL;

    analyzer->config = *config;
    pthread_mutex_init(&analyzer->mutex, NULL);

    // Initialize component health
    for (int i = 0; i < COMPONENT_COUNT; i++) {
        analyzer->components[i].component = (component_type_t)i;
        analyzer->components[i].score = 100.0;
        analyzer->components[i].status = HEALTH_OPTIMAL;
        analyzer->component_weights[i] = 1.0;
    }

    // Allocate history buffers
    analyzer->health_history_capacity = 1000;
    analyzer->health_history = calloc(analyzer->health_history_capacity,
                                      sizeof(system_health_t));

    analyzer->anomaly_capacity = config->max_anomaly_history;
    analyzer->anomaly_history = calloc(analyzer->anomaly_capacity,
                                       sizeof(detected_anomaly_t));
    analyzer->anomaly_sigma_threshold = config->anomaly_threshold_sigma;

    analyzer->incident_capacity = config->max_incident_history;
    analyzer->incidents = calloc(analyzer->incident_capacity,
                                 sizeof(incident_record_t));
    analyzer->next_incident_id = 1;

    analyzer->slo_capacity = MAX_SLOS;
    analyzer->slos = calloc(analyzer->slo_capacity, sizeof(slo_definition_t));

    analyzer->correlation_capacity = MAX_CORRELATIONS;
    analyzer->correlations = calloc(analyzer->correlation_capacity,
                                    sizeof(metric_correlation_t));

    // Default capacity thresholds
    for (int i = 0; i < COMPONENT_COUNT; i++) {
        analyzer->capacity_warning[i] = 0.70;
        analyzer->capacity_critical[i] = 0.90;
    }

    analyzer->start_time_ns = system_get_timestamp_ns();

    return analyzer;
}

void system_analyzer_destroy(system_analyzer_t* analyzer) {
    if (!analyzer) return;

    // Free metrics
    for (size_t i = 0; i < METRIC_HASH_SIZE; i++) {
        metric_storage_t* m = analyzer->metrics[i];
        while (m) {
            metric_storage_t* next = m->next;
            free(m->samples);
            free(m);
            m = next;
        }
    }

    free(analyzer->health_history);
    free(analyzer->anomaly_history);
    free(analyzer->incidents);
    free(analyzer->slos);
    free(analyzer->correlations);

    pthread_mutex_destroy(&analyzer->mutex);
    free(analyzer);
}

bool system_analyzer_reset(system_analyzer_t* analyzer) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Reset health
    for (int i = 0; i < COMPONENT_COUNT; i++) {
        analyzer->components[i].score = 100.0;
        analyzer->components[i].status = HEALTH_OPTIMAL;
        analyzer->components[i].issue_count = 0;
    }

    // Clear metrics
    for (size_t i = 0; i < METRIC_HASH_SIZE; i++) {
        metric_storage_t* m = analyzer->metrics[i];
        while (m) {
            m->sample_count = 0;
            m = m->next;
        }
    }

    // Clear history
    analyzer->health_history_count = 0;
    analyzer->anomaly_count = 0;
    analyzer->incident_count = 0;
    analyzer->correlation_count = 0;

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Health Score Computation
// ============================================================================

static health_status_t score_to_status(double score) {
    if (score >= 95.0) return HEALTH_OPTIMAL;
    if (score >= 80.0) return HEALTH_GOOD;
    if (score >= 60.0) return HEALTH_DEGRADED;
    if (score >= 40.0) return HEALTH_WARNING;
    if (score >= 20.0) return HEALTH_CRITICAL;
    return HEALTH_FAILING;
}

bool system_compute_health(system_analyzer_t* analyzer,
                          system_health_t* health) {
    if (!analyzer || !health) return false;

    pthread_mutex_lock(&analyzer->mutex);

    memset(health, 0, sizeof(system_health_t));

    // Copy component health
    memcpy(health->components, analyzer->components, sizeof(health->components));

    // Calculate weighted average
    double total_weight = 0;
    double weighted_sum = 0;
    size_t total_issues = 0;
    size_t critical_issues = 0;

    for (int i = 0; i < COMPONENT_COUNT; i++) {
        weighted_sum += analyzer->components[i].score * analyzer->component_weights[i];
        total_weight += analyzer->component_weights[i];
        total_issues += analyzer->components[i].issue_count;

        if (analyzer->components[i].status >= HEALTH_CRITICAL) {
            critical_issues += analyzer->components[i].issue_count;
        }
    }

    if (total_weight > 0) {
        health->overall_score = weighted_sum / total_weight;
    } else {
        health->overall_score = 100.0;
    }

    health->overall_status = score_to_status(health->overall_score);
    health->total_issues = total_issues;
    health->critical_issues = critical_issues;

    // Calculate uptime
    uint64_t now = system_get_timestamp_ns();
    health->uptime_seconds = (now - analyzer->start_time_ns) / 1000000000ULL;

    // Find last incident time
    if (analyzer->incident_count > 0) {
        health->last_incident_ns =
            analyzer->incidents[analyzer->incident_count - 1].start_time_ns;
    }

    // Store in history
    if (analyzer->health_history_count < analyzer->health_history_capacity) {
        analyzer->health_history[analyzer->health_history_count++] = *health;
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_get_component_health(system_analyzer_t* analyzer,
                                 component_type_t component,
                                 component_health_t* health) {
    if (!analyzer || !health || component >= COMPONENT_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);
    *health = analyzer->components[component];
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

void system_update_component_health(system_analyzer_t* analyzer,
                                    component_type_t component,
                                    double score,
                                    const char* message) {
    if (!analyzer || component >= COMPONENT_COUNT) return;

    pthread_mutex_lock(&analyzer->mutex);

    analyzer->components[component].score = score;
    analyzer->components[component].status = score_to_status(score);
    analyzer->components[component].last_check_ns = system_get_timestamp_ns();

    if (message) {
        strncpy(analyzer->components[component].status_message, message,
                sizeof(analyzer->components[component].status_message) - 1);
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

void system_set_component_weight(system_analyzer_t* analyzer,
                                 component_type_t component,
                                 double weight) {
    if (!analyzer || component >= COMPONENT_COUNT) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->component_weights[component] = weight;
    pthread_mutex_unlock(&analyzer->mutex);
}

bool system_get_health_history(system_analyzer_t* analyzer,
                               system_health_t** history,
                               size_t* count) {
    if (!analyzer || !history || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->health_history_count;
    if (*count == 0) {
        *history = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *history = calloc(*count, sizeof(system_health_t));
    if (*history) {
        memcpy(*history, analyzer->health_history,
               *count * sizeof(system_health_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *history != NULL;
}

// ============================================================================
// Metric Recording and Storage
// ============================================================================

static metric_storage_t* find_or_create_metric(system_analyzer_t* analyzer,
                                               component_type_t component,
                                               const char* name) {
    size_t hash = hash_metric_name(component, name);
    metric_storage_t* m = analyzer->metrics[hash];

    while (m) {
        if (m->component == component && strcmp(m->name, name) == 0) {
            return m;
        }
        m = m->next;
    }

    // Create new
    m = calloc(1, sizeof(metric_storage_t));
    if (!m) return NULL;

    strncpy(m->name, name, sizeof(m->name) - 1);
    m->component = component;
    m->sample_capacity = MAX_SAMPLES_PER_METRIC;
    m->samples = calloc(m->sample_capacity, sizeof(metric_sample_t));

    if (!m->samples) {
        free(m);
        return NULL;
    }

    m->next = analyzer->metrics[hash];
    analyzer->metrics[hash] = m;
    analyzer->metric_count++;

    return m;
}

void system_record_metric(system_analyzer_t* analyzer,
                          component_type_t component,
                          const char* metric_name,
                          double value) {
    if (!analyzer || !metric_name || component >= COMPONENT_COUNT) return;

    pthread_mutex_lock(&analyzer->mutex);

    metric_storage_t* m = find_or_create_metric(analyzer, component, metric_name);
    if (m && m->sample_count < m->sample_capacity) {
        m->samples[m->sample_count].value = value;
        m->samples[m->sample_count].timestamp_ns = system_get_timestamp_ns();
        m->sample_count++;

        // Update running statistics
        if (m->sample_count == 1) {
            m->mean = value;
            m->variance = 0;
            m->std_dev = 0;
        } else {
            // Welford's online algorithm
            double n = (double)m->sample_count;
            double delta = value - m->mean;
            m->mean += delta / n;
            double delta2 = value - m->mean;
            m->variance += delta * delta2;
            m->std_dev = sqrt(m->variance / (n - 1));
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Anomaly Detection
// ============================================================================

static void add_anomaly(system_analyzer_t* analyzer,
                        anomaly_type_t type,
                        component_type_t component,
                        const char* metric_name,
                        double expected,
                        double actual,
                        double deviation_sigma,
                        const char* description) {
    if (analyzer->anomaly_count >= analyzer->anomaly_capacity) return;

    detected_anomaly_t* a = &analyzer->anomaly_history[analyzer->anomaly_count++];
    a->type = type;
    a->component = component;
    strncpy(a->metric_name, metric_name, sizeof(a->metric_name) - 1);
    a->expected_value = expected;
    a->actual_value = actual;
    a->deviation_sigma = deviation_sigma;
    a->confidence = fmin(1.0, fabs(deviation_sigma) / 5.0);
    a->detection_time_ns = system_get_timestamp_ns();
    strncpy(a->description, description, sizeof(a->description) - 1);
    a->is_critical = fabs(deviation_sigma) > 5.0;

    if (analyzer->anomaly_callback) {
        analyzer->anomaly_callback(a, analyzer->anomaly_user_data);
    }
}

bool system_detect_anomalies(system_analyzer_t* analyzer,
                             detected_anomaly_t** anomalies,
                             size_t* count) {
    if (!analyzer || !anomalies || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    size_t start_count = analyzer->anomaly_count;

    // Check each metric for anomalies
    for (size_t i = 0; i < METRIC_HASH_SIZE; i++) {
        metric_storage_t* m = analyzer->metrics[i];
        while (m) {
            if (m->sample_count >= 10 && m->std_dev > 0) {
                // Check last few samples for anomalies
                size_t check_count = fmin(5, m->sample_count);

                for (size_t j = m->sample_count - check_count; j < m->sample_count; j++) {
                    double z_score = (m->samples[j].value - m->mean) / m->std_dev;

                    if (fabs(z_score) > analyzer->anomaly_sigma_threshold) {
                        char desc[256];
                        snprintf(desc, sizeof(desc),
                                 "Value %.2f is %.1f sigma from mean %.2f",
                                 m->samples[j].value, z_score, m->mean);

                        anomaly_type_t type = z_score > 0 ? ANOMALY_SPIKE : ANOMALY_DROP;
                        add_anomaly(analyzer, type, m->component, m->name,
                                   m->mean, m->samples[j].value, z_score, desc);
                    }
                }
            }
            m = m->next;
        }
    }

    // Return newly detected anomalies
    size_t new_count = analyzer->anomaly_count - start_count;
    *count = new_count;

    if (new_count > 0) {
        *anomalies = calloc(new_count, sizeof(detected_anomaly_t));
        if (*anomalies) {
            memcpy(*anomalies, &analyzer->anomaly_history[start_count],
                   new_count * sizeof(detected_anomaly_t));
        }
    } else {
        *anomalies = NULL;
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_get_anomaly_history(system_analyzer_t* analyzer,
                                detected_anomaly_t** anomalies,
                                size_t* count) {
    if (!analyzer || !anomalies || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->anomaly_count;
    if (*count == 0) {
        *anomalies = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *anomalies = calloc(*count, sizeof(detected_anomaly_t));
    if (*anomalies) {
        memcpy(*anomalies, analyzer->anomaly_history,
               *count * sizeof(detected_anomaly_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *anomalies != NULL;
}

void system_clear_anomaly_history(system_analyzer_t* analyzer) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->anomaly_count = 0;
    pthread_mutex_unlock(&analyzer->mutex);
}

void system_set_anomaly_sensitivity(system_analyzer_t* analyzer,
                                    double sigma_threshold) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->anomaly_sigma_threshold = sigma_threshold;
    pthread_mutex_unlock(&analyzer->mutex);
}

void system_set_anomaly_callback(system_analyzer_t* analyzer,
                                 anomaly_callback_t callback,
                                 void* user_data) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->anomaly_callback = callback;
    analyzer->anomaly_user_data = user_data;
    pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Trend Analysis
// ============================================================================

bool system_analyze_trend(system_analyzer_t* analyzer,
                          component_type_t component,
                          const char* metric_name,
                          trend_analysis_t* trend) {
    if (!analyzer || !metric_name || !trend || component >= COMPONENT_COUNT) {
        return false;
    }

    pthread_mutex_lock(&analyzer->mutex);

    size_t hash = hash_metric_name(component, metric_name);
    metric_storage_t* m = analyzer->metrics[hash];

    while (m && (m->component != component || strcmp(m->name, metric_name) != 0)) {
        m = m->next;
    }

    if (!m || m->sample_count < 2) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    // Linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    uint64_t base_time = m->samples[0].timestamp_ns;

    for (size_t i = 0; i < m->sample_count; i++) {
        double x = (double)(m->samples[i].timestamp_ns - base_time) / 1e9; // seconds
        double y = m->samples[i].value;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    double n = (double)m->sample_count;
    double denom = n * sum_xx - sum_x * sum_x;

    trend->component = component;
    strncpy(trend->metric_name, metric_name, sizeof(trend->metric_name) - 1);

    if (fabs(denom) > 1e-10) {
        trend->slope = (n * sum_xy - sum_x * sum_y) / denom;
        double intercept = (sum_y - trend->slope * sum_x) / n;

        // Calculate R-squared
        double mean_y = sum_y / n;
        double ss_tot = 0, ss_res = 0;

        for (size_t i = 0; i < m->sample_count; i++) {
            double x = (double)(m->samples[i].timestamp_ns - base_time) / 1e9;
            double y = m->samples[i].value;
            double y_pred = trend->slope * x + intercept;

            ss_tot += (y - mean_y) * (y - mean_y);
            ss_res += (y - y_pred) * (y - y_pred);
        }

        trend->r_squared = (ss_tot > 0) ? 1.0 - (ss_res / ss_tot) : 0;

        // Determine direction
        if (fabs(trend->slope) < 1e-6) {
            trend->direction = TREND_STABLE;
        } else if (trend->slope > 0) {
            trend->direction = TREND_INCREASING;
        } else {
            trend->direction = TREND_DECREASING;
        }

        // Predictions
        double last_x = (double)(m->samples[m->sample_count-1].timestamp_ns - base_time) / 1e9;
        trend->predicted_value_1h = trend->slope * (last_x + 3600) + intercept;
        trend->predicted_value_24h = trend->slope * (last_x + 86400) + intercept;

        // Time to threshold (if trending up toward capacity)
        trend->time_to_threshold = -1;
        double threshold = analyzer->capacity_warning[component] * 100.0;
        if (trend->slope > 0 && m->mean < threshold) {
            double current = m->samples[m->sample_count-1].value;
            if (current < threshold) {
                trend->time_to_threshold = (threshold - current) / trend->slope;
            }
        }

        trend->is_concerning = (trend->direction != TREND_STABLE) &&
                               (trend->r_squared > 0.7) &&
                               (trend->time_to_threshold > 0 &&
                                trend->time_to_threshold < 86400 * 7);  // < 1 week
    } else {
        trend->slope = 0;
        trend->direction = TREND_STABLE;
        trend->r_squared = 0;
        trend->predicted_value_1h = m->mean;
        trend->predicted_value_24h = m->mean;
        trend->time_to_threshold = -1;
        trend->is_concerning = false;
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_analyze_all_trends(system_analyzer_t* analyzer,
                               trend_analysis_t** trends,
                               size_t* count) {
    if (!analyzer || !trends || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Count metrics
    *count = analyzer->metric_count;
    if (*count == 0) {
        *trends = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *trends = calloc(*count, sizeof(trend_analysis_t));
    if (!*trends) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < METRIC_HASH_SIZE && idx < *count; i++) {
        metric_storage_t* m = analyzer->metrics[i];
        while (m && idx < *count) {
            pthread_mutex_unlock(&analyzer->mutex);
            system_analyze_trend(analyzer, m->component, m->name, &(*trends)[idx]);
            pthread_mutex_lock(&analyzer->mutex);
            idx++;
            m = m->next;
        }
    }
    *count = idx;

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

double system_predict_value(system_analyzer_t* analyzer,
                            component_type_t component,
                            const char* metric_name,
                            double seconds_ahead) {
    if (!analyzer || !metric_name || component >= COMPONENT_COUNT) return 0;

    trend_analysis_t trend;
    if (!system_analyze_trend(analyzer, component, metric_name, &trend)) {
        return 0;
    }

    pthread_mutex_lock(&analyzer->mutex);

    size_t hash = hash_metric_name(component, metric_name);
    metric_storage_t* m = analyzer->metrics[hash];
    while (m && (m->component != component || strcmp(m->name, metric_name) != 0)) {
        m = m->next;
    }

    double result = 0;
    if (m && m->sample_count > 0) {
        result = m->samples[m->sample_count-1].value + trend.slope * seconds_ahead;
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return result;
}

double system_time_to_threshold(system_analyzer_t* analyzer,
                                component_type_t component,
                                const char* metric_name,
                                double threshold) {
    if (!analyzer || !metric_name || component >= COMPONENT_COUNT) return -1;

    trend_analysis_t trend;
    if (!system_analyze_trend(analyzer, component, metric_name, &trend)) {
        return -1;
    }

    if (trend.slope <= 0) return -1;  // Not trending toward threshold

    pthread_mutex_lock(&analyzer->mutex);

    size_t hash = hash_metric_name(component, metric_name);
    metric_storage_t* m = analyzer->metrics[hash];
    while (m && (m->component != component || strcmp(m->name, metric_name) != 0)) {
        m = m->next;
    }

    double result = -1;
    if (m && m->sample_count > 0) {
        double current = m->samples[m->sample_count-1].value;
        if (current < threshold) {
            result = (threshold - current) / trend.slope;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return result;
}

// ============================================================================
// Capacity Planning
// ============================================================================

bool system_generate_capacity_plan(system_analyzer_t* analyzer,
                                   component_type_t component,
                                   capacity_plan_t* plan) {
    if (!analyzer || !plan || component >= COMPONENT_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);

    memset(plan, 0, sizeof(capacity_plan_t));
    plan->component = component;

    // Find usage metric for this component
    const char* usage_metric = NULL;
    switch (component) {
        case COMPONENT_CPU: usage_metric = "cpu_usage"; break;
        case COMPONENT_MEMORY: usage_metric = "memory_usage"; break;
        case COMPONENT_DISK: usage_metric = "disk_usage"; break;
        case COMPONENT_GPU: usage_metric = "gpu_usage"; break;
        default: usage_metric = "usage"; break;
    }

    size_t hash = hash_metric_name(component, usage_metric);
    metric_storage_t* m = analyzer->metrics[hash];
    while (m && (m->component != component || strcmp(m->name, usage_metric) != 0)) {
        m = m->next;
    }

    if (m && m->sample_count > 0) {
        // Current and peak usage
        plan->current_usage_percent = m->samples[m->sample_count-1].value;
        plan->avg_usage_percent = m->mean;

        plan->peak_usage_percent = 0;
        for (size_t i = 0; i < m->sample_count; i++) {
            if (m->samples[i].value > plan->peak_usage_percent) {
                plan->peak_usage_percent = m->samples[i].value;
            }
        }

        // Growth rate analysis
        pthread_mutex_unlock(&analyzer->mutex);
        trend_analysis_t trend;
        if (system_analyze_trend(analyzer, component, usage_metric, &trend)) {
            plan->growth_rate_per_day = trend.slope * 86400;

            if (trend.slope > 0) {
                double current = plan->current_usage_percent;
                plan->days_until_80_percent =
                    current < 80 ? (80 - current) / (trend.slope * 86400) : 0;
                plan->days_until_full =
                    current < 100 ? (100 - current) / (trend.slope * 86400) : 0;

                // Recommendation
                if (plan->days_until_80_percent < 30) {
                    plan->recommended_capacity = (size_t)((plan->current_usage_percent + 50) / 100.0 * 100);
                    snprintf(plan->recommendation, sizeof(plan->recommendation),
                             "Increase capacity by %.0f%% within %d days",
                             plan->recommended_capacity - 100.0,
                             (int)plan->days_until_80_percent);
                }
            }
        }
        pthread_mutex_lock(&analyzer->mutex);
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_generate_all_capacity_plans(system_analyzer_t* analyzer,
                                        capacity_plan_t** plans,
                                        size_t* count) {
    if (!analyzer || !plans || !count) return false;

    *count = COMPONENT_COUNT;
    *plans = calloc(*count, sizeof(capacity_plan_t));
    if (!*plans) return false;

    for (int i = 0; i < COMPONENT_COUNT; i++) {
        system_generate_capacity_plan(analyzer, (component_type_t)i, &(*plans)[i]);
    }

    return true;
}

void system_set_capacity_threshold(system_analyzer_t* analyzer,
                                   component_type_t component,
                                   double warning_percent,
                                   double critical_percent) {
    if (!analyzer || component >= COMPONENT_COUNT) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->capacity_warning[component] = warning_percent / 100.0;
    analyzer->capacity_critical[component] = critical_percent / 100.0;
    pthread_mutex_unlock(&analyzer->mutex);
}

char* system_get_capacity_recommendations(system_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    capacity_plan_t* plans;
    size_t count;

    if (!system_generate_all_capacity_plans(analyzer, &plans, &count)) {
        return NULL;
    }

    size_t buffer_size = 4096;
    char* result = malloc(buffer_size);
    if (!result) {
        system_free_capacity_plans(plans);
        return NULL;
    }

    size_t offset = snprintf(result, buffer_size, "Capacity Recommendations:\n\n");

    for (size_t i = 0; i < count; i++) {
        if (plans[i].recommendation[0]) {
            offset += snprintf(result + offset, buffer_size - offset,
                              "%s: %s\n",
                              system_component_name(plans[i].component),
                              plans[i].recommendation);
        }
    }

    system_free_capacity_plans(plans);
    return result;
}

// ============================================================================
// SLO Management
// ============================================================================

bool system_define_slo(system_analyzer_t* analyzer,
                       const slo_definition_t* slo) {
    if (!analyzer || !slo) return false;

    pthread_mutex_lock(&analyzer->mutex);

    if (analyzer->slo_count >= analyzer->slo_capacity) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    analyzer->slos[analyzer->slo_count++] = *slo;

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_get_slo_status(system_analyzer_t* analyzer,
                           const char* slo_name,
                           slo_definition_t* slo) {
    if (!analyzer || !slo_name || !slo) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->slo_count; i++) {
        if (strcmp(analyzer->slos[i].name, slo_name) == 0) {
            *slo = analyzer->slos[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool system_get_all_slo_statuses(system_analyzer_t* analyzer,
                                 slo_definition_t** slos,
                                 size_t* count) {
    if (!analyzer || !slos || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->slo_count;
    if (*count == 0) {
        *slos = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *slos = calloc(*count, sizeof(slo_definition_t));
    if (*slos) {
        memcpy(*slos, analyzer->slos, *count * sizeof(slo_definition_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *slos != NULL;
}

void system_record_slo_metric(system_analyzer_t* analyzer,
                              const char* slo_name,
                              double value,
                              bool is_compliant) {
    if (!analyzer || !slo_name) return;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < analyzer->slo_count; i++) {
        if (strcmp(analyzer->slos[i].name, slo_name) == 0) {
            // Update compliance tracking (simplified)
            if (!is_compliant) {
                analyzer->slos[i].violations_count++;
            }

            // Simple moving average for compliance
            double n = (double)(analyzer->slos[i].violations_count + 1);
            double old_compliance = analyzer->slos[i].current_compliance;
            analyzer->slos[i].current_compliance =
                old_compliance + (is_compliant ? 1.0 : 0.0 - old_compliance) / n;

            // Update remaining budget
            analyzer->slos[i].remaining_budget =
                analyzer->slos[i].error_budget_percent -
                (1.0 - analyzer->slos[i].current_compliance);

            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

bool system_get_slo_compliance_history(system_analyzer_t* analyzer,
                                       const char* slo_name,
                                       double** compliance,
                                       uint64_t** timestamps,
                                       size_t* count) {
    (void)analyzer;
    (void)slo_name;
    (void)compliance;
    (void)timestamps;
    (void)count;
    // Simplified - would need additional storage for full history
    return false;
}

double system_get_error_budget_burn_rate(system_analyzer_t* analyzer,
                                         const char* slo_name) {
    if (!analyzer || !slo_name) return 0;

    pthread_mutex_lock(&analyzer->mutex);

    double burn_rate = 0;
    for (size_t i = 0; i < analyzer->slo_count; i++) {
        if (strcmp(analyzer->slos[i].name, slo_name) == 0) {
            double consumed = analyzer->slos[i].error_budget_percent -
                             analyzer->slos[i].remaining_budget;
            // Assume 30-day window
            burn_rate = consumed / 30.0;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return burn_rate;
}

// ============================================================================
// Incident Management
// ============================================================================

uint64_t system_create_incident(system_analyzer_t* analyzer,
                                component_type_t component,
                                health_status_t severity,
                                const char* title,
                                const char* description) {
    if (!analyzer || !title) return 0;

    pthread_mutex_lock(&analyzer->mutex);

    if (analyzer->incident_count >= analyzer->incident_capacity) {
        pthread_mutex_unlock(&analyzer->mutex);
        return 0;
    }

    uint64_t id = analyzer->next_incident_id++;
    incident_record_t* incident = &analyzer->incidents[analyzer->incident_count++];

    incident->id = id;
    incident->status = INCIDENT_OPEN;
    incident->component = component;
    incident->severity = severity;
    strncpy(incident->title, title, sizeof(incident->title) - 1);
    if (description) {
        strncpy(incident->description, description, sizeof(incident->description) - 1);
    }
    incident->start_time_ns = system_get_timestamp_ns();
    incident->impact_score = (severity >= HEALTH_CRITICAL) ? 80.0 : 50.0;

    // Update component health
    analyzer->components[component].issue_count++;

    pthread_mutex_unlock(&analyzer->mutex);

    return id;
}

bool system_update_incident(system_analyzer_t* analyzer,
                            uint64_t incident_id,
                            incident_status_t status,
                            const char* update_message) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->incident_count; i++) {
        if (analyzer->incidents[i].id == incident_id) {
            analyzer->incidents[i].status = status;
            // Append to description (simplified)
            if (update_message) {
                size_t len = strlen(analyzer->incidents[i].description);
                snprintf(analyzer->incidents[i].description + len,
                        sizeof(analyzer->incidents[i].description) - len,
                        "\n[Update] %s", update_message);
            }
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool system_resolve_incident(system_analyzer_t* analyzer,
                             uint64_t incident_id,
                             const char* root_cause,
                             const char* resolution) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->incident_count; i++) {
        if (analyzer->incidents[i].id == incident_id) {
            analyzer->incidents[i].status = INCIDENT_RESOLVED;
            analyzer->incidents[i].end_time_ns = system_get_timestamp_ns();
            analyzer->incidents[i].duration_seconds =
                (analyzer->incidents[i].end_time_ns -
                 analyzer->incidents[i].start_time_ns) / 1000000000ULL;

            if (root_cause) {
                strncpy(analyzer->incidents[i].root_cause, root_cause,
                        sizeof(analyzer->incidents[i].root_cause) - 1);
            }
            if (resolution) {
                strncpy(analyzer->incidents[i].resolution, resolution,
                        sizeof(analyzer->incidents[i].resolution) - 1);
            }

            // Update component health
            if (analyzer->components[analyzer->incidents[i].component].issue_count > 0) {
                analyzer->components[analyzer->incidents[i].component].issue_count--;
            }

            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool system_get_incident(system_analyzer_t* analyzer,
                         uint64_t incident_id,
                         incident_record_t* incident) {
    if (!analyzer || !incident) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->incident_count; i++) {
        if (analyzer->incidents[i].id == incident_id) {
            *incident = analyzer->incidents[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool system_get_open_incidents(system_analyzer_t* analyzer,
                               incident_record_t** incidents,
                               size_t* count) {
    if (!analyzer || !incidents || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Count open incidents
    size_t open_count = 0;
    for (size_t i = 0; i < analyzer->incident_count; i++) {
        if (analyzer->incidents[i].status != INCIDENT_RESOLVED &&
            analyzer->incidents[i].status != INCIDENT_CLOSED) {
            open_count++;
        }
    }

    *count = open_count;
    if (open_count == 0) {
        *incidents = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *incidents = calloc(open_count, sizeof(incident_record_t));
    if (!*incidents) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < analyzer->incident_count && idx < open_count; i++) {
        if (analyzer->incidents[i].status != INCIDENT_RESOLVED &&
            analyzer->incidents[i].status != INCIDENT_CLOSED) {
            (*incidents)[idx++] = analyzer->incidents[i];
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_get_incident_history(system_analyzer_t* analyzer,
                                 incident_record_t** incidents,
                                 size_t* count) {
    if (!analyzer || !incidents || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->incident_count;
    if (*count == 0) {
        *incidents = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *incidents = calloc(*count, sizeof(incident_record_t));
    if (*incidents) {
        memcpy(*incidents, analyzer->incidents,
               *count * sizeof(incident_record_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *incidents != NULL;
}

bool system_correlate_incident(system_analyzer_t* analyzer,
                               uint64_t incident_id,
                               detected_anomaly_t** related_anomalies,
                               size_t* count) {
    if (!analyzer || !related_anomalies || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    incident_record_t* incident = NULL;
    for (size_t i = 0; i < analyzer->incident_count; i++) {
        if (analyzer->incidents[i].id == incident_id) {
            incident = &analyzer->incidents[i];
            break;
        }
    }

    if (!incident) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    // Find anomalies around incident time
    uint64_t window_ns = 60 * 1000000000ULL;  // 1 minute window
    size_t related_count = 0;

    for (size_t i = 0; i < analyzer->anomaly_count; i++) {
        uint64_t anomaly_time = analyzer->anomaly_history[i].detection_time_ns;
        if (anomaly_time >= incident->start_time_ns - window_ns &&
            anomaly_time <= incident->start_time_ns + window_ns &&
            analyzer->anomaly_history[i].component == incident->component) {
            related_count++;
        }
    }

    *count = related_count;
    if (related_count == 0) {
        *related_anomalies = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *related_anomalies = calloc(related_count, sizeof(detected_anomaly_t));
    if (!*related_anomalies) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < analyzer->anomaly_count && idx < related_count; i++) {
        uint64_t anomaly_time = analyzer->anomaly_history[i].detection_time_ns;
        if (anomaly_time >= incident->start_time_ns - window_ns &&
            anomaly_time <= incident->start_time_ns + window_ns &&
            analyzer->anomaly_history[i].component == incident->component) {
            (*related_anomalies)[idx++] = analyzer->anomaly_history[i];
        }
    }

    incident->related_anomalies = related_count;

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Correlation Analysis
// ============================================================================

bool system_compute_correlations(system_analyzer_t* analyzer,
                                 metric_correlation_t** correlations,
                                 size_t* count) {
    (void)analyzer;
    (void)correlations;
    (void)count;
    // Complex implementation - would compare all metric pairs
    return false;
}

bool system_get_correlation(system_analyzer_t* analyzer,
                            component_type_t comp1,
                            const char* metric1,
                            component_type_t comp2,
                            const char* metric2,
                            metric_correlation_t* correlation) {
    if (!analyzer || !metric1 || !metric2 || !correlation) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Find both metrics
    metric_storage_t* m1 = NULL;
    metric_storage_t* m2 = NULL;

    size_t hash1 = hash_metric_name(comp1, metric1);
    m1 = analyzer->metrics[hash1];
    while (m1 && (m1->component != comp1 || strcmp(m1->name, metric1) != 0)) {
        m1 = m1->next;
    }

    size_t hash2 = hash_metric_name(comp2, metric2);
    m2 = analyzer->metrics[hash2];
    while (m2 && (m2->component != comp2 || strcmp(m2->name, metric2) != 0)) {
        m2 = m2->next;
    }

    if (!m1 || !m2 || m1->sample_count < 10 || m2->sample_count < 10) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    // Compute Pearson correlation
    size_t n = fmin(m1->sample_count, m2->sample_count);
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0, sum_yy = 0;

    for (size_t i = 0; i < n; i++) {
        double x = m1->samples[i].value;
        double y = m2->samples[i].value;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        sum_yy += y * y;
    }

    double dn = (double)n;
    double denom = sqrt((dn * sum_xx - sum_x * sum_x) *
                        (dn * sum_yy - sum_y * sum_y));

    strncpy(correlation->metric1_name, metric1, sizeof(correlation->metric1_name) - 1);
    strncpy(correlation->metric2_name, metric2, sizeof(correlation->metric2_name) - 1);
    correlation->component1 = comp1;
    correlation->component2 = comp2;

    if (fabs(denom) > 1e-10) {
        correlation->correlation = (dn * sum_xy - sum_x * sum_y) / denom;
    } else {
        correlation->correlation = 0;
    }

    correlation->is_significant = fabs(correlation->correlation) > 0.5;

    // Simplified p-value calculation
    double t = correlation->correlation * sqrt((n - 2) /
               (1 - correlation->correlation * correlation->correlation + 1e-10));
    correlation->p_value = fmax(0, 1.0 - fabs(t) / 10.0);  // Simplified

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool system_find_correlated_metrics(system_analyzer_t* analyzer,
                                    component_type_t component,
                                    const char* metric_name,
                                    double min_correlation,
                                    metric_correlation_t** correlations,
                                    size_t* count) {
    (void)analyzer;
    (void)component;
    (void)metric_name;
    (void)min_correlation;
    (void)correlations;
    (void)count;
    // Would iterate through all metrics and compute correlations
    return false;
}

// ============================================================================
// Reporting
// ============================================================================

char* system_generate_health_report(system_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    system_health_t health;
    if (!system_compute_health(analyzer, &health)) {
        return NULL;
    }

    size_t buffer_size = 4096;
    char* report = malloc(buffer_size);
    if (!report) return NULL;

    size_t offset = snprintf(report, buffer_size,
        "=== System Health Report ===\n\n"
        "Overall Score: %.1f\n"
        "Status: %s\n"
        "Uptime: %llu seconds\n"
        "Total Issues: %zu\n"
        "Critical Issues: %zu\n\n"
        "Component Status:\n",
        health.overall_score,
        system_health_status_name(health.overall_status),
        (unsigned long long)health.uptime_seconds,
        health.total_issues,
        health.critical_issues);

    for (int i = 0; i < COMPONENT_COUNT; i++) {
        offset += snprintf(report + offset, buffer_size - offset,
            "  %s: %.1f (%s) - %s\n",
            system_component_name((component_type_t)i),
            health.components[i].score,
            system_health_status_name(health.components[i].status),
            health.components[i].status_message[0] ?
                health.components[i].status_message : "OK");
    }

    return report;
}

char* system_generate_incident_report(system_analyzer_t* analyzer,
                                      uint64_t start_time_ns,
                                      uint64_t end_time_ns) {
    (void)start_time_ns;
    (void)end_time_ns;

    if (!analyzer) return NULL;

    incident_record_t* incidents;
    size_t count;

    if (!system_get_incident_history(analyzer, &incidents, &count)) {
        return NULL;
    }

    size_t buffer_size = 8192;
    char* report = malloc(buffer_size);
    if (!report) {
        system_free_incidents(incidents);
        return NULL;
    }

    size_t offset = snprintf(report, buffer_size,
        "=== Incident Report ===\n\n"
        "Total Incidents: %zu\n\n",
        count);

    for (size_t i = 0; i < count && offset < buffer_size - 256; i++) {
        offset += snprintf(report + offset, buffer_size - offset,
            "Incident #%llu: %s\n"
            "  Component: %s\n"
            "  Status: %s\n"
            "  Severity: %s\n"
            "  Duration: %llu seconds\n\n",
            (unsigned long long)incidents[i].id,
            incidents[i].title,
            system_component_name(incidents[i].component),
            system_incident_status_name(incidents[i].status),
            system_health_status_name(incidents[i].severity),
            (unsigned long long)incidents[i].duration_seconds);
    }

    system_free_incidents(incidents);
    return report;
}

char* system_generate_capacity_report(system_analyzer_t* analyzer) {
    return system_get_capacity_recommendations(analyzer);
}

char* system_generate_slo_report(system_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    slo_definition_t* slos;
    size_t count;

    if (!system_get_all_slo_statuses(analyzer, &slos, &count)) {
        return NULL;
    }

    size_t buffer_size = 4096;
    char* report = malloc(buffer_size);
    if (!report) {
        system_free_slos(slos);
        return NULL;
    }

    size_t offset = snprintf(report, buffer_size,
        "=== SLO Report ===\n\n");

    for (size_t i = 0; i < count && offset < buffer_size - 256; i++) {
        offset += snprintf(report + offset, buffer_size - offset,
            "SLO: %s\n"
            "  Target: %.2f\n"
            "  Current Compliance: %.2f%%\n"
            "  Violations: %llu\n"
            "  Remaining Budget: %.4f%%\n\n",
            slos[i].name,
            slos[i].target_value,
            slos[i].current_compliance * 100.0,
            (unsigned long long)slos[i].violations_count,
            slos[i].remaining_budget * 100.0);
    }

    system_free_slos(slos);
    return report;
}

char* system_export_json(system_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    system_health_t health;
    system_compute_health(analyzer, &health);

    size_t buffer_size = 8192;
    char* json = malloc(buffer_size);
    if (!json) return NULL;

    snprintf(json, buffer_size,
        "{\n"
        "  \"overall_score\": %.2f,\n"
        "  \"overall_status\": \"%s\",\n"
        "  \"uptime_seconds\": %llu,\n"
        "  \"total_issues\": %zu,\n"
        "  \"critical_issues\": %zu,\n"
        "  \"anomaly_count\": %zu,\n"
        "  \"incident_count\": %zu,\n"
        "  \"slo_count\": %zu\n"
        "}",
        health.overall_score,
        system_health_status_name(health.overall_status),
        (unsigned long long)health.uptime_seconds,
        health.total_issues,
        health.critical_issues,
        analyzer->anomaly_count,
        analyzer->incident_count,
        analyzer->slo_count);

    return json;
}

bool system_export_to_file(system_analyzer_t* analyzer,
                           const char* filename) {
    if (!analyzer || !filename) return false;

    char* json = system_export_json(analyzer);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    fputs(json, f);
    fclose(f);
    free(json);

    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* system_health_status_name(health_status_t status) {
    switch (status) {
        case HEALTH_OPTIMAL: return "optimal";
        case HEALTH_GOOD: return "good";
        case HEALTH_DEGRADED: return "degraded";
        case HEALTH_WARNING: return "warning";
        case HEALTH_CRITICAL: return "critical";
        case HEALTH_FAILING: return "failing";
        default: return "unknown";
    }
}

const char* system_component_name(component_type_t component) {
    switch (component) {
        case COMPONENT_CPU: return "cpu";
        case COMPONENT_MEMORY: return "memory";
        case COMPONENT_DISK: return "disk";
        case COMPONENT_NETWORK: return "network";
        case COMPONENT_GPU: return "gpu";
        case COMPONENT_QUANTUM_BACKEND: return "quantum_backend";
        case COMPONENT_MPI: return "mpi";
        case COMPONENT_LIBRARY_CORE: return "library_core";
        default: return "unknown";
    }
}

const char* system_anomaly_type_name(anomaly_type_t type) {
    switch (type) {
        case ANOMALY_SPIKE: return "spike";
        case ANOMALY_DROP: return "drop";
        case ANOMALY_TREND_CHANGE: return "trend_change";
        case ANOMALY_VARIANCE_CHANGE: return "variance_change";
        case ANOMALY_OUTLIER: return "outlier";
        case ANOMALY_PERIODIC: return "periodic";
        case ANOMALY_CORRELATION: return "correlation";
        default: return "unknown";
    }
}

const char* system_trend_direction_name(trend_direction_t direction) {
    switch (direction) {
        case TREND_STABLE: return "stable";
        case TREND_INCREASING: return "increasing";
        case TREND_DECREASING: return "decreasing";
        case TREND_OSCILLATING: return "oscillating";
        case TREND_UNKNOWN: return "unknown";
        default: return "unknown";
    }
}

const char* system_incident_status_name(incident_status_t status) {
    switch (status) {
        case INCIDENT_OPEN: return "open";
        case INCIDENT_ACKNOWLEDGED: return "acknowledged";
        case INCIDENT_INVESTIGATING: return "investigating";
        case INCIDENT_MITIGATING: return "mitigating";
        case INCIDENT_RESOLVED: return "resolved";
        case INCIDENT_CLOSED: return "closed";
        default: return "unknown";
    }
}

const char* system_get_last_error(void) {
    return tls_error[0] ? tls_error : "No error";
}

void system_free_anomalies(detected_anomaly_t* anomalies) {
    free(anomalies);
}

void system_free_trends(trend_analysis_t* trends) {
    free(trends);
}

void system_free_capacity_plans(capacity_plan_t* plans) {
    free(plans);
}

void system_free_slos(slo_definition_t* slos) {
    free(slos);
}

void system_free_incidents(incident_record_t* incidents) {
    free(incidents);
}

void system_free_correlations(metric_correlation_t* correlations) {
    free(correlations);
}

void system_free_health_history(system_health_t* history) {
    free(history);
}
