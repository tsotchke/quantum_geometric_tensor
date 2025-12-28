/**
 * @file warning_generator.h
 * @brief Warning and Alert Generation for Quantum Operations
 *
 * Provides warning generation including:
 * - Threshold-based alerting
 * - Anomaly detection warnings
 * - Resource limit warnings
 * - Performance degradation alerts
 * - Error condition notifications
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef WARNING_GENERATOR_H
#define WARNING_GENERATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define WARNING_MAX_MESSAGE_LENGTH 512
#define WARNING_MAX_CONTEXT_LENGTH 256
#define WARNING_MAX_ACTIVE 1024
#define WARNING_MAX_RULES 256

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    WARNING_LEVEL_DEBUG,
    WARNING_LEVEL_INFO,
    WARNING_LEVEL_WARNING,
    WARNING_LEVEL_ERROR,
    WARNING_LEVEL_CRITICAL
} warning_level_t;

typedef enum {
    WARNING_CATEGORY_PERFORMANCE,
    WARNING_CATEGORY_MEMORY,
    WARNING_CATEGORY_ACCURACY,
    WARNING_CATEGORY_RESOURCE,
    WARNING_CATEGORY_SECURITY,
    WARNING_CATEGORY_STABILITY,
    WARNING_CATEGORY_CONFIGURATION,
    WARNING_CATEGORY_CUSTOM
} warning_category_t;

typedef enum {
    WARNING_TRIGGER_THRESHOLD_EXCEEDED,
    WARNING_TRIGGER_THRESHOLD_BELOW,
    WARNING_TRIGGER_RATE_OF_CHANGE,
    WARNING_TRIGGER_ANOMALY_DETECTED,
    WARNING_TRIGGER_PATTERN_MATCH,
    WARNING_TRIGGER_TIMEOUT,
    WARNING_TRIGGER_MANUAL
} warning_trigger_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    uint64_t id;
    warning_level_t level;
    warning_category_t category;
    warning_trigger_t trigger;
    char message[WARNING_MAX_MESSAGE_LENGTH];
    char context[WARNING_MAX_CONTEXT_LENGTH];
    uint64_t timestamp_ns;
    uint64_t count;                   // Times this warning occurred
    bool acknowledged;
    bool suppressed;
} warning_entry_t;

typedef struct {
    char name[128];
    warning_level_t level;
    warning_category_t category;
    warning_trigger_t trigger;
    double threshold;
    char message_template[256];
    bool enabled;
    uint64_t cooldown_ns;             // Min time between same warnings
} warning_rule_t;

typedef struct {
    uint64_t total_warnings;
    uint64_t warnings_by_level[5];
    uint64_t warnings_by_category[8];
    uint64_t suppressed_count;
    uint64_t acknowledged_count;
} warning_stats_t;

typedef struct {
    bool enable_suppression;
    bool enable_aggregation;
    uint64_t default_cooldown_ns;
    size_t max_active_warnings;
    warning_level_t min_log_level;
    void (*callback)(const warning_entry_t* warning, void* user_data);
    void* callback_data;
} warning_generator_config_t;

typedef struct warning_generator warning_generator_t;

// ============================================================================
// API Functions
// ============================================================================

warning_generator_t* warning_generator_create(void);
warning_generator_t* warning_generator_create_with_config(
    const warning_generator_config_t* config);
warning_generator_config_t warning_generator_default_config(void);
void warning_generator_destroy(warning_generator_t* generator);
bool warning_generator_reset(warning_generator_t* generator);

bool warning_add_rule(warning_generator_t* generator,
                      const warning_rule_t* rule);
bool warning_remove_rule(warning_generator_t* generator,
                         const char* name);
bool warning_enable_rule(warning_generator_t* generator,
                         const char* name, bool enabled);

uint64_t warning_emit(warning_generator_t* generator,
                      warning_level_t level,
                      warning_category_t category,
                      const char* message);

uint64_t warning_emit_with_context(warning_generator_t* generator,
                                    warning_level_t level,
                                    warning_category_t category,
                                    const char* message,
                                    const char* context);

bool warning_check_threshold(warning_generator_t* generator,
                              const char* rule_name,
                              double value);

bool warning_acknowledge(warning_generator_t* generator, uint64_t id);
bool warning_suppress(warning_generator_t* generator, uint64_t id);
bool warning_clear(warning_generator_t* generator, uint64_t id);
bool warning_clear_all(warning_generator_t* generator);

bool warning_get_active(warning_generator_t* generator,
                        warning_entry_t** warnings,
                        size_t* count);
bool warning_get_by_level(warning_generator_t* generator,
                          warning_level_t level,
                          warning_entry_t** warnings,
                          size_t* count);
bool warning_get_stats(warning_generator_t* generator,
                       warning_stats_t* stats);

char* warning_generate_report(warning_generator_t* generator);
char* warning_export_json(warning_generator_t* generator);
bool warning_export_to_file(warning_generator_t* generator,
                             const char* filename);

const char* warning_level_name(warning_level_t level);
const char* warning_category_name(warning_category_t category);
const char* warning_trigger_name(warning_trigger_t trigger);
void warning_free_entries(warning_entry_t* entries, size_t count);
const char* warning_get_last_error(warning_generator_t* generator);

#ifdef __cplusplus
}
#endif

#endif // WARNING_GENERATOR_H
