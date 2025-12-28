/**
 * @file response_coordinator.h
 * @brief Alert Response Coordination for Quantum Systems
 *
 * Provides response coordination including:
 * - Automated response actions
 * - Escalation management
 * - Recovery procedures
 * - Response prioritization
 * - Action logging and tracking
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef RESPONSE_COORDINATOR_H
#define RESPONSE_COORDINATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define RESPONSE_MAX_ACTIONS 256
#define RESPONSE_MAX_NAME_LENGTH 128
#define RESPONSE_MAX_DESCRIPTION_LENGTH 512

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    RESPONSE_ACTION_LOG,
    RESPONSE_ACTION_NOTIFY,
    RESPONSE_ACTION_THROTTLE,
    RESPONSE_ACTION_PAUSE,
    RESPONSE_ACTION_RESTART,
    RESPONSE_ACTION_ROLLBACK,
    RESPONSE_ACTION_FAILOVER,
    RESPONSE_ACTION_ESCALATE,
    RESPONSE_ACTION_CUSTOM
} response_action_type_t;

typedef enum {
    RESPONSE_PRIORITY_LOW,
    RESPONSE_PRIORITY_MEDIUM,
    RESPONSE_PRIORITY_HIGH,
    RESPONSE_PRIORITY_CRITICAL,
    RESPONSE_PRIORITY_EMERGENCY
} response_priority_t;

typedef enum {
    RESPONSE_STATUS_PENDING,
    RESPONSE_STATUS_IN_PROGRESS,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_FAILED,
    RESPONSE_STATUS_CANCELLED
} response_status_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    uint64_t id;
    response_action_type_t type;
    response_priority_t priority;
    response_status_t status;
    char name[RESPONSE_MAX_NAME_LENGTH];
    char description[RESPONSE_MAX_DESCRIPTION_LENGTH];
    uint64_t trigger_time_ns;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    bool success;
    char result[256];
} response_action_t;

typedef struct {
    char name[RESPONSE_MAX_NAME_LENGTH];
    response_action_type_t action;
    response_priority_t min_priority;
    bool enabled;
    uint64_t cooldown_ns;
    bool (*handler)(const void* context, void* user_data);
    void* user_data;
} response_rule_t;

typedef struct {
    uint64_t total_responses;
    uint64_t successful_responses;
    uint64_t failed_responses;
    uint64_t pending_responses;
    double avg_response_time_ns;
    uint64_t responses_by_type[9];
} response_stats_t;

typedef struct {
    bool enable_auto_response;
    bool enable_escalation;
    uint64_t escalation_timeout_ns;
    size_t max_concurrent_actions;
    response_priority_t min_auto_priority;
} response_coordinator_config_t;

typedef struct response_coordinator response_coordinator_t;

// ============================================================================
// API Functions
// ============================================================================

response_coordinator_t* response_coordinator_create(void);
response_coordinator_t* response_coordinator_create_with_config(
    const response_coordinator_config_t* config);
response_coordinator_config_t response_coordinator_default_config(void);
void response_coordinator_destroy(response_coordinator_t* coordinator);
bool response_coordinator_reset(response_coordinator_t* coordinator);

bool response_add_rule(response_coordinator_t* coordinator,
                       const response_rule_t* rule);
bool response_remove_rule(response_coordinator_t* coordinator,
                          const char* name);
bool response_enable_rule(response_coordinator_t* coordinator,
                          const char* name, bool enabled);

uint64_t response_trigger_action(response_coordinator_t* coordinator,
                                  response_action_type_t type,
                                  response_priority_t priority,
                                  const char* description);

bool response_execute_pending(response_coordinator_t* coordinator);
bool response_cancel_action(response_coordinator_t* coordinator, uint64_t id);
bool response_escalate(response_coordinator_t* coordinator, uint64_t id);

bool response_get_action_status(response_coordinator_t* coordinator,
                                 uint64_t id,
                                 response_action_t* action);
bool response_get_pending_actions(response_coordinator_t* coordinator,
                                   response_action_t** actions,
                                   size_t* count);
bool response_get_history(response_coordinator_t* coordinator,
                          response_action_t** actions,
                          size_t* count);
bool response_get_stats(response_coordinator_t* coordinator,
                        response_stats_t* stats);

char* response_generate_report(response_coordinator_t* coordinator);
char* response_export_json(response_coordinator_t* coordinator);
bool response_export_to_file(response_coordinator_t* coordinator,
                              const char* filename);

const char* response_action_type_name(response_action_type_t type);
const char* response_priority_name(response_priority_t priority);
const char* response_status_name(response_status_t status);
void response_free_actions(response_action_t* actions, size_t count);
const char* response_get_last_error(response_coordinator_t* coordinator);

#ifdef __cplusplus
}
#endif

#endif // RESPONSE_COORDINATOR_H
