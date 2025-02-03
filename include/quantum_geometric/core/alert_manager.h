#ifndef ALERT_MANAGER_H
#define ALERT_MANAGER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Alert types
typedef enum {
    ALERT_ERROR,           // Error alert
    ALERT_WARNING,         // Warning alert
    ALERT_INFO,            // Information alert
    ALERT_DEBUG,           // Debug alert
    ALERT_QUANTUM         // Quantum-specific alert
} alert_type_t;

// Alert priorities
typedef enum {
    PRIORITY_CRITICAL,     // Critical priority
    PRIORITY_HIGH,         // High priority
    PRIORITY_MEDIUM,       // Medium priority
    PRIORITY_LOW,          // Low priority
    PRIORITY_NEGLIGIBLE   // Negligible priority
} alert_priority_t;

// Notification methods
typedef enum {
    NOTIFY_CONSOLE,        // Console notification
    NOTIFY_LOG,            // Log file notification
    NOTIFY_EMAIL,          // Email notification
    NOTIFY_WEBHOOK,        // Webhook notification
    NOTIFY_CALLBACK       // Callback notification
} notification_method_t;

// Alert states
typedef enum {
    STATE_NEW,             // New alert
    STATE_ACKNOWLEDGED,    // Acknowledged alert
    STATE_IN_PROGRESS,     // In progress
    STATE_RESOLVED,        // Resolved alert
    STATE_CLOSED          // Closed alert
} alert_state_t;

// Manager configuration
typedef struct {
    bool enable_logging;           // Enable logging
    bool enable_notifications;     // Enable notifications
    bool track_history;            // Track alert history
    size_t history_size;          // History size
    char* log_path;               // Log file path
    void* user_data;              // User data for callbacks
} manager_config_t;

// Alert definition
typedef struct {
    alert_type_t type;            // Alert type
    alert_priority_t priority;     // Alert priority
    alert_state_t state;          // Alert state
    char* message;                // Alert message
    char* source;                 // Alert source
    struct timespec timestamp;    // Alert timestamp
    void* alert_data;            // Additional data
} alert_def_t;

// Notification configuration
typedef struct {
    notification_method_t method;  // Notification method
    char* recipient;               // Notification recipient
    char* template_t;                // Message template
    bool batch_notifications;      // Batch notifications
    size_t batch_size;            // Batch size
    void* config_data;           // Additional config
} notification_config_t;

// Alert history
typedef struct {
    alert_def_t* alerts;          // Alert array
    size_t num_alerts;            // Number of alerts
    size_t capacity;              // History capacity
    struct timespec start_time;   // History start time
    struct timespec end_time;     // History end time
    void* history_data;          // Additional data
} alert_history_t;

// Opaque manager handle
typedef struct alert_manager_t alert_manager_t;

// Core functions
alert_manager_t* create_alert_manager(const manager_config_t* config);
void destroy_alert_manager(alert_manager_t* manager);

// Alert management
bool create_alert(alert_manager_t* manager,
                 const alert_def_t* alert);
bool update_alert(alert_manager_t* manager,
                 const char* alert_id,
                 const alert_def_t* alert);
bool delete_alert(alert_manager_t* manager,
                 const char* alert_id);

// Alert handling
bool acknowledge_alert(alert_manager_t* manager,
                      const char* alert_id,
                      const char* acknowledger);
bool resolve_alert(alert_manager_t* manager,
                  const char* alert_id,
                  const char* resolution);
bool close_alert(alert_manager_t* manager,
                const char* alert_id,
                const char* reason);

// Notification management
bool configure_notification(alert_manager_t* manager,
                          notification_method_t method,
                          const notification_config_t* config);
bool send_notification(alert_manager_t* manager,
                      const alert_def_t* alert,
                      notification_method_t method);
bool batch_notifications(alert_manager_t* manager,
                        const alert_def_t* alerts,
                        size_t num_alerts);

// Query functions
bool get_alert(const alert_manager_t* manager,
               const char* alert_id,
               alert_def_t* alert);
bool get_active_alerts(const alert_manager_t* manager,
                      alert_def_t* alerts,
                      size_t* num_alerts);
bool get_alert_history(const alert_manager_t* manager,
                      alert_history_t* history);

// Filter functions
bool filter_alerts_by_type(const alert_manager_t* manager,
                          alert_type_t type,
                          alert_def_t* alerts,
                          size_t* num_alerts);
bool filter_alerts_by_priority(const alert_manager_t* manager,
                             alert_priority_t priority,
                             alert_def_t* alerts,
                             size_t* num_alerts);
bool filter_alerts_by_state(const alert_manager_t* manager,
                           alert_state_t state,
                           alert_def_t* alerts,
                           size_t* num_alerts);

// Quantum-specific functions
bool handle_quantum_alert(alert_manager_t* manager,
                         const alert_def_t* alert);
bool configure_quantum_notifications(alert_manager_t* manager,
                                   const notification_config_t* config);
bool validate_quantum_alert(alert_manager_t* manager,
                          const alert_def_t* alert);

// Utility functions
bool export_alert_data(const alert_manager_t* manager,
                      const char* filename);
bool import_alert_data(alert_manager_t* manager,
                      const char* filename);
void free_alert_history(alert_history_t* history);

#endif // ALERT_MANAGER_H
