#ifndef CONTENTION_MANAGER_H
#define CONTENTION_MANAGER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Contention types
typedef enum {
    CONTENTION_RESOURCE,      // Resource contention
    CONTENTION_ACCESS,        // Access contention
    CONTENTION_QUANTUM,       // Quantum resource contention
    CONTENTION_MEMORY,        // Memory contention
    CONTENTION_NETWORK       // Network contention
} contention_type_t;

// Resolution strategies
typedef enum {
    RESOLVE_FIFO,            // First-in-first-out
    RESOLVE_PRIORITY,        // Priority-based
    RESOLVE_ADAPTIVE,        // Adaptive resolution
    RESOLVE_QUANTUM         // Quantum-aware resolution
} resolution_strategy_t;

// Resource states
typedef enum {
    STATE_FREE,              // Resource is free
    STATE_LOCKED,            // Resource is locked
    STATE_CONTENDED,         // Resource is contended
    STATE_DEADLOCKED        // Resource is deadlocked
} resource_state_t;

// Access modes
typedef enum {
    ACCESS_READ,             // Read access
    ACCESS_WRITE,            // Write access
    ACCESS_EXCLUSIVE,        // Exclusive access
    ACCESS_SHARED           // Shared access
} access_mode_t;

// Manager configuration
typedef struct {
    resolution_strategy_t strategy;   // Resolution strategy
    size_t timeout;                   // Resolution timeout
    bool enable_deadlock_detection;   // Enable deadlock detection
    bool enable_starvation_prevention; // Prevent starvation
    bool track_history;               // Track contention history
    size_t max_retries;              // Maximum retry attempts
} manager_config_t;

// Resource descriptor
typedef struct {
    void* resource;                   // Resource pointer
    resource_state_t state;           // Resource state
    size_t waiters;                   // Number of waiters
    access_mode_t mode;               // Current access mode
    void* owner;                      // Current owner
    struct timespec lock_time;        // Lock timestamp
} resource_desc_t;

// Contention record
typedef struct {
    contention_type_t type;           // Contention type
    resource_desc_t* resource;        // Contended resource
    size_t participants;              // Number of participants
    double duration;                  // Contention duration
    struct timespec start_time;       // Start timestamp
    struct timespec end_time;         // End timestamp
} contention_record_t;

// Resolution result
typedef struct {
    bool resolved;                    // Resolution success
    size_t attempts;                  // Resolution attempts
    double resolution_time;           // Resolution time
    char* resolution_method;          // Resolution method
    void* winner;                     // Winning participant
    void* resolution_data;           // Additional data
} resolution_result_t;

// Performance metrics
typedef struct {
    size_t total_contentions;         // Total contentions
    size_t resolved_contentions;      // Resolved contentions
    double average_resolution_time;   // Average resolution time
    size_t deadlocks_detected;        // Deadlocks detected
    size_t starvation_incidents;      // Starvation incidents
    double resource_utilization;      // Resource utilization
} performance_metrics_t;

// Opaque manager handle
typedef struct contention_manager_t contention_manager_t;

// Core functions
contention_manager_t* create_contention_manager(const manager_config_t* config);
void destroy_contention_manager(contention_manager_t* manager);

// Resource management
bool register_resource(contention_manager_t* manager,
                      const resource_desc_t* resource);
bool unregister_resource(contention_manager_t* manager,
                        void* resource);
bool update_resource_state(contention_manager_t* manager,
                         void* resource,
                         resource_state_t state);

// Access control
bool request_access(contention_manager_t* manager,
                   void* resource,
                   access_mode_t mode,
                   void* requester);
bool release_access(contention_manager_t* manager,
                   void* resource,
                   void* owner);
bool validate_access(contention_manager_t* manager,
                    void* resource,
                    void* accessor);

// Contention handling
bool detect_contention(contention_manager_t* manager,
                      void* resource,
                      contention_record_t* record);
bool resolve_contention(contention_manager_t* manager,
                       const contention_record_t* record,
                       resolution_result_t* result);
bool handle_deadlock(contention_manager_t* manager,
                    void* resource,
                    resolution_result_t* result);

// Monitoring functions
bool get_resource_state(const contention_manager_t* manager,
                       void* resource,
                       resource_desc_t* desc);
bool get_contention_history(const contention_manager_t* manager,
                           contention_record_t* history,
                           size_t* num_records);
bool get_performance_metrics(const contention_manager_t* manager,
                           performance_metrics_t* metrics);

// Quantum-specific functions
bool manage_quantum_resources(contention_manager_t* manager,
                            void* quantum_resource,
                            resolution_strategy_t strategy);
bool optimize_quantum_access(contention_manager_t* manager,
                           void* quantum_resource,
                           access_mode_t mode);
bool validate_quantum_state(contention_manager_t* manager,
                          void* quantum_resource);

// Utility functions
bool export_manager_data(const contention_manager_t* manager,
                        const char* filename);
bool import_manager_data(contention_manager_t* manager,
                        const char* filename);
void free_resolution_result(resolution_result_t* result);

#endif // CONTENTION_MANAGER_H
