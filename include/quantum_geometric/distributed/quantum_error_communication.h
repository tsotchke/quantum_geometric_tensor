/**
 * @file quantum_error_communication.h
 * @brief Distributed Error Communication for Quantum Computing Systems
 *
 * Provides error propagation and communication across distributed nodes:
 * - Error message broadcasting and gathering
 * - Syndrome data distribution for QEC
 * - Fault notification across nodes
 * - Error recovery coordination
 * - Distributed error logging
 * - Error consensus protocols
 *
 * Part of the QGTL Distributed Computing Framework.
 */

#ifndef QUANTUM_ERROR_COMMUNICATION_H
#define QUANTUM_ERROR_COMMUNICATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define QEC_COMM_MAX_MESSAGE_SIZE 4096
#define QEC_COMM_MAX_NODES 256
#define QEC_COMM_MAX_SYNDROMES 1024
#define QEC_COMM_MAX_ERROR_HISTORY 10000
#define QEC_COMM_MAX_NAME_LENGTH 128
#define QEC_COMM_TAG_BASE 5000

// Priority levels for error propagation
#define QEC_PRIORITY_LOW 0
#define QEC_PRIORITY_NORMAL 1
#define QEC_PRIORITY_HIGH 2
#define QEC_PRIORITY_CRITICAL 3
#define QEC_PRIORITY_EMERGENCY 4

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Error types for quantum computing
 */
typedef enum {
    QEC_ERROR_BIT_FLIP,               // Pauli X error
    QEC_ERROR_PHASE_FLIP,             // Pauli Z error
    QEC_ERROR_BIT_PHASE_FLIP,         // Pauli Y error
    QEC_ERROR_DEPOLARIZING,           // Depolarizing noise
    QEC_ERROR_AMPLITUDE_DAMPING,      // T1 decay
    QEC_ERROR_PHASE_DAMPING,          // T2 decay
    QEC_ERROR_MEASUREMENT,            // Measurement error
    QEC_ERROR_LEAKAGE,                // State leakage
    QEC_ERROR_CROSSTALK,              // Qubit crosstalk
    QEC_ERROR_COSMIC_RAY,             // Cosmic ray event
    QEC_ERROR_CLASSICAL,              // Classical system error
    QEC_ERROR_COMMUNICATION,          // Communication failure
    QEC_ERROR_UNKNOWN                 // Unknown error type
} qec_error_type_t;

/**
 * Error severity levels
 */
typedef enum {
    QEC_SEVERITY_DEBUG,               // Debug information
    QEC_SEVERITY_INFO,                // Informational
    QEC_SEVERITY_WARNING,             // Warning (correctable)
    QEC_SEVERITY_ERROR,               // Error (may be correctable)
    QEC_SEVERITY_CRITICAL,            // Critical (uncorrectable)
    QEC_SEVERITY_FATAL                // Fatal (system failure)
} qec_severity_t;

/**
 * Communication patterns
 */
typedef enum {
    QEC_COMM_BROADCAST,               // One-to-all
    QEC_COMM_GATHER,                  // All-to-one
    QEC_COMM_SCATTER,                 // One-to-all (partitioned)
    QEC_COMM_ALLGATHER,               // All-to-all gather
    QEC_COMM_ALLREDUCE,               // Reduce across all
    QEC_COMM_NEIGHBOR,                // Nearest neighbor
    QEC_COMM_RING,                    // Ring pattern
    QEC_COMM_TREE,                    // Tree reduction
    QEC_COMM_POINT_TO_POINT           // Direct p2p
} qec_comm_pattern_t;

/**
 * Syndrome aggregation methods
 */
typedef enum {
    QEC_AGGREGATE_NONE,               // No aggregation
    QEC_AGGREGATE_OR,                 // Bitwise OR
    QEC_AGGREGATE_AND,                // Bitwise AND
    QEC_AGGREGATE_XOR,                // Bitwise XOR
    QEC_AGGREGATE_MAJORITY,           // Majority vote
    QEC_AGGREGATE_WEIGHTED,           // Weighted combination
    QEC_AGGREGATE_CUSTOM              // Custom aggregation
} qec_aggregate_method_t;

/**
 * Recovery coordination modes
 */
typedef enum {
    QEC_RECOVERY_LOCAL,               // Each node recovers independently
    QEC_RECOVERY_COORDINATED,         // Coordinated global recovery
    QEC_RECOVERY_HIERARCHICAL,        // Hierarchical recovery
    QEC_RECOVERY_CONSENSUS            // Consensus-based recovery
} qec_recovery_mode_t;

/**
 * Message delivery guarantees
 */
typedef enum {
    QEC_DELIVERY_BEST_EFFORT,         // May be lost
    QEC_DELIVERY_AT_LEAST_ONCE,       // May duplicate
    QEC_DELIVERY_EXACTLY_ONCE,        // Guaranteed single delivery
    QEC_DELIVERY_ORDERED              // Ordered delivery
} qec_delivery_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Error syndrome data
 */
typedef struct {
    uint64_t syndrome_bits;           // Syndrome measurement results
    size_t num_bits;                  // Number of syndrome bits
    uint64_t timestamp_ns;            // When syndrome was measured
    int source_node;                  // Node that measured syndrome
    size_t qubit_start;               // First qubit index
    size_t qubit_count;               // Number of qubits covered
    double confidence;                // Measurement confidence (0-1)
} qec_syndrome_t;

/**
 * Error event description
 */
typedef struct {
    qec_error_type_t type;            // Error type
    qec_severity_t severity;          // Severity level
    int source_node;                  // Originating node
    int* affected_nodes;              // Nodes affected
    size_t num_affected;              // Number of affected nodes
    size_t* qubit_indices;            // Affected qubit indices
    size_t num_qubits;                // Number of affected qubits
    uint64_t timestamp_ns;            // When error occurred
    char message[QEC_COMM_MAX_MESSAGE_SIZE];  // Error message
    uint64_t event_id;                // Unique event ID
    double error_rate;                // Estimated error rate
} qec_error_event_t;

/**
 * Recovery action descriptor
 */
typedef struct {
    uint64_t event_id;                // Associated error event
    int target_node;                  // Node to perform recovery
    size_t* correction_qubits;        // Qubits to correct
    size_t num_corrections;           // Number of corrections
    uint8_t* correction_ops;          // Correction operations (Pauli)
    bool requires_barrier;            // Needs synchronization
    int priority;                     // Action priority
    uint64_t deadline_ns;             // Deadline for action
} qec_recovery_action_t;

/**
 * Error statistics
 */
typedef struct {
    uint64_t total_errors;            // Total errors detected
    uint64_t errors_by_type[13];      // By error type
    uint64_t errors_by_severity[6];   // By severity
    uint64_t messages_sent;           // Error messages sent
    uint64_t messages_received;       // Error messages received
    uint64_t syndromes_processed;     // Syndromes processed
    uint64_t recoveries_attempted;    // Recovery attempts
    uint64_t recoveries_successful;   // Successful recoveries
    double avg_detection_latency_ns;  // Average detection latency
    double avg_recovery_latency_ns;   // Average recovery latency
    double error_rate_per_qubit;      // Error rate per qubit
    uint64_t bytes_communicated;      // Total bytes for errors
} qec_comm_stats_t;

/**
 * Node information for error communication
 */
typedef struct {
    int node_id;                      // Node identifier
    int rank;                         // MPI rank
    char name[QEC_COMM_MAX_NAME_LENGTH];  // Node name
    size_t qubit_range_start;         // First qubit on node
    size_t qubit_range_end;           // Last qubit on node
    int* neighbors;                   // Neighbor node IDs
    size_t num_neighbors;             // Number of neighbors
    bool is_active;                   // Node active status
} qec_node_info_t;

/**
 * Communication configuration
 */
typedef struct {
    qec_comm_pattern_t default_pattern;   // Default comm pattern
    qec_aggregate_method_t syndrome_agg;  // Syndrome aggregation
    qec_recovery_mode_t recovery_mode;    // Recovery coordination
    qec_delivery_t delivery_guarantee;    // Message delivery
    bool enable_compression;              // Compress messages
    bool enable_batching;                 // Batch messages
    size_t batch_size;                    // Messages per batch
    double batch_timeout_ms;              // Batch timeout
    bool enable_encryption;               // Encrypt messages
    int max_retries;                      // Max retry attempts
    double retry_timeout_ms;              // Retry timeout
    size_t history_size;                  // Error history size
} qec_comm_config_t;

/**
 * Opaque error communication manager handle
 */
typedef struct qec_comm_manager qec_comm_manager_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

/**
 * Create error communication manager
 */
qec_comm_manager_t* qec_comm_create(void);

/**
 * Create with configuration
 */
qec_comm_manager_t* qec_comm_create_with_config(
    const qec_comm_config_t* config);

/**
 * Get default configuration
 */
qec_comm_config_t qec_comm_default_config(void);

/**
 * Destroy manager
 */
void qec_comm_destroy(qec_comm_manager_t* manager);

/**
 * Initialize communication (call after MPI init)
 */
bool qec_comm_init(qec_comm_manager_t* manager);

/**
 * Finalize communication (call before MPI finalize)
 */
bool qec_comm_finalize(qec_comm_manager_t* manager);

/**
 * Reset manager state
 */
bool qec_comm_reset(qec_comm_manager_t* manager);

// ============================================================================
// Node Registration
// ============================================================================

/**
 * Register this node
 */
bool qec_comm_register_node(qec_comm_manager_t* manager,
                            const qec_node_info_t* info);

/**
 * Get node information
 */
bool qec_comm_get_node_info(qec_comm_manager_t* manager,
                            int node_id,
                            qec_node_info_t* info);

/**
 * Get all node information
 */
bool qec_comm_get_all_nodes(qec_comm_manager_t* manager,
                            qec_node_info_t** nodes,
                            size_t* count);

/**
 * Set node neighbors for local error propagation
 */
bool qec_comm_set_neighbors(qec_comm_manager_t* manager,
                            const int* neighbors,
                            size_t count);

// ============================================================================
// Syndrome Communication
// ============================================================================

/**
 * Broadcast syndrome to all nodes
 */
bool qec_comm_broadcast_syndrome(qec_comm_manager_t* manager,
                                 const qec_syndrome_t* syndrome);

/**
 * Send syndrome to specific node
 */
bool qec_comm_send_syndrome(qec_comm_manager_t* manager,
                            const qec_syndrome_t* syndrome,
                            int target_node);

/**
 * Receive syndrome (blocking)
 */
bool qec_comm_receive_syndrome(qec_comm_manager_t* manager,
                               qec_syndrome_t* syndrome,
                               int source_node);

/**
 * Receive syndrome (non-blocking)
 */
bool qec_comm_receive_syndrome_async(qec_comm_manager_t* manager,
                                     qec_syndrome_t* syndrome,
                                     int source_node,
                                     int* request);

/**
 * Gather syndromes from all nodes to root
 */
bool qec_comm_gather_syndromes(qec_comm_manager_t* manager,
                               const qec_syndrome_t* local,
                               qec_syndrome_t** all_syndromes,
                               size_t* count);

/**
 * All-gather syndromes (all nodes get all)
 */
bool qec_comm_allgather_syndromes(qec_comm_manager_t* manager,
                                  const qec_syndrome_t* local,
                                  qec_syndrome_t** all_syndromes,
                                  size_t* count);

/**
 * Aggregate syndromes using configured method
 */
bool qec_comm_aggregate_syndromes(qec_comm_manager_t* manager,
                                  const qec_syndrome_t* syndromes,
                                  size_t count,
                                  qec_syndrome_t* result);

// ============================================================================
// Error Event Communication
// ============================================================================

/**
 * Report local error event
 */
bool qec_comm_report_error(qec_comm_manager_t* manager,
                           const qec_error_event_t* event);

/**
 * Broadcast error to all nodes
 */
bool qec_comm_broadcast_error(qec_comm_manager_t* manager,
                              const qec_error_event_t* event);

/**
 * Send error to specific nodes
 */
bool qec_comm_send_error(qec_comm_manager_t* manager,
                         const qec_error_event_t* event,
                         const int* target_nodes,
                         size_t num_targets);

/**
 * Receive error event (blocking)
 */
bool qec_comm_receive_error(qec_comm_manager_t* manager,
                            qec_error_event_t* event);

/**
 * Receive error event (non-blocking)
 */
bool qec_comm_receive_error_async(qec_comm_manager_t* manager,
                                  qec_error_event_t* event,
                                  int* request);

/**
 * Check for pending error notifications
 */
bool qec_comm_poll_errors(qec_comm_manager_t* manager,
                          bool* has_pending);

/**
 * Get all pending error events
 */
bool qec_comm_get_pending_errors(qec_comm_manager_t* manager,
                                 qec_error_event_t** events,
                                 size_t* count);

// ============================================================================
// Recovery Coordination
// ============================================================================

/**
 * Coordinate recovery action
 */
bool qec_comm_coordinate_recovery(qec_comm_manager_t* manager,
                                  const qec_recovery_action_t* action);

/**
 * Broadcast recovery action to all
 */
bool qec_comm_broadcast_recovery(qec_comm_manager_t* manager,
                                 const qec_recovery_action_t* action);

/**
 * Request consensus on recovery
 */
bool qec_comm_recovery_consensus(qec_comm_manager_t* manager,
                                 const qec_recovery_action_t* proposed,
                                 qec_recovery_action_t* agreed);

/**
 * Barrier for recovery synchronization
 */
bool qec_comm_recovery_barrier(qec_comm_manager_t* manager);

/**
 * Report recovery completion
 */
bool qec_comm_report_recovery_complete(qec_comm_manager_t* manager,
                                       uint64_t event_id,
                                       bool success);

/**
 * Check global recovery status
 */
bool qec_comm_check_recovery_status(qec_comm_manager_t* manager,
                                    uint64_t event_id,
                                    bool* all_complete);

// ============================================================================
// Error Handlers and Callbacks
// ============================================================================

/**
 * Error notification callback type
 */
typedef void (*qec_error_callback_t)(const qec_error_event_t* event,
                                     void* user_data);

/**
 * Syndrome callback type
 */
typedef void (*qec_syndrome_callback_t)(const qec_syndrome_t* syndrome,
                                        void* user_data);

/**
 * Recovery callback type
 */
typedef void (*qec_recovery_callback_t)(const qec_recovery_action_t* action,
                                        void* user_data);

/**
 * Register error notification callback
 */
bool qec_comm_register_error_callback(qec_comm_manager_t* manager,
                                      qec_error_callback_t callback,
                                      void* user_data,
                                      qec_severity_t min_severity);

/**
 * Register syndrome callback
 */
bool qec_comm_register_syndrome_callback(qec_comm_manager_t* manager,
                                         qec_syndrome_callback_t callback,
                                         void* user_data);

/**
 * Register recovery callback
 */
bool qec_comm_register_recovery_callback(qec_comm_manager_t* manager,
                                         qec_recovery_callback_t callback,
                                         void* user_data);

/**
 * Unregister all callbacks
 */
void qec_comm_clear_callbacks(qec_comm_manager_t* manager);

// ============================================================================
// Error History and Logging
// ============================================================================

/**
 * Enable error logging
 */
bool qec_comm_enable_logging(qec_comm_manager_t* manager, bool enable);

/**
 * Get error history
 */
bool qec_comm_get_error_history(qec_comm_manager_t* manager,
                                qec_error_event_t** events,
                                size_t* count);

/**
 * Get errors by type
 */
bool qec_comm_get_errors_by_type(qec_comm_manager_t* manager,
                                 qec_error_type_t type,
                                 qec_error_event_t** events,
                                 size_t* count);

/**
 * Get errors by severity
 */
bool qec_comm_get_errors_by_severity(qec_comm_manager_t* manager,
                                     qec_severity_t severity,
                                     qec_error_event_t** events,
                                     size_t* count);

/**
 * Get errors in time range
 */
bool qec_comm_get_errors_in_range(qec_comm_manager_t* manager,
                                  uint64_t start_ns,
                                  uint64_t end_ns,
                                  qec_error_event_t** events,
                                  size_t* count);

/**
 * Clear error history
 */
void qec_comm_clear_history(qec_comm_manager_t* manager);

/**
 * Export error log to file
 */
bool qec_comm_export_log(qec_comm_manager_t* manager,
                         const char* filename);

// ============================================================================
// Statistics and Monitoring
// ============================================================================

/**
 * Get communication statistics
 */
bool qec_comm_get_stats(qec_comm_manager_t* manager,
                        qec_comm_stats_t* stats);

/**
 * Reset statistics
 */
void qec_comm_reset_stats(qec_comm_manager_t* manager);

/**
 * Get error rate for specific qubit
 */
double qec_comm_get_qubit_error_rate(qec_comm_manager_t* manager,
                                     size_t qubit_index);

/**
 * Get error rate for node
 */
double qec_comm_get_node_error_rate(qec_comm_manager_t* manager,
                                    int node_id);

/**
 * Get communication latency statistics
 */
bool qec_comm_get_latency_stats(qec_comm_manager_t* manager,
                                double* min_ns,
                                double* max_ns,
                                double* avg_ns);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate error communication report
 */
char* qec_comm_generate_report(qec_comm_manager_t* manager);

/**
 * Export to JSON
 */
char* qec_comm_export_json(qec_comm_manager_t* manager);

/**
 * Export to file
 */
bool qec_comm_export_to_file(qec_comm_manager_t* manager,
                             const char* filename);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get error type name
 */
const char* qec_error_type_name(qec_error_type_t type);

/**
 * Get severity name
 */
const char* qec_severity_name(qec_severity_t severity);

/**
 * Get communication pattern name
 */
const char* qec_comm_pattern_name(qec_comm_pattern_t pattern);

/**
 * Get aggregation method name
 */
const char* qec_aggregate_method_name(qec_aggregate_method_t method);

/**
 * Get recovery mode name
 */
const char* qec_recovery_mode_name(qec_recovery_mode_t mode);

/**
 * Get delivery guarantee name
 */
const char* qec_delivery_name(qec_delivery_t delivery);

/**
 * Free error event array
 */
void qec_comm_free_events(qec_error_event_t* events, size_t count);

/**
 * Free syndrome array
 */
void qec_comm_free_syndromes(qec_syndrome_t* syndromes, size_t count);

/**
 * Free node info array
 */
void qec_comm_free_nodes(qec_node_info_t* nodes, size_t count);

/**
 * Free allocated string
 */
void qec_comm_free_string(char* str);

/**
 * Get last error message
 */
const char* qec_comm_get_last_error(qec_comm_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ERROR_COMMUNICATION_H
