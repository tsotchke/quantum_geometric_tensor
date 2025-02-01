#include "quantum_geometric/hardware/quantum_error_constants.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/distributed/quantum_distributed_operations.h"
#include <string.h>
#include <time.h>

// Internal state for distributed error tracking
static struct {
    bool initialized;
    char node_id[32];
    size_t num_nodes;
    DistributedConfig* config;
    pthread_mutex_t mutex;
} distributed_state = {0};

// Initialize distributed error tracking
int init_distributed_error_tracking(const char* node_id, const DistributedConfig* config) {
    if (!node_id || !config) return -1;
    
    pthread_mutex_lock(&distributed_state.mutex);
    
    strncpy(distributed_state.node_id, node_id, sizeof(distributed_state.node_id) - 1);
    distributed_state.node_id[sizeof(distributed_state.node_id) - 1] = '\0';
    
    distributed_state.config = malloc(sizeof(DistributedConfig));
    if (distributed_state.config) {
        memcpy(distributed_state.config, config, sizeof(DistributedConfig));
    }
    
    distributed_state.initialized = true;
    
    pthread_mutex_unlock(&distributed_state.mutex);
    return 0;
}

// Broadcast error statistics to all nodes
int broadcast_error_stats(const ErrorStatsMessage* msg) {
    if (!msg || !distributed_state.initialized) return -1;
    
    // Create message packet
    DistributedMessage packet = {
        .type = DISTRIBUTED_MSG_ERROR_UPDATE,
        .source_node = distributed_state.node_id,
        .timestamp = time(NULL),
        .data_size = sizeof(ErrorStatsMessage),
        .data = (void*)msg
    };
    
    // Broadcast to all nodes using distributed operations
    return broadcast_to_nodes(&packet);
}

// Handle incoming error statistics from other nodes
static void handle_error_update(const ErrorStatsMessage* msg) {
    if (!msg) return;
    
    pthread_mutex_lock(&distributed_state.mutex);
    
    // Update local error tracking based on remote data
    ErrorTrackingStats* local_stats = get_error_stats(NULL, NULL);
    if (local_stats) {
        // Weighted average of error metrics
        double weight = 1.0 / (distributed_state.num_nodes + 1);
        local_stats->total_error = 
            (local_stats->total_error + msg->total_error * weight) / (1 + weight);
        local_stats->error_variance = 
            (local_stats->error_variance + msg->error_variance * weight) / (1 + weight);
        local_stats->confidence_level = 
            (local_stats->confidence_level + msg->confidence_level * weight) / (1 + weight);
        
        // Update error history if space available
        if (local_stats->history_size < MAX_ERROR_HISTORY) {
            local_stats->error_history[local_stats->history_size++] = msg->latest_error;
        }
    }
    
    pthread_mutex_unlock(&distributed_state.mutex);
}

// Message handler for distributed operations
static void error_message_handler(const DistributedMessage* msg) {
    if (!msg || msg->type != DISTRIBUTED_MSG_ERROR_UPDATE) return;
    
    const ErrorStatsMessage* error_msg = (const ErrorStatsMessage*)msg->data;
    handle_error_update(error_msg);
}

// Register error message handler
int register_error_handlers(void) {
    if (!distributed_state.initialized) return -1;
    
    return register_message_handler(DISTRIBUTED_MSG_ERROR_UPDATE, 
                                  error_message_handler);
}

// Clean up distributed error tracking
void cleanup_distributed_error_tracking(void) {
    pthread_mutex_lock(&distributed_state.mutex);
    
    if (distributed_state.config) {
        free(distributed_state.config);
        distributed_state.config = NULL;
    }
    
    distributed_state.initialized = false;
    
    pthread_mutex_unlock(&distributed_state.mutex);
    pthread_mutex_destroy(&distributed_state.mutex);
}

// Synchronize error statistics across nodes
int sync_error_stats(void) {
    if (!distributed_state.initialized) return -1;
    
    // Get local error statistics
    ErrorTrackingStats* local_stats = get_error_stats(NULL, NULL);
    if (!local_stats) return -1;
    
    // Create sync message
    ErrorStatsMessage msg = {
        .type = ERROR_MSG_SYNC,
        .total_error = local_stats->total_error,
        .error_variance = local_stats->error_variance,
        .confidence_level = local_stats->confidence_level,
        .error_count = local_stats->error_count,
        .latest_error = local_stats->error_history[local_stats->history_size - 1],
        .timestamp = time(NULL)
    };
    strncpy(msg.backend_id, distributed_state.node_id, sizeof(msg.backend_id) - 1);
    
    // Broadcast sync message
    return broadcast_error_stats(&msg);
}

// Check if node should be primary for error tracking
bool is_primary_error_node(void) {
    if (!distributed_state.initialized) return false;
    
    // Simple strategy: node with lowest ID is primary
    return strcmp(distributed_state.node_id, 
                 distributed_state.config->primary_node_id) == 0;
}

// Get aggregated error statistics from all nodes
ErrorTrackingStats* get_aggregated_error_stats(void) {
    if (!distributed_state.initialized) return NULL;
    
    // Force sync to get latest data
    if (sync_error_stats() != 0) return NULL;
    
    // Wait for responses (with timeout)
    struct timespec timeout = {
        .tv_sec = time(NULL) + distributed_state.config->sync_timeout,
        .tv_nsec = 0
    };
    
    if (wait_for_responses(DISTRIBUTED_MSG_ERROR_UPDATE, &timeout) != 0) {
        return NULL;
    }
    
    // Return local stats which now include aggregated data
    return get_error_stats(NULL, NULL);
}
