#ifndef COMMUNICATION_OPTIMIZATION_H
#define COMMUNICATION_OPTIMIZATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Message types for communication
typedef enum {
    MSG_TYPE_DATA,
    MSG_TYPE_GRADIENT,
    MSG_TYPE_CONTROL,
    MSG_TYPE_SYNC,
    MSG_TYPE_CHECKPOINT,
    MESSAGE_AGGREGATED
} MessageType;

// Communication statistics
typedef struct {
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t compression_savings;
    uint64_t bytes_saved;
    uint64_t messages_compressed;
    uint64_t zero_copy_transfers;
    double average_latency;
    double max_latency;
    double min_latency;
} CommunicationStats;

// Forward declaration for opaque type
typedef struct CommunicationManager CommunicationManager;

// Communication manager functions
CommunicationManager* init_communication_manager(void);
void cleanup_communication_manager(CommunicationManager* manager);

// Send/receive functions
int send_message(CommunicationManager* manager, const void* data, size_t size,
                int dest_rank, MessageType type);
int receive_message(CommunicationManager* manager, void* data, size_t* size,
                   int* source_rank, MessageType* type);

// Collective operations
int broadcast_data(CommunicationManager* manager, void* data, size_t size, int root);
int allreduce_data(CommunicationManager* manager, void* send_data, void* recv_data,
                  size_t count, size_t element_size);
int allgather_data(CommunicationManager* manager, void* send_data, size_t send_count,
                  void* recv_data, size_t recv_count);

// Async operations
int async_send(CommunicationManager* manager, const void* data, size_t size,
              int dest_rank, MessageType type, int* request_id);
int async_recv(CommunicationManager* manager, void* data, size_t size,
              int source_rank, MessageType type, int* request_id);
int wait_request(CommunicationManager* manager, int request_id);
int wait_all_requests(CommunicationManager* manager);

// Statistics
CommunicationStats get_communication_stats(const CommunicationManager* manager);
void reset_communication_stats(CommunicationManager* manager);

// Utility functions
int get_rank(const CommunicationManager* manager);
int get_comm_world_size(const CommunicationManager* manager);
void barrier(CommunicationManager* manager);

// Zero-copy support
bool check_zero_copy_support(void);
void* allocate_rdma_buffer(size_t size);
void free_rdma_buffer(void* buffer);

// Message aggregation
int handle_aggregated_message(CommunicationManager* manager, const void* data, size_t size);
int flush_aggregated_message(CommunicationManager* manager);

#ifdef __cplusplus
}
#endif

#endif // COMMUNICATION_OPTIMIZATION_H
