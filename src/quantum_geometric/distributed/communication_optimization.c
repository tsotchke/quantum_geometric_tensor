#include "quantum_geometric/distributed/communication_optimization.h"
#include <lz4.h>

#ifndef NO_MPI
#include <mpi.h>
#endif

// Communication parameters
#define MAX_BUFFER_SIZE (64 * 1024 * 1024)  // 64MB
#define COMPRESSION_THRESHOLD (16 * 1024)    // 16KB
#define MAX_PENDING_REQUESTS 32
#define AGGREGATION_TIMEOUT 100  // ms
#define MAX_AGGREGATION_SIZE (1024 * 1024)  // 1MB

#ifndef NO_MPI

// Full MPI implementation structures
typedef struct {
    size_t original_size;
    size_t compressed_size;
    bool is_compressed;
    MessageType type;
    int source_rank;
    int dest_rank;
    uint64_t sequence_number;
} MessageHeader;

typedef struct {
    void* data;
    size_t size;
    double timeout;
    int dest_rank;
    MessageType type;
} AggregatedMessage;

typedef struct {
    void* send_buffer;
    void* recv_buffer;
    void* compression_buffer;
    size_t buffer_size;
    MPI_Request pending_requests[MAX_PENDING_REQUESTS];
    size_t num_pending;
    pthread_mutex_t mutex;
    pthread_t aggregation_thread;
    AggregatedMessage* aggregation_buffers;
    size_t num_aggregation_buffers;
    bool running;
    uint64_t next_sequence;
    bool use_zero_copy;
    CommunicationStats stats;
} MPICommunicationManager;

#else

// Stub implementation structures
typedef struct {
    void* send_buffer;
    void* recv_buffer;
    void* compression_buffer;
    size_t buffer_size;
    CommunicationStats stats;
    bool initialized;
} StubCommunicationManager;

#endif

// Common type definition
typedef struct {
#ifndef NO_MPI
    MPICommunicationManager mpi;
#else
    StubCommunicationManager stub;
#endif
} CommunicationManager;

// Initialize communication manager
CommunicationManager* init_communication_manager(void) {
#ifndef NO_MPI
    CommunicationManager* manager = malloc(sizeof(CommunicationManager));
    if (!manager) return NULL;
    
    manager->buffer_size = MAX_BUFFER_SIZE;
    manager->num_pending = 0;
    manager->running = true;
    manager->next_sequence = 0;
    
    // Check for zero-copy support
    manager->use_zero_copy = check_zero_copy_support();
    
    // Allocate buffers
    if (manager->use_zero_copy) {
        // Use RDMA-capable memory
        manager->send_buffer = allocate_rdma_buffer(MAX_BUFFER_SIZE);
        manager->recv_buffer = allocate_rdma_buffer(MAX_BUFFER_SIZE);
    } else {
        manager->send_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
        manager->recv_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    }
    
    manager->compression_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    
    if (!manager->send_buffer || !manager->recv_buffer ||
        !manager->compression_buffer) {
        cleanup_communication_manager(manager);
        return NULL;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&manager->mutex, NULL) != 0) {
        cleanup_communication_manager(manager);
        return NULL;
    }
    
    // Initialize aggregation buffers
    manager->aggregation_buffers = malloc(
        MAX_PENDING_REQUESTS * sizeof(AggregatedMessage));
    if (!manager->aggregation_buffers) {
        cleanup_communication_manager(manager);
        return NULL;
    }
    manager->num_aggregation_buffers = 0;
    
    // Start aggregation thread
    if (pthread_create(&manager->aggregation_thread,
                      NULL,
                      handle_aggregation,
                      manager) != 0) {
        cleanup_communication_manager(manager);
        return NULL;
    }
    
    // Initialize stats
    memset(&manager->stats, 0, sizeof(CommunicationStats));
    
    return manager;
#else
    // Stub implementation
    CommunicationManager* manager = malloc(sizeof(CommunicationManager));
    if (!manager) return NULL;
    
    manager->stub.send_buffer = NULL;
    manager->stub.recv_buffer = NULL;
    manager->stub.compression_buffer = NULL;
    manager->stub.buffer_size = 0;
    manager->stub.initialized = false;
    memset(&manager->stub.stats, 0, sizeof(CommunicationStats));
    
    return manager;
#endif
}

// Send message with optimization
int send_optimized(CommunicationManager* manager,
                  const void* data,
                  size_t size,
                  int dest,
                  MessageType type) {
#ifndef NO_MPI
    if (!manager || !data) return -1;
    
    pthread_mutex_lock(&manager->mutex);
    
    // Try message aggregation for small messages
    if (size < MAX_AGGREGATION_SIZE &&
        should_aggregate(manager, size, dest, type)) {
        int status = aggregate_message(manager, data, size, dest, type);
        pthread_mutex_unlock(&manager->mutex);
        return status;
    }
    
    // Prepare header
    MessageHeader header = {
        .original_size = size,
        .compressed_size = 0,
        .is_compressed = false,
        .type = type,
        .source_rank = manager->rank,
        .dest_rank = dest,
        .sequence_number = manager->next_sequence++
    };
    
    // Try compression if worth it
    if (size > COMPRESSION_THRESHOLD) {
        int compressed_size = LZ4_compress_default(
            data,
            manager->compression_buffer,
            size,
            manager->buffer_size);
        
        if (compressed_size > 0 && compressed_size < size) {
            header.compressed_size = compressed_size;
            header.is_compressed = true;
            
            // Update compression stats
            manager->stats.bytes_saved += size - compressed_size;
            manager->stats.messages_compressed++;
        }
    }
    
    void* send_data;
    size_t send_size;
    
    if (header.is_compressed) {
        send_data = manager->compression_buffer;
        send_size = header.compressed_size;
    } else {
        if (manager->use_zero_copy && size >= COMPRESSION_THRESHOLD) {
            // Use zero-copy for large uncompressed messages
            send_data = (void*)data;  // Remove const for zero-copy
            manager->stats.zero_copy_transfers++;
        } else {
            memcpy(manager->send_buffer, data, size);
            send_data = manager->send_buffer;
        }
        send_size = size;
    }
    
    // Send header
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE,
             dest, 0, MPI_COMM_WORLD);
    
    // Send data
    if (manager->num_pending >= MAX_PENDING_REQUESTS) {
        wait_for_pending(manager);
    }
    
    MPI_Isend(send_data, send_size, MPI_BYTE,
              dest, 1, MPI_COMM_WORLD,
              &manager->pending_requests[manager->num_pending++]);
    
    // Update stats
    manager->stats.bytes_sent += send_size;
    manager->stats.messages_sent++;
    
    pthread_mutex_unlock(&manager->mutex);
    return 0;
#else
    // Stub implementation - return error when MPI is disabled
    return -1;
#endif
}

// Receive message with optimization
int receive_optimized(CommunicationManager* manager,
                     void* data,
                     size_t* size,
                     int source,
                     MessageType* type) {
#ifndef NO_MPI
    if (!manager || !data || !size || !type) return -1;
    
    pthread_mutex_lock(&manager->mutex);
    
    // Receive header
    MessageHeader header;
    MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE,
             source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Check for aggregated message
    if (header.type == MESSAGE_AGGREGATED) {
        int status = handle_aggregated_message(manager,
                                             data,
                                             size,
                                             source,
                                             type);
        pthread_mutex_unlock(&manager->mutex);
        return status;
    }
    
    void* recv_buffer = manager->use_zero_copy &&
                       header.original_size >= COMPRESSION_THRESHOLD ?
                       data : manager->recv_buffer;
    
    // Receive data
    MPI_Recv(recv_buffer,
             header.is_compressed ?
             header.compressed_size : header.original_size,
             MPI_BYTE, source, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    
    // Decompress if needed
    if (header.is_compressed) {
        LZ4_decompress_safe(recv_buffer,
                           data,
                           header.compressed_size,
                           header.original_size);
        
        // Update stats
        manager->stats.bytes_saved += header.original_size -
                                    header.compressed_size;
        manager->stats.messages_compressed++;
    } else if (recv_buffer != data) {
        memcpy(data, recv_buffer, header.original_size);
    }
    
    *size = header.original_size;
    *type = header.type;
    
    // Update stats
    manager->stats.bytes_received += header.is_compressed ?
                                   header.compressed_size :
                                   header.original_size;
    manager->stats.messages_received++;
    
    pthread_mutex_unlock(&manager->mutex);
    return 0;
#else
    // Stub implementation - return error when MPI is disabled
    return -1;
#endif
}

// Aggregation thread function
static void* handle_aggregation(void* arg) {
    CommunicationManager* manager = (CommunicationManager*)arg;
    
    while (manager->running) {
        pthread_mutex_lock(&manager->mutex);
        
        // Check for timed out buffers
        double now = get_time_ms();
        
        for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
            AggregatedMessage* msg = &manager->aggregation_buffers[i];
            
            if (now - msg->timeout >= AGGREGATION_TIMEOUT) {
                // Send aggregated message
                flush_aggregated_message(manager, i);
                
                // Remove from list
                if (i < manager->num_aggregation_buffers - 1) {
                    memmove(&manager->aggregation_buffers[i],
                           &manager->aggregation_buffers[i + 1],
                           (manager->num_aggregation_buffers - i - 1) *
                           sizeof(AggregatedMessage));
                }
                manager->num_aggregation_buffers--;
                i--;  // Check the next message at this index
            }
        }
        
        pthread_mutex_unlock(&manager->mutex);
        usleep(1000);  // 1ms sleep
    }
    
    return NULL;
}

// Message aggregation helpers
static bool should_aggregate(CommunicationManager* manager,
                           size_t size,
                           int dest,
                           MessageType type) {
    // Find existing buffer
    for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
        AggregatedMessage* msg = &manager->aggregation_buffers[i];
        if (msg->dest_rank == dest && msg->type == type &&
            msg->size + size <= MAX_AGGREGATION_SIZE) {
            return true;
        }
    }
    
    // Or start new buffer if limit not reached
    return manager->num_aggregation_buffers < MAX_PENDING_REQUESTS;
}

static int aggregate_message(CommunicationManager* manager,
                           const void* data,
                           size_t size,
                           int dest,
                           MessageType type) {
    // Find or create buffer
    AggregatedMessage* msg = NULL;
    
    for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
        if (manager->aggregation_buffers[i].dest_rank == dest &&
            manager->aggregation_buffers[i].type == type) {
            msg = &manager->aggregation_buffers[i];
            break;
        }
    }
    
    if (!msg) {
        // Create new buffer
        if (manager->num_aggregation_buffers >= MAX_PENDING_REQUESTS) {
            return -1;
        }
        
        msg = &manager->aggregation_buffers[
            manager->num_aggregation_buffers++];
        msg->data = malloc(MAX_AGGREGATION_SIZE);
        msg->size = 0;
        msg->dest_rank = dest;
        msg->type = type;
    }
    
    // Add message
    memcpy((char*)msg->data + msg->size, data, size);
    msg->size += size;
    msg->timeout = get_time_ms();
    
    // Flush if full
    if (msg->size >= MAX_AGGREGATION_SIZE) {
        flush_aggregated_message(manager,
            msg - manager->aggregation_buffers);
    }
    
    return 0;
}

static void flush_aggregated_message(CommunicationManager* manager,
                                   size_t index) {
    AggregatedMessage* msg = &manager->aggregation_buffers[index];
    
    // Send as single message
    MessageHeader header = {
        .original_size = msg->size,
        .compressed_size = 0,
        .is_compressed = false,
        .type = MESSAGE_AGGREGATED,
        .source_rank = manager->rank,
        .dest_rank = msg->dest_rank,
        .sequence_number = manager->next_sequence++
    };
    
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE,
             msg->dest_rank, 0, MPI_COMM_WORLD);
    
    MPI_Send(msg->data, msg->size, MPI_BYTE,
             msg->dest_rank, 1, MPI_COMM_WORLD);
    
    // Update stats
    manager->stats.messages_aggregated++;
    manager->stats.bytes_saved += sizeof(MessageHeader) *
        (msg->size / COMPRESSION_THRESHOLD - 1);
    
    // Free buffer
    free(msg->data);
}

// Helper functions
static bool check_zero_copy_support(void) {
    // Check for RDMA support
    return false;  // Placeholder
}

static void* allocate_rdma_buffer(size_t size) {
    // Allocate RDMA-capable memory
    return NULL;  // Placeholder
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Wait for pending requests
static void wait_for_pending(CommunicationManager* manager) {
    if (manager->num_pending > 0) {
        MPI_Waitall(manager->num_pending,
                   manager->pending_requests,
                   MPI_STATUSES_IGNORE);
        manager->num_pending = 0;
    }
}

// Get communication statistics
CommunicationStats get_communication_stats(
    const CommunicationManager* manager) {
#ifndef NO_MPI
    return manager->mpi.stats;
#else
    return manager->stub.stats;
#endif
}

// Clean up
void cleanup_communication_manager(CommunicationManager* manager) {
#ifndef NO_MPI
    if (!manager) return;
    
    manager->running = false;
    pthread_join(manager->aggregation_thread, NULL);
    
    wait_for_pending(manager);
    pthread_mutex_destroy(&manager->mutex);
    
    if (manager->use_zero_copy) {
        // Free RDMA buffers
        free_rdma_buffer(manager->send_buffer);
        free_rdma_buffer(manager->recv_buffer);
    } else {
        free(manager->send_buffer);
        free(manager->recv_buffer);
    }
    
    free(manager->compression_buffer);
    
    for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
        free(manager->aggregation_buffers[i].data);
    }
    free(manager->aggregation_buffers);
    
    free(manager);
#else
    // Stub implementation cleanup
    if (manager) {
        free(manager->stub.send_buffer);
        free(manager->stub.recv_buffer);
        free(manager->stub.compression_buffer);
        free(manager);
    }
#endif
}
