#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#endif

#include "quantum_geometric/distributed/communication_optimization.h"

#ifdef HAVE_LZ4
#include <lz4.h>
#else
// LZ4 fallback stubs - no compression when LZ4 not available
static inline int LZ4_compress_default(const char* src, char* dst, int srcSize, int dstCapacity) {
    if (srcSize > dstCapacity) return 0;
    memcpy(dst, src, srcSize);
    return srcSize;
}
static inline int LZ4_decompress_safe(const char* src, char* dst, int compressedSize, int dstCapacity) {
    if (compressedSize > dstCapacity) return -1;
    memcpy(dst, src, compressedSize);
    return compressedSize;
}
static inline int LZ4_compressBound(int inputSize) { return inputSize; }
#endif

// Define NO_MPI by default since MPI is optional
#ifndef HAS_MPI
#define NO_MPI
#endif

#ifndef NO_MPI
#include <mpi.h>
#endif

// Communication parameters
#define MAX_BUFFER_SIZE (64 * 1024 * 1024)  // 64MB
#define COMPRESSION_THRESHOLD (16 * 1024)    // 16KB
#define MAX_PENDING_REQUESTS 32
#define AGGREGATION_TIMEOUT 100  // ms
#define MAX_AGGREGATION_SIZE (1024 * 1024)  // 1MB

// RDMA buffer tracking to prevent mmap/malloc confusion
#define MAX_TRACKED_RDMA_BUFFERS 64

typedef struct {
    void* buffer;
    size_t size;
    bool is_mmap;
} RDMABufferInfo;

static RDMABufferInfo g_rdma_buffers[MAX_TRACKED_RDMA_BUFFERS];
static size_t g_num_rdma_buffers = 0;
static pthread_mutex_t g_rdma_mutex = PTHREAD_MUTEX_INITIALIZER;

static void track_rdma_buffer(void* buffer, size_t size, bool is_mmap) {
    pthread_mutex_lock(&g_rdma_mutex);
    if (g_num_rdma_buffers < MAX_TRACKED_RDMA_BUFFERS) {
        g_rdma_buffers[g_num_rdma_buffers].buffer = buffer;
        g_rdma_buffers[g_num_rdma_buffers].size = size;
        g_rdma_buffers[g_num_rdma_buffers].is_mmap = is_mmap;
        g_num_rdma_buffers++;
    }
    pthread_mutex_unlock(&g_rdma_mutex);
}

static bool untrack_rdma_buffer(void* buffer, size_t* size_out, bool* is_mmap_out) {
    pthread_mutex_lock(&g_rdma_mutex);
    for (size_t i = 0; i < g_num_rdma_buffers; i++) {
        if (g_rdma_buffers[i].buffer == buffer) {
            *size_out = g_rdma_buffers[i].size;
            *is_mmap_out = g_rdma_buffers[i].is_mmap;
            // Remove by shifting
            if (i < g_num_rdma_buffers - 1) {
                memmove(&g_rdma_buffers[i], &g_rdma_buffers[i + 1],
                       (g_num_rdma_buffers - i - 1) * sizeof(RDMABufferInfo));
            }
            g_num_rdma_buffers--;
            pthread_mutex_unlock(&g_rdma_mutex);
            return true;
        }
    }
    pthread_mutex_unlock(&g_rdma_mutex);
    return false;
}

// Forward declarations for helper functions
static double get_time_ms(void);
static void flush_aggregated_message_at(CommunicationManager* manager, size_t index);
static bool should_aggregate(CommunicationManager* manager, size_t size, int dest, MessageType type);
static int aggregate_message(CommunicationManager* manager, const void* data, size_t size, int dest, MessageType type);
static void* handle_aggregation(void* arg);
static void wait_for_pending(CommunicationManager* manager);
static int process_aggregated_message(CommunicationManager* manager, const void* data, size_t size, int source, MessageType type);

// Message header for communication
typedef struct {
    size_t original_size;
    size_t compressed_size;
    bool is_compressed;
    MessageType type;
    int source_rank;
    int dest_rank;
    uint64_t sequence_number;
} MessageHeader;

// Aggregated message structure
typedef struct {
    void* data;
    size_t size;
    double timeout;
    int dest_rank;
    MessageType type;
} AggregatedMessage;

// Communication manager implementation
struct CommunicationManager {
    void* send_buffer;
    void* recv_buffer;
    void* compression_buffer;
    size_t buffer_size;
    size_t num_pending;
    pthread_mutex_t mutex;
    pthread_t aggregation_thread;
    AggregatedMessage* aggregation_buffers;
    size_t num_aggregation_buffers;
    bool running;
    uint64_t next_sequence;
    bool use_zero_copy;
    CommunicationStats stats;
    bool initialized;
    int rank;
    int world_size;
#ifndef NO_MPI
    MPI_Request pending_requests[MAX_PENDING_REQUESTS];
#else
    int pending_requests[MAX_PENDING_REQUESTS];
#endif
};

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
    // Stub implementation (no MPI)
    CommunicationManager* manager = malloc(sizeof(CommunicationManager));
    if (!manager) return NULL;

    manager->send_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    manager->recv_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    manager->compression_buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    manager->buffer_size = MAX_BUFFER_SIZE;
    manager->num_pending = 0;
    manager->running = true;
    manager->next_sequence = 0;
    manager->use_zero_copy = false;
    manager->initialized = true;
    manager->rank = 0;
    manager->world_size = 1;
    manager->aggregation_buffers = NULL;
    manager->num_aggregation_buffers = 0;
    memset(&manager->stats, 0, sizeof(CommunicationStats));
    pthread_mutex_init(&manager->mutex, NULL);

    if (!manager->send_buffer || !manager->recv_buffer || !manager->compression_buffer) {
        cleanup_communication_manager(manager);
        return NULL;
    }

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
        
        if (compressed_size > 0 && (size_t)compressed_size < size) {
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
        int status = process_aggregated_message(manager,
                                                data,
                                                *size,
                                                source,
                                                *type);
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
                flush_aggregated_message_at(manager, i);
                
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
        flush_aggregated_message_at(manager,
            (size_t)(msg - manager->aggregation_buffers));
    }
    
    return 0;
}

static void flush_aggregated_message_at(CommunicationManager* manager,
                                         size_t index) {
#ifndef NO_MPI
    AggregatedMessage* msg = &manager->aggregation_buffers[index];

    // Send as single message
    MessageHeader header = {
        .original_size = msg->size,
        .compressed_size = 0,
        .is_compressed = false,
        .type = msg->type,
        .source_rank = manager->rank,
        .dest_rank = msg->dest_rank,
        .sequence_number = manager->next_sequence++
    };

    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE,
             msg->dest_rank, 0, MPI_COMM_WORLD);

    MPI_Send(msg->data, msg->size, MPI_BYTE,
             msg->dest_rank, 1, MPI_COMM_WORLD);

    // Update stats
    manager->stats.messages_sent++;
    manager->stats.bytes_sent += msg->size;

    // Free buffer
    free(msg->data);
#else
    (void)manager;
    (void)index;
#endif
}

// Helper functions
bool check_zero_copy_support(void) {
    // Check for RDMA/InfiniBand support through environment and runtime detection

    // Method 1: Check for RDMA environment variables
    const char* rdma_env = getenv("OMPI_MCA_btl_openib_want_cuda_gdr");
    const char* ucx_env = getenv("UCX_TLS");

    if (rdma_env && strcmp(rdma_env, "1") == 0) {
        return true;
    }

    if (ucx_env && strstr(ucx_env, "rc") != NULL) {
        return true;  // UCX with reliable connection (InfiniBand)
    }

    // Method 2: Check for InfiniBand device presence
#ifdef __linux__
    struct stat st;
    if (stat("/sys/class/infiniband", &st) == 0 && S_ISDIR(st.st_mode)) {
        // InfiniBand subsystem exists, check for devices
        DIR* dir = opendir("/sys/class/infiniband");
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_name[0] != '.') {
                    closedir(dir);
                    return true;  // Found InfiniBand device
                }
            }
            closedir(dir);
        }
    }
#endif

#ifndef NO_MPI
    // Method 3: Query MPI for RDMA capability
    int flag = 0;
    void* attr_val;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attr_val, &flag);
    if (flag) {
        // Large tag range often indicates high-performance interconnect
        int max_tag = *(int*)attr_val;
        if (max_tag > 1000000) {
            return true;  // Likely high-performance interconnect
        }
    }
#endif

    return false;
}

void* allocate_rdma_buffer(size_t size) {
    // Allocate memory suitable for RDMA operations
    // Requirements: Page-aligned, locked in physical memory, registered

    void* buffer = NULL;
    bool is_mmap = false;

    // Use mmap for page-aligned allocation with MAP_LOCKED if available
#ifdef __linux__
    buffer = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

    if (buffer == MAP_FAILED) {
        // Fall back to regular mmap without huge pages
        buffer = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }

    if (buffer != MAP_FAILED) {
        is_mmap = true;
        // Lock memory to prevent paging
        if (mlock(buffer, size) != 0) {
            // mlock failed but continue with warning
            // The buffer is still usable, just may not be optimal for RDMA
        }
        track_rdma_buffer(buffer, size, is_mmap);
        return buffer;
    }
#endif

    // Fallback: use posix_memalign for page-aligned allocation
    size_t page_size = sysconf(_SC_PAGESIZE);
    if (page_size == 0) page_size = 4096;

    if (posix_memalign(&buffer, page_size, size) == 0) {
        // Try to lock memory
#ifdef __linux__
        mlock(buffer, size);
#endif
        track_rdma_buffer(buffer, size, false);  // Not mmap
        return buffer;
    }

    // Last resort: regular aligned allocation
    buffer = aligned_alloc(64, size);
    if (buffer) {
        track_rdma_buffer(buffer, size, false);  // Not mmap
    }
    return buffer;
}

void free_rdma_buffer(void* buffer) {
    if (!buffer) return;

    // Look up the buffer in our tracking table to determine the correct free method
    size_t size = 0;
    bool is_mmap = false;

    if (untrack_rdma_buffer(buffer, &size, &is_mmap)) {
        // Found in tracking table - use the correct free method
        if (is_mmap) {
#ifdef __linux__
            munlock(buffer, size);
            munmap(buffer, size);
#endif
        } else {
            free(buffer);
        }
    } else {
        // Buffer not in tracking table - this is an error condition
        // but we must not crash. Log and use free as fallback (safest guess)
        // In production, this indicates a programming error.
        fprintf(stderr, "WARNING: free_rdma_buffer called on untracked buffer %p\n", buffer);
        free(buffer);
    }
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Wait for pending requests
static void wait_for_pending(CommunicationManager* manager) {
#ifndef NO_MPI
    if (manager->num_pending > 0) {
        MPI_Waitall(manager->num_pending,
                   manager->pending_requests,
                   MPI_STATUSES_IGNORE);
        manager->num_pending = 0;
    }
#else
    (void)manager;
#endif
}

// Get communication statistics
CommunicationStats get_communication_stats(
    const CommunicationManager* manager) {
    return manager->stats;
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
        pthread_mutex_destroy(&manager->mutex);
        free(manager->send_buffer);
        free(manager->recv_buffer);
        free(manager->compression_buffer);
        free(manager);
    }
#endif
}

// Process aggregated message (internal helper for receive)
static int process_aggregated_message(CommunicationManager* manager,
                                       const void* data, size_t size,
                                       int source, MessageType type) {
    if (!manager || !data) return -1;

    // Aggregated messages contain multiple sub-messages
    // Process each sub-message by extracting and handling individually
    (void)size;
    (void)source;
    (void)type;

    // In production, this would parse the aggregated buffer and
    // dispatch each sub-message to appropriate handlers
    manager->stats.messages_received++;

    return 0;
}

// Handle aggregated message (public API - matches header)
int handle_aggregated_message(CommunicationManager* manager,
                               const void* data, size_t size) {
    if (!manager || !data || size == 0) return -1;

    pthread_mutex_lock(&manager->mutex);

    // Store aggregated data for later processing
    // Find an available aggregation slot
    if (manager->num_aggregation_buffers < MAX_PENDING_REQUESTS) {
        AggregatedMessage* msg = &manager->aggregation_buffers[
            manager->num_aggregation_buffers++];

        msg->data = malloc(size);
        if (msg->data) {
            memcpy(msg->data, data, size);
            msg->size = size;
            msg->timeout = get_time_ms();
            msg->type = MESSAGE_AGGREGATED;
            msg->dest_rank = manager->rank;  // Local processing
        }
    }

    manager->stats.messages_received++;

    pthread_mutex_unlock(&manager->mutex);
    return 0;
}

// Flush all aggregated messages (public API - matches header)
int flush_aggregated_message(CommunicationManager* manager) {
    if (!manager) return -1;

    pthread_mutex_lock(&manager->mutex);

    // Flush all pending aggregated messages
    for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
        flush_aggregated_message_at(manager, i);
    }

    // Clear the aggregation buffers after flushing
    for (size_t i = 0; i < manager->num_aggregation_buffers; i++) {
        free(manager->aggregation_buffers[i].data);
        manager->aggregation_buffers[i].data = NULL;
        manager->aggregation_buffers[i].size = 0;
    }
    manager->num_aggregation_buffers = 0;

    pthread_mutex_unlock(&manager->mutex);
    return 0;
}

// Reset communication statistics
void reset_communication_stats(CommunicationManager* manager) {
    if (!manager) return;

    pthread_mutex_lock(&manager->mutex);
    memset(&manager->stats, 0, sizeof(CommunicationStats));
    pthread_mutex_unlock(&manager->mutex);
}

// Get local rank
int get_rank(const CommunicationManager* manager) {
    if (!manager) return -1;
    return manager->rank;
}

// Get world size from a CommunicationManager
// Note: get_world_size(void) exists in distributed_training.c for the global world size
int get_comm_world_size(const CommunicationManager* manager) {
    if (!manager) return -1;
    return manager->world_size;
}

// Barrier synchronization
void barrier(CommunicationManager* manager) {
    if (!manager) return;

#ifndef NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#else
    (void)manager;
#endif
}
