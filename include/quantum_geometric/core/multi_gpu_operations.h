#ifndef MULTI_GPU_OPERATIONS_H
#define MULTI_GPU_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MultiGPUManager MultiGPUManager;
typedef void (*KernelFunction)(void*, void*);

// Data types for tensor operations
typedef enum {
    TYPE_FLOAT,           // Single precision float
    TYPE_DOUBLE,          // Double precision float
    TYPE_COMPLEX_FLOAT,   // Complex single precision
    TYPE_COMPLEX_DOUBLE,  // Complex double precision
    TYPE_INT,             // Integer
    TYPE_QUANTUM         // Quantum state
} DataType;

// Reduction operations
typedef enum {
    REDUCE_SUM,          // Sum reduction
    REDUCE_PROD,         // Product reduction
    REDUCE_MAX,          // Maximum reduction
    REDUCE_MIN,          // Minimum reduction
    REDUCE_NORM,         // Norm reduction
    REDUCE_QUANTUM      // Quantum state reduction
} ReduceOp;

// Memory types
typedef enum {
    MEM_HOST,            // Host memory
    MEM_DEVICE,          // Device memory
    MEM_UNIFIED,         // Unified memory
    MEM_PINNED,         // Pinned memory
    MEM_QUANTUM         // Quantum memory
} MemoryType;

// Synchronization modes
typedef enum {
    SYNC_DEFAULT,        // Default synchronization
    SYNC_BARRIER,        // Barrier synchronization
    SYNC_EVENT,          // Event-based synchronization
    SYNC_STREAM,         // Stream synchronization
    SYNC_QUANTUM        // Quantum synchronization
} SyncMode;

// Device properties
typedef struct {
    int device_id;                // Device ID
    size_t total_memory;          // Total memory
    size_t available_memory;      // Available memory
    int compute_capability;       // Compute capability
    int num_multiprocessors;      // Number of multiprocessors
    int max_threads_per_block;    // Max threads per block
    bool supports_quantum;        // Quantum support flag
} DeviceProperties;

// Kernel configuration
typedef struct {
    size_t grid_size[3];         // Grid dimensions
    size_t block_size[3];        // Block dimensions
    size_t shared_memory;        // Shared memory size
    int stream_id;               // Stream identifier
    bool enable_profiling;       // Profiling flag
    void* kernel_params;         // Additional parameters
} KernelConfig;

// Performance metrics
typedef struct {
    double execution_time;        // Execution time
    double memory_throughput;     // Memory throughput
    double compute_throughput;    // Compute throughput
    double efficiency;            // GPU efficiency
    size_t memory_used;          // Memory usage
    void* metric_data;           // Additional metrics
} PerformanceMetrics;

// Core functions
MultiGPUManager* init_multi_gpu_manager(void);
void cleanup_multi_gpu_manager(MultiGPUManager* manager);

// Device management
int get_device_count(MultiGPUManager* manager);
int get_device_properties(MultiGPUManager* manager, int device_id, DeviceProperties* props);
int set_device(MultiGPUManager* manager, int device_id);
int enable_peer_access(MultiGPUManager* manager, int device_id, int peer_id);

// Memory operations
int allocate_memory(MultiGPUManager* manager, void** ptr, size_t size, MemoryType type);
int free_memory(MultiGPUManager* manager, void* ptr, MemoryType type);
int copy_memory(MultiGPUManager* manager, void* dst, const void* src, size_t size, 
               MemoryType dst_type, MemoryType src_type);
int memset_memory(MultiGPUManager* manager, void* ptr, int value, size_t size);

// Data distribution
int distribute_tensor(MultiGPUManager* manager, const void* host_data, size_t size, DataType dtype);
int gather_results(MultiGPUManager* manager, void* host_data, size_t size);
int redistribute_data(MultiGPUManager* manager, void* data, size_t size, int* device_map);

// Collective operations
int all_reduce(MultiGPUManager* manager, void* data, size_t count, DataType dtype, ReduceOp op);
int broadcast(MultiGPUManager* manager, void* data, size_t size, int root_device);
int all_gather(MultiGPUManager* manager, void* data, size_t size);
int reduce_scatter(MultiGPUManager* manager, void* data, size_t size, ReduceOp op);

// Kernel execution
int execute_multi_gpu_kernel(MultiGPUManager* manager, KernelFunction kernel, 
                           void* args, const KernelConfig* config);
int launch_cooperative_kernel(MultiGPUManager* manager, KernelFunction kernel,
                            void* args, const KernelConfig* config);
int synchronize_kernel(MultiGPUManager* manager, SyncMode mode);

// Stream management
int create_stream(MultiGPUManager* manager, int device_id, int* stream_id);
int destroy_stream(MultiGPUManager* manager, int stream_id);
int synchronize_stream(MultiGPUManager* manager, int stream_id);
int wait_event(MultiGPUManager* manager, int event_id);

// Performance monitoring
int start_profiling(MultiGPUManager* manager);
int stop_profiling(MultiGPUManager* manager);
int get_performance_metrics(MultiGPUManager* manager, PerformanceMetrics* metrics);
int reset_performance_metrics(MultiGPUManager* manager);

// Quantum-specific operations
int allocate_quantum_memory(MultiGPUManager* manager, void** ptr, size_t num_qubits);
int execute_quantum_kernel(MultiGPUManager* manager, KernelFunction kernel,
                         void* quantum_state, const KernelConfig* config);
int synchronize_quantum_state(MultiGPUManager* manager, void* quantum_state);

// Error handling
const char* get_last_error(MultiGPUManager* manager);
int get_error_code(MultiGPUManager* manager);
void clear_error(MultiGPUManager* manager);

#ifdef __cplusplus
}
#endif

#endif // MULTI_GPU_OPERATIONS_H
