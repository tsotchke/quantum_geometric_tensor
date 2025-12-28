/**
 * @file mpi_initialization.h
 * @brief MPI Initialization and Management for Distributed Quantum Computing
 *
 * Provides MPI setup and teardown utilities including:
 * - MPI environment initialization
 * - Process topology configuration
 * - Communicator management
 * - Thread safety configuration
 * - Error handling for MPI operations
 * - Resource cleanup and finalization
 *
 * Part of the QGTL Distributed Computing Framework.
 */

#ifndef MPI_INITIALIZATION_H
#define MPI_INITIALIZATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MPI_INIT_MAX_PROCESSORS 4096
#define MPI_INIT_MAX_NAME_LENGTH 256
#define MPI_INIT_MAX_TOPOLOGY_DIMS 8
#define MPI_INIT_MAX_COMMUNICATORS 64
#define MPI_INIT_TAG_BASE 1000

// Special rank values
#define MPI_INIT_RANK_ANY (-1)
#define MPI_INIT_RANK_ROOT 0

// ============================================================================
// Enumerations
// ============================================================================

/**
 * MPI thread support levels
 */
typedef enum {
    MPI_THREAD_SINGLE,            // Single-threaded only
    MPI_THREAD_FUNNELED,          // Multi-threaded, MPI from main only
    MPI_THREAD_SERIALIZED,        // Multi-threaded, MPI serialized
    MPI_THREAD_MULTIPLE           // Multi-threaded, full MPI support
} mpi_thread_level_t;

/**
 * Process topology types
 */
typedef enum {
    MPI_TOPO_NONE,                // No specific topology
    MPI_TOPO_CARTESIAN,           // Cartesian grid
    MPI_TOPO_GRAPH,               // General graph
    MPI_TOPO_DISTRIBUTED_GRAPH,   // Distributed graph
    MPI_TOPO_TREE,                // Tree topology
    MPI_TOPO_HYPERCUBE            // Hypercube topology
} mpi_topology_type_t;

/**
 * Communicator types
 */
typedef enum {
    MPI_COMM_TYPE_WORLD,          // World communicator
    MPI_COMM_TYPE_SELF,           // Self communicator
    MPI_COMM_TYPE_SHARED,         // Shared memory communicator
    MPI_COMM_TYPE_NODE,           // Node-local communicator
    MPI_COMM_TYPE_CUSTOM          // Custom user communicator
} mpi_comm_type_t;

/**
 * MPI initialization state
 */
typedef enum {
    MPI_STATE_UNINITIALIZED,      // Not initialized
    MPI_STATE_INITIALIZING,       // Currently initializing
    MPI_STATE_INITIALIZED,        // Successfully initialized
    MPI_STATE_FINALIZING,         // Currently finalizing
    MPI_STATE_FINALIZED,          // Successfully finalized
    MPI_STATE_ERROR               // Error state
} mpi_state_t;

/**
 * Error categories for MPI operations
 */
typedef enum {
    MPI_ERR_NONE,                 // No error
    MPI_ERR_INIT_FAILED,          // Initialization failed
    MPI_ERR_ALREADY_INIT,         // Already initialized
    MPI_ERR_NOT_INIT,             // Not initialized
    MPI_ERR_COMM_FAILED,          // Communication error
    MPI_ERR_TOPO_FAILED,          // Topology creation failed
    MPI_ERR_THREAD_LEVEL,         // Thread level not supported
    MPI_ERR_RESOURCE,             // Resource allocation failed
    MPI_ERR_INVALID_ARG,          // Invalid argument
    MPI_ERR_FINALIZE_FAILED       // Finalization failed
} mpi_error_t;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * MPI process information
 */
typedef struct {
    int rank;                     // Process rank in world
    int world_size;               // Total number of processes
    char processor_name[MPI_INIT_MAX_NAME_LENGTH];  // Processor name
    int name_length;              // Actual length of processor name
    int local_rank;               // Rank within node
    int local_size;               // Processes on same node
    int node_id;                  // Unique node identifier
    int num_nodes;                // Total number of nodes
} mpi_process_info_t;

/**
 * Cartesian topology configuration
 */
typedef struct {
    int num_dims;                 // Number of dimensions
    int dims[MPI_INIT_MAX_TOPOLOGY_DIMS];     // Size in each dimension
    bool periods[MPI_INIT_MAX_TOPOLOGY_DIMS]; // Periodicity in each dim
    bool reorder;                 // Allow process reordering
} mpi_cartesian_config_t;

/**
 * Graph topology configuration
 */
typedef struct {
    int num_vertices;             // Number of vertices (processes)
    int* neighbors;               // Neighbor list
    int* neighbor_counts;         // Edges per vertex
    int total_edges;              // Total number of edges
    bool reorder;                 // Allow process reordering
} mpi_graph_config_t;

/**
 * Communicator handle wrapper
 */
typedef struct {
    int handle;                   // Underlying MPI_Comm handle
    mpi_comm_type_t type;         // Communicator type
    char name[MPI_INIT_MAX_NAME_LENGTH];  // Communicator name
    int size;                     // Number of processes
    int rank;                     // This process's rank
    bool is_intercomm;            // Is intercommunicator
} mpi_communicator_t;

/**
 * MPI timing information
 */
typedef struct {
    double init_time_sec;         // Time for initialization
    double finalize_time_sec;     // Time for finalization
    double total_comm_time_sec;   // Total communication time
    uint64_t messages_sent;       // Messages sent
    uint64_t messages_received;   // Messages received
    uint64_t bytes_sent;          // Bytes sent
    uint64_t bytes_received;      // Bytes received
} mpi_timing_t;

/**
 * MPI configuration options
 */
typedef struct {
    mpi_thread_level_t thread_level;      // Desired thread level
    bool enable_error_handler;            // Enable custom error handler
    bool enable_timing;                   // Enable timing collection
    bool enable_profiling;                // Enable MPI profiling
    mpi_topology_type_t topology;         // Topology type
    mpi_cartesian_config_t cartesian;     // Cartesian config (if applicable)
    char config_file[MPI_INIT_MAX_NAME_LENGTH];  // Config file path
} mpi_config_t;

/**
 * MPI environment state
 */
typedef struct {
    mpi_state_t state;            // Current state
    mpi_thread_level_t thread_level_provided;  // Actual thread level
    mpi_process_info_t process;   // Process information
    mpi_timing_t timing;          // Timing information
    mpi_error_t last_error;       // Last error code
    char error_message[MPI_INIT_MAX_NAME_LENGTH];  // Error message
    bool is_thread_safe;          // Thread safety flag
} mpi_environment_t;

/**
 * Opaque MPI manager handle
 */
typedef struct mpi_manager mpi_manager_t;

// ============================================================================
// Initialization and Finalization
// ============================================================================

/**
 * Create MPI manager (does not initialize MPI)
 */
mpi_manager_t* mpi_manager_create(void);

/**
 * Create manager with configuration
 */
mpi_manager_t* mpi_manager_create_with_config(const mpi_config_t* config);

/**
 * Get default configuration
 */
mpi_config_t mpi_default_config(void);

/**
 * Initialize MPI environment
 */
bool mpi_init(mpi_manager_t* manager, int* argc, char*** argv);

/**
 * Initialize MPI with thread support
 */
bool mpi_init_thread(mpi_manager_t* manager,
                     int* argc,
                     char*** argv,
                     mpi_thread_level_t required,
                     mpi_thread_level_t* provided);

/**
 * Check if MPI is initialized
 */
bool mpi_is_initialized(mpi_manager_t* manager);

/**
 * Check if MPI is finalized
 */
bool mpi_is_finalized(mpi_manager_t* manager);

/**
 * Finalize MPI environment
 */
bool mpi_finalize(mpi_manager_t* manager);

/**
 * Abort MPI (emergency shutdown)
 */
void mpi_abort(mpi_manager_t* manager, int error_code);

/**
 * Destroy MPI manager
 */
void mpi_manager_destroy(mpi_manager_t* manager);

// ============================================================================
// Process Information
// ============================================================================

/**
 * Get process information
 */
bool mpi_get_process_info(mpi_manager_t* manager,
                          mpi_process_info_t* info);

/**
 * Get this process's rank
 */
int mpi_get_rank(mpi_manager_t* manager);

/**
 * Get world size
 */
int mpi_get_world_size(mpi_manager_t* manager);

/**
 * Get processor name
 */
const char* mpi_get_processor_name(mpi_manager_t* manager);

/**
 * Get node-local rank
 */
int mpi_get_local_rank(mpi_manager_t* manager);

/**
 * Get number of processes on this node
 */
int mpi_get_local_size(mpi_manager_t* manager);

/**
 * Check if this is the root process
 */
bool mpi_is_root(mpi_manager_t* manager);

// ============================================================================
// Communicator Management
// ============================================================================

/**
 * Get world communicator
 */
mpi_communicator_t* mpi_get_world_comm(mpi_manager_t* manager);

/**
 * Get self communicator
 */
mpi_communicator_t* mpi_get_self_comm(mpi_manager_t* manager);

/**
 * Create shared memory communicator
 */
mpi_communicator_t* mpi_create_shared_comm(mpi_manager_t* manager);

/**
 * Create node-local communicator
 */
mpi_communicator_t* mpi_create_node_comm(mpi_manager_t* manager);

/**
 * Split communicator by color
 */
mpi_communicator_t* mpi_comm_split(mpi_manager_t* manager,
                                   mpi_communicator_t* comm,
                                   int color,
                                   int key);

/**
 * Duplicate communicator
 */
mpi_communicator_t* mpi_comm_dup(mpi_manager_t* manager,
                                  mpi_communicator_t* comm);

/**
 * Free communicator
 */
void mpi_comm_free(mpi_manager_t* manager, mpi_communicator_t* comm);

/**
 * Set communicator name
 */
bool mpi_comm_set_name(mpi_communicator_t* comm, const char* name);

/**
 * Get communicator name
 */
const char* mpi_comm_get_name(mpi_communicator_t* comm);

// ============================================================================
// Topology Configuration
// ============================================================================

/**
 * Create Cartesian topology
 */
mpi_communicator_t* mpi_create_cart_topology(
    mpi_manager_t* manager,
    const mpi_cartesian_config_t* config);

/**
 * Get Cartesian coordinates for rank
 */
bool mpi_cart_coords(mpi_manager_t* manager,
                     mpi_communicator_t* comm,
                     int rank,
                     int* coords);

/**
 * Get rank from Cartesian coordinates
 */
int mpi_cart_rank(mpi_manager_t* manager,
                  mpi_communicator_t* comm,
                  const int* coords);

/**
 * Get shifted ranks for Cartesian communication
 */
bool mpi_cart_shift(mpi_manager_t* manager,
                    mpi_communicator_t* comm,
                    int direction,
                    int displacement,
                    int* source,
                    int* dest);

/**
 * Create graph topology
 */
mpi_communicator_t* mpi_create_graph_topology(
    mpi_manager_t* manager,
    const mpi_graph_config_t* config);

/**
 * Optimize topology for hardware
 */
bool mpi_optimize_topology(mpi_manager_t* manager,
                           mpi_communicator_t* comm);

// ============================================================================
// Barrier and Synchronization
// ============================================================================

/**
 * Global barrier on world communicator
 */
bool mpi_barrier_world(mpi_manager_t* manager);

/**
 * Barrier on specific communicator
 */
bool mpi_barrier(mpi_manager_t* manager, mpi_communicator_t* comm);

/**
 * Non-blocking barrier
 */
bool mpi_ibarrier(mpi_manager_t* manager,
                  mpi_communicator_t* comm,
                  int* request);

/**
 * Wait for request completion
 */
bool mpi_wait(mpi_manager_t* manager, int request);

/**
 * Test if request is complete (non-blocking)
 */
bool mpi_test(mpi_manager_t* manager, int request, bool* completed);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Get last error code
 */
mpi_error_t mpi_get_last_error(mpi_manager_t* manager);

/**
 * Get last error message
 */
const char* mpi_get_error_message(mpi_manager_t* manager);

/**
 * Get error string for code
 */
const char* mpi_error_string(mpi_error_t error);

/**
 * Clear error state
 */
void mpi_clear_error(mpi_manager_t* manager);

/**
 * Set custom error handler
 */
typedef void (*mpi_error_handler_t)(mpi_error_t error,
                                    const char* message,
                                    void* user_data);

bool mpi_set_error_handler(mpi_manager_t* manager,
                           mpi_error_handler_t handler,
                           void* user_data);

// ============================================================================
// Environment Queries
// ============================================================================

/**
 * Get MPI environment state
 */
bool mpi_get_environment(mpi_manager_t* manager, mpi_environment_t* env);

/**
 * Get thread level provided
 */
mpi_thread_level_t mpi_query_thread_level(mpi_manager_t* manager);

/**
 * Check if thread-safe operations are supported
 */
bool mpi_is_thread_safe(mpi_manager_t* manager);

/**
 * Get MPI version
 */
bool mpi_get_version(int* major, int* minor);

/**
 * Get MPI library version string
 */
const char* mpi_get_library_version(void);

// ============================================================================
// Timing and Statistics
// ============================================================================

/**
 * Get wall clock time
 */
double mpi_wtime(void);

/**
 * Get clock resolution
 */
double mpi_wtick(void);

/**
 * Get timing statistics
 */
bool mpi_get_timing(mpi_manager_t* manager, mpi_timing_t* timing);

/**
 * Reset timing statistics
 */
void mpi_reset_timing(mpi_manager_t* manager);

// ============================================================================
// Reporting
// ============================================================================

/**
 * Generate MPI environment report
 */
char* mpi_generate_report(mpi_manager_t* manager);

/**
 * Export to JSON
 */
char* mpi_export_json(mpi_manager_t* manager);

/**
 * Export to file
 */
bool mpi_export_to_file(mpi_manager_t* manager, const char* filename);

/**
 * Print summary to stdout (root only)
 */
void mpi_print_summary(mpi_manager_t* manager);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get thread level name
 */
const char* mpi_thread_level_name(mpi_thread_level_t level);

/**
 * Get topology type name
 */
const char* mpi_topology_type_name(mpi_topology_type_t type);

/**
 * Get communicator type name
 */
const char* mpi_comm_type_name(mpi_comm_type_t type);

/**
 * Get state name
 */
const char* mpi_state_name(mpi_state_t state);

/**
 * Free allocated string
 */
void mpi_free_string(char* str);

// ============================================================================
// Convenience Macros
// ============================================================================

/**
 * Root-only execution macro
 */
#define MPI_ROOT_ONLY(manager, code) \
    do { \
        if (mpi_is_root(manager)) { \
            code; \
        } \
    } while(0)

/**
 * All-but-root execution macro
 */
#define MPI_WORKERS_ONLY(manager, code) \
    do { \
        if (!mpi_is_root(manager)) { \
            code; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif // MPI_INITIALIZATION_H
