/**
 * @file elastic_scaling.c
 * @brief Elastic scaling implementation for distributed training
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

// MPI guard
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs
typedef int MPI_Comm;
typedef int MPI_Group;
typedef int MPI_Status;
typedef int MPI_Info;
typedef int MPI_Datatype;
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL 0
#define MPI_BYTE 0
#define MPI_INT 0
#define MPI_SUCCESS 0
#define MPI_INFO_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_UNDEFINED (-1)

// Stub MPI functions
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { *size = 1; return 0; }
static inline int MPI_Comm_dup(MPI_Comm comm, MPI_Comm* newcomm) { *newcomm = 0; return 0; }
static inline int MPI_Comm_free(MPI_Comm* comm) { *comm = 0; return 0; }
static inline int MPI_Comm_group(MPI_Comm comm, MPI_Group* group) { *group = 0; return 0; }
static inline int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm) { *newcomm = 0; return 0; }
static inline int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm) { *newcomm = 0; return 0; }
static inline int MPI_Group_incl(MPI_Group group, int n, const int* ranks, MPI_Group* newgroup) { *newgroup = 0; return 0; }
static inline int MPI_Group_free(MPI_Group* group) { *group = 0; return 0; }
static inline int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { return 0; }
static inline int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { return 0; }
static inline int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { return 0; }
static inline int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { return 0; }
static inline int MPI_Barrier(MPI_Comm comm) { return 0; }
#endif

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#endif

#include "quantum_geometric/distributed/elastic_scaling.h"

// Forward declarations for static functions
static void* monitor_resources(void* arg);
static void update_node_status(ElasticManager* manager);
static void check_node_health(ElasticManager* manager);
static void handle_node_failure(ElasticManager* manager, size_t node_id);
static void check_scaling_needs(ElasticManager* manager);
static void scale_up(ElasticManager* manager, size_t target);
static void scale_down(ElasticManager* manager, size_t target);
static void activate_node(ElasticManager* manager, size_t node_id);
static void deactivate_node(ElasticManager* manager, size_t node_id);
static void request_new_nodes(ElasticManager* manager, size_t count);
static void redistribute_workload(ElasticManager* manager);
static double get_gpu_utilization(void);
static double get_network_utilization(void);

// Use macros from header with local fallbacks
#ifndef MIN_WORKERS
#define MIN_WORKERS ELASTIC_MIN_WORKERS
#endif
#ifndef MAX_WORKERS
#define MAX_WORKERS ELASTIC_MAX_WORKERS
#endif
#ifndef SCALING_INTERVAL
#define SCALING_INTERVAL ELASTIC_SCALING_INTERVAL
#endif
#ifndef LOAD_THRESHOLD
#define LOAD_THRESHOLD ELASTIC_LOAD_THRESHOLD
#endif
#ifndef SCALE_FACTOR
#define SCALE_FACTOR ELASTIC_SCALE_FACTOR
#endif

// Initialize elastic scaling
ElasticManager* init_elastic_scaling(void) {
    ElasticManager* manager = malloc(sizeof(ElasticManager));
    if (!manager) return NULL;
    
    // Get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &manager->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &manager->world_size);
    
    manager->num_nodes = manager->world_size;
    manager->active_nodes = manager->world_size;
    manager->running = true;
    
    // Create communicator for scaling operations
    MPI_Comm_dup(MPI_COMM_WORLD, &manager->scale_comm);
    
    // Allocate node status array
    manager->nodes = calloc(MAX_WORKERS, sizeof(NodeStatus));
    if (!manager->nodes) {
        free(manager);
        return NULL;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&manager->mutex, NULL) != 0) {
        free(manager->nodes);
        free(manager);
        return NULL;
    }
    
    // Start monitoring thread
    if (pthread_create(&manager->monitor_thread,
                      NULL,
                      monitor_resources,
                      manager) != 0) {
        pthread_mutex_destroy(&manager->mutex);
        free(manager->nodes);
        free(manager);
        return NULL;
    }
    
    return manager;
}

// Monitor thread function
static void* monitor_resources(void* arg) {
    ElasticManager* manager = (ElasticManager*)arg;
    
    while (manager->running) {
        pthread_mutex_lock(&manager->mutex);
        
        // Update node status
        update_node_status(manager);
        
        // Check for failed nodes
        check_node_health(manager);
        
        // Scale if needed
        check_scaling_needs(manager);
        
        pthread_mutex_unlock(&manager->mutex);
        
        // Wait for next interval
        sleep(SCALING_INTERVAL);
    }
    
    return NULL;
}

// Update node status
static void update_node_status(ElasticManager* manager) {
    NodeStatus status;
    memset(&status, 0, sizeof(NodeStatus));
    status.rank = manager->world_rank;

#ifdef __linux__
    // Linux: Use sysinfo for CPU and memory
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        // Convert load average to utilization (0-1 range)
        // loads[0] is 1-minute average, scaled by 2^SI_LOAD_SHIFT
        double load_avg = (double)si.loads[0] / (double)(1 << SI_LOAD_SHIFT);
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        status.cpu_util = fmin(load_avg / (double)num_cpus, 1.0);

        // Memory utilization
        if (si.totalram > 0) {
            status.memory_util = (double)(si.totalram - si.freeram) / (double)si.totalram;
        }
    }
#elif defined(__APPLE__)
    // macOS: Use getloadavg for CPU and mach APIs for memory
    double loadavg[1];
    if (getloadavg(loadavg, 1) == 1) {
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        status.cpu_util = fmin(loadavg[0] / (double)num_cpus, 1.0);
    }

    // macOS memory via mach API
    mach_port_t host_port = mach_host_self();
    vm_size_t page_size;
    host_page_size(host_port, &page_size);

    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        uint64_t used = (uint64_t)(vm_stats.active_count + vm_stats.wire_count) * page_size;
        uint64_t total = (uint64_t)(vm_stats.active_count + vm_stats.inactive_count +
                                     vm_stats.wire_count + vm_stats.free_count) * page_size;
        if (total > 0) {
            status.memory_util = (double)used / (double)total;
        }
    }
#else
    // Generic fallback using getloadavg (POSIX)
    double loadavg[1];
    if (getloadavg(loadavg, 1) == 1) {
        status.cpu_util = fmin(loadavg[0], 1.0);
    }
    status.memory_util = 0.5;  // Assume 50% if we can't determine
#endif

    // Get GPU utilization if available
    status.gpu_util = get_gpu_utilization();

    // Get network utilization
    status.network_util = get_network_utilization();

    status.is_active = true;
    status.last_heartbeat = time(NULL);

    // Share status with all nodes
    MPI_Allgather(&status, sizeof(NodeStatus), MPI_BYTE,
                  manager->nodes, sizeof(NodeStatus), MPI_BYTE,
                  manager->scale_comm);
}

// Check node health
static void check_node_health(ElasticManager* manager) {
    time_t now = time(NULL);
    size_t active = 0;
    
    for (size_t i = 0; i < manager->num_nodes; i++) {
        NodeStatus* node = &manager->nodes[i];
        
        // Check if node is responsive
        if (now - node->last_heartbeat > SCALING_INTERVAL * 2) {
            node->is_active = false;
            handle_node_failure(manager, i);
        }
        
        if (node->is_active) active++;
    }
    
    manager->active_nodes = active;
}

// Handle node failure
static void handle_node_failure(ElasticManager* manager, size_t node_id) {
    // Notify all nodes
    MPI_Barrier(manager->scale_comm);
    
    // Redistribute workload
    redistribute_workload(manager);
    
    // Update communicator
    MPI_Comm new_comm;
    int* ranks = malloc(manager->active_nodes * sizeof(int));
    size_t idx = 0;
    
    for (size_t i = 0; i < manager->num_nodes; i++) {
        if (manager->nodes[i].is_active) {
            ranks[idx++] = i;
        }
    }
    
    MPI_Group world_group, new_group;
    MPI_Comm_group(manager->scale_comm, &world_group);
    MPI_Group_incl(world_group, manager->active_nodes, ranks, &new_group);
    MPI_Comm_create(manager->scale_comm, new_group, &new_comm);
    
    // Update communicator
    MPI_Comm_free(&manager->scale_comm);
    manager->scale_comm = new_comm;
    
    free(ranks);
    MPI_Group_free(&world_group);
    MPI_Group_free(&new_group);
}

// Check scaling needs
static void check_scaling_needs(ElasticManager* manager) {
    double avg_load = 0.0;
    size_t overloaded = 0;
    
    // Calculate average load
    for (size_t i = 0; i < manager->num_nodes; i++) {
        if (!manager->nodes[i].is_active) continue;
        
        double load = max(manager->nodes[i].cpu_util,
                         manager->nodes[i].gpu_util);
        avg_load += load;
        
        if (load > LOAD_THRESHOLD) overloaded++;
    }
    
    avg_load /= manager->active_nodes;
    
    // Check if scaling is needed
    if (overloaded > manager->active_nodes / 2) {
        // Scale up
        size_t target = min(manager->active_nodes * SCALE_FACTOR,
                          MAX_WORKERS);
        scale_up(manager, target);
    } else if (avg_load < LOAD_THRESHOLD / 2) {
        // Scale down
        size_t target = max(manager->active_nodes / SCALE_FACTOR,
                          MIN_WORKERS);
        scale_down(manager, target);
    }
}

// Scale up cluster
static void scale_up(ElasticManager* manager, size_t target) {
    size_t to_add = target - manager->active_nodes;
    
    // Find inactive nodes
    for (size_t i = 0; i < manager->num_nodes && to_add > 0; i++) {
        if (!manager->nodes[i].is_active) {
            activate_node(manager, i);
            to_add--;
        }
    }
    
    // Request new nodes if needed
    if (to_add > 0) {
        request_new_nodes(manager, to_add);
    }
    
    // Rebalance workload
    redistribute_workload(manager);
}

// Scale down cluster
static void scale_down(ElasticManager* manager, size_t target) {
    size_t to_remove = manager->active_nodes - target;
    
    // Find nodes to deactivate
    for (size_t i = 0; i < manager->num_nodes && to_remove > 0; i++) {
        if (manager->nodes[i].is_active &&
            i != manager->world_rank) {  // Don't remove self
            deactivate_node(manager, i);
            to_remove--;
        }
    }
    
    // Rebalance workload
    redistribute_workload(manager);
}

// Helper functions

static double get_gpu_utilization(void) {
    // Try NVIDIA GPU first via nvidia-smi
#ifdef __linux__
    FILE* fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1", "r");
    if (fp) {
        char buffer[32];
        if (fgets(buffer, sizeof(buffer), fp) != NULL) {
            pclose(fp);
            double util = atof(buffer) / 100.0;
            if (util >= 0.0 && util <= 1.0) {
                return util;
            }
        }
        pclose(fp);
    }

    // Try AMD GPU via rocm-smi
    fp = popen("rocm-smi --showuse 2>/dev/null | grep 'GPU use' | awk '{print $4}' | head -1", "r");
    if (fp) {
        char buffer[32];
        if (fgets(buffer, sizeof(buffer), fp) != NULL) {
            pclose(fp);
            double util = atof(buffer) / 100.0;
            if (util >= 0.0 && util <= 1.0) {
                return util;
            }
        }
        pclose(fp);
    }
#endif

#ifdef __APPLE__
    // macOS Metal GPU monitoring via IOKit
    // For now, return estimate based on system load as Metal doesn't expose direct utilization
    io_iterator_t iterator;
    kern_return_t result = IOServiceGetMatchingServices(kIOMasterPortDefault,
        IOServiceMatching("IOAccelerator"), &iterator);
    if (result == KERN_SUCCESS) {
        // GPU is available; estimate utilization from process activity
        io_object_t service;
        if ((service = IOIteratorNext(iterator)) != 0) {
            IOObjectRelease(service);
            IOObjectRelease(iterator);
            // Return system load as rough GPU estimate on macOS
            double loadavg[1];
            if (getloadavg(loadavg, 1) == 1) {
                return fmin(loadavg[0] / (double)sysconf(_SC_NPROCESSORS_ONLN), 1.0);
            }
        }
        IOObjectRelease(iterator);
    }
#endif

    return 0.0;  // No GPU or unable to determine
}

static double get_network_utilization(void) {
#ifdef __linux__
    // Read network statistics from /proc/net/dev
    static unsigned long long prev_rx = 0, prev_tx = 0;
    static time_t prev_time = 0;

    FILE* fp = fopen("/proc/net/dev", "r");
    if (!fp) return 0.0;

    char line[512];
    unsigned long long total_rx = 0, total_tx = 0;

    // Skip header lines
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);

    // Sum all interface bytes
    while (fgets(line, sizeof(line), fp)) {
        char iface[32];
        unsigned long long rx, tx;
        // Parse: iface: rx_bytes rx_packets rx_errs ...
        if (sscanf(line, " %[^:]: %llu %*u %*u %*u %*u %*u %*u %*u %llu",
                   iface, &rx, &tx) == 3) {
            // Skip loopback
            if (strcmp(iface, "lo") != 0) {
                total_rx += rx;
                total_tx += tx;
            }
        }
    }
    fclose(fp);

    time_t now = time(NULL);
    if (prev_time > 0 && now > prev_time) {
        double dt = (double)(now - prev_time);
        unsigned long long bytes_per_sec = ((total_rx - prev_rx) + (total_tx - prev_tx)) / dt;

        prev_rx = total_rx;
        prev_tx = total_tx;
        prev_time = now;

        // Estimate utilization assuming 10 Gbps max bandwidth
        const double max_bandwidth = 10.0 * 1000 * 1000 * 1000 / 8;  // bytes/sec
        return fmin((double)bytes_per_sec / max_bandwidth, 1.0);
    }

    prev_rx = total_rx;
    prev_tx = total_tx;
    prev_time = now;
    return 0.0;

#elif defined(__APPLE__)
    // macOS network statistics via netstat or nettop
    static unsigned long long prev_total = 0;
    static time_t prev_time = 0;

    FILE* fp = popen("netstat -ib | tail -n +2 | awk '{sum+=$7+$10} END {print sum}' 2>/dev/null", "r");
    if (fp) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), fp) != NULL) {
            pclose(fp);
            unsigned long long total = strtoull(buffer, NULL, 10);

            time_t now = time(NULL);
            if (prev_time > 0 && now > prev_time) {
                double dt = (double)(now - prev_time);
                double bytes_per_sec = (double)(total - prev_total) / dt;

                prev_total = total;
                prev_time = now;

                // Estimate utilization assuming 10 Gbps max
                const double max_bandwidth = 10.0 * 1000 * 1000 * 1000 / 8;
                return fmin(bytes_per_sec / max_bandwidth, 1.0);
            }

            prev_total = total;
            prev_time = now;
        } else {
            pclose(fp);
        }
    }
#endif

    return 0.0;
}

static void activate_node(ElasticManager* manager, size_t node_id) {
    manager->nodes[node_id].is_active = true;
    manager->nodes[node_id].last_heartbeat = time(NULL);
    manager->active_nodes++;

    // Notify the node to become active
    int activate_msg = 1;
    MPI_Send(&activate_msg, 1, MPI_INT, (int)node_id, 0, manager->scale_comm);
}

static void deactivate_node(ElasticManager* manager, size_t node_id) {
    // First migrate any work from this node
    int deactivate_msg = 0;
    MPI_Send(&deactivate_msg, 1, MPI_INT, (int)node_id, 0, manager->scale_comm);

    manager->nodes[node_id].is_active = false;
    manager->active_nodes--;
}

static void request_new_nodes(ElasticManager* manager, size_t count) {
    // Integration with job schedulers (SLURM, PBS, Kubernetes)

    // Try SLURM first
    const char* slurm_job_id = getenv("SLURM_JOB_ID");
    if (slurm_job_id) {
        // SLURM: Use srun to add nodes
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                 "scontrol update JobId=%s NumNodes=%zu 2>/dev/null",
                 slurm_job_id, manager->active_nodes + count);
        int result = system(cmd);
        if (result == 0) {
            // Wait for nodes to become available
            sleep(5);
            // Update MPI world
            MPI_Comm_size(manager->scale_comm, &manager->world_size);
            return;
        }
    }

    // Try PBS/Torque
    const char* pbs_job_id = getenv("PBS_JOBID");
    if (pbs_job_id) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                 "qalter -l nodes=%zu %s 2>/dev/null",
                 manager->active_nodes + count, pbs_job_id);
        int result = system(cmd);
        if (result == 0) {
            sleep(5);
            return;
        }
    }

    // Try Kubernetes
    const char* k8s_pod = getenv("KUBERNETES_SERVICE_HOST");
    if (k8s_pod) {
        // Scale up via Kubernetes API (requires kubectl or client library)
        char cmd[512];
        const char* deployment = getenv("K8S_DEPLOYMENT_NAME");
        if (deployment) {
            snprintf(cmd, sizeof(cmd),
                     "kubectl scale deployment %s --replicas=%zu 2>/dev/null",
                     deployment, manager->active_nodes + count);
            system(cmd);
            sleep(10);  // K8s scaling takes longer
        }
    }

    // If no scheduler, just log that scaling was requested
    if (manager->world_rank == 0) {
        fprintf(stderr, "[ElasticScaling] Requested %zu additional nodes (no scheduler available)\n",
                count);
    }
}

static void redistribute_workload(ElasticManager* manager) {
    // Synchronize all active nodes
    MPI_Barrier(manager->scale_comm);

    // Collect workload information from all nodes
    typedef struct {
        int rank;
        double load;
        size_t work_units;
    } WorkloadInfo;

    WorkloadInfo local_info = {
        .rank = manager->world_rank,
        .load = manager->nodes[manager->world_rank].cpu_util,
        .work_units = 0  // Will be filled by caller
    };

    // Gather all workload info at rank 0
    WorkloadInfo* all_info = NULL;
    if (manager->world_rank == 0) {
        all_info = malloc(manager->active_nodes * sizeof(WorkloadInfo));
    }

    MPI_Gather(&local_info, sizeof(WorkloadInfo), MPI_BYTE,
               all_info, sizeof(WorkloadInfo), MPI_BYTE,
               0, manager->scale_comm);

    // Rank 0 computes redistribution plan
    int* redistribution = malloc(manager->active_nodes * sizeof(int));
    memset(redistribution, 0, manager->active_nodes * sizeof(int));

    if (manager->world_rank == 0 && all_info) {
        // Calculate average load
        double total_load = 0.0;
        for (size_t i = 0; i < manager->active_nodes; i++) {
            total_load += all_info[i].load;
        }
        double avg_load = total_load / manager->active_nodes;

        // Identify overloaded and underloaded nodes
        for (size_t i = 0; i < manager->active_nodes; i++) {
            if (all_info[i].load > avg_load * 1.2) {
                // Overloaded: donate work
                redistribution[i] = -1;  // Negative = donate
            } else if (all_info[i].load < avg_load * 0.8) {
                // Underloaded: accept work
                redistribution[i] = 1;   // Positive = accept
            }
        }

        free(all_info);
    }

    // Broadcast redistribution plan
    MPI_Bcast(redistribution, manager->active_nodes, MPI_INT, 0, manager->scale_comm);

    // Execute redistribution (actual work movement depends on application)
    // This sends a signal to the training loop to rebalance data
    if (redistribution[manager->world_rank] != 0) {
        // Signal that rebalancing is needed
        manager->nodes[manager->world_rank].is_active = true;  // Trigger rebalance flag
    }

    free(redistribution);

    // Final synchronization
    MPI_Barrier(manager->scale_comm);
}

// Clean up elastic scaling
void cleanup_elastic_scaling(ElasticManager* manager) {
    if (!manager) return;
    
    manager->running = false;
    pthread_join(manager->monitor_thread, NULL);
    
    pthread_mutex_destroy(&manager->mutex);
    MPI_Comm_free(&manager->scale_comm);
    
    free(manager->nodes);
    free(manager);
}
