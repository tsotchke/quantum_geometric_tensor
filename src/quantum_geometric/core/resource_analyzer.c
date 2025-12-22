#include "quantum_geometric/core/resource_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <libproc.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <unistd.h>
#include <dirent.h>
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_ALLOCATIONS 100000
#define MAX_ALERTS 1000
#define MAX_HISTORY_ENTRIES 10000
#define HASH_TABLE_SIZE 8192

// ============================================================================
// Internal Structures
// ============================================================================

// Allocation entry in hash table
typedef struct alloc_entry {
    memory_allocation_t data;
    struct alloc_entry* next;
} alloc_entry_t;

// Resource history entry
typedef struct {
    double value;
    uint64_t timestamp_ns;
} history_entry_t;

// Resource history
typedef struct {
    history_entry_t* entries;
    size_t count;
    size_t capacity;
    bool recording;
    double interval_ms;
} resource_history_t;

// Main analyzer structure
struct resource_analyzer {
    resource_analyzer_config_t config;

    // Memory tracking
    alloc_entry_t* alloc_table[HASH_TABLE_SIZE];
    memory_stats_t memory_stats;
    size_t alloc_count;
    pthread_mutex_t alloc_mutex;

    // CPU stats
    cpu_stats_t cpu_stats;
    thread_cpu_stats_t* thread_stats;
    size_t thread_count;
    size_t thread_capacity;

    // GPU stats
    gpu_stats_t* gpu_stats;
    size_t gpu_count;

    // Disk stats
    disk_stats_t* disk_stats;
    size_t disk_count;

    // Network stats
    network_stats_t* network_stats;
    size_t network_count;

    // File descriptor stats
    fd_stats_t fd_stats;

    // Thresholds and alerts
    resource_threshold_t thresholds[RESOURCE_TYPE_COUNT];
    resource_alert_t* alerts;
    size_t alert_count;
    size_t alert_capacity;
    resource_alert_callback_t alert_callback;
    void* alert_user_data;

    // Resource history
    resource_history_t history[RESOURCE_TYPE_COUNT];

    // Intentionally long-lived allocations (not leaks)
    void** intentional_allocs;
    size_t intentional_count;
    size_t intentional_capacity;

    // Thread safety
    pthread_mutex_t mutex;
    bool thread_safe;

    // Error message
    char last_error[256];
};

// Thread-local error storage
static __thread char tls_error[256] = {0};

static void set_error(resource_analyzer_t* analyzer, const char* msg) {
    if (analyzer) {
        strncpy(analyzer->last_error, msg, sizeof(analyzer->last_error) - 1);
    }
    strncpy(tls_error, msg, sizeof(tls_error) - 1);
}

// ============================================================================
// Hash Function
// ============================================================================

static size_t hash_pointer(void* ptr) {
    uintptr_t val = (uintptr_t)ptr;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = (val >> 16) ^ val;
    return val % HASH_TABLE_SIZE;
}

// ============================================================================
// Timestamp
// ============================================================================

static uint64_t get_timestamp_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return (mach_absolute_time() * timebase.numer) / timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// ============================================================================
// Initialization
// ============================================================================

resource_analyzer_config_t resource_analyzer_default_config(void) {
    return (resource_analyzer_config_t){
        .track_allocations = true,
        .track_per_thread_cpu = true,
        .monitor_gpu = true,
        .monitor_network = true,
        .enable_leak_detection = true,
        .allocation_history_size = MAX_ALLOCATIONS,
        .sampling_interval_ms = 100.0,
        .alert_buffer_size = MAX_ALERTS
    };
}

resource_analyzer_t* resource_analyzer_create(void) {
    resource_analyzer_config_t config = resource_analyzer_default_config();
    return resource_analyzer_create_with_config(&config);
}

resource_analyzer_t* resource_analyzer_create_with_config(
    const resource_analyzer_config_t* config) {

    if (!config) return NULL;

    resource_analyzer_t* analyzer = calloc(1, sizeof(resource_analyzer_t));
    if (!analyzer) return NULL;

    analyzer->config = *config;
    analyzer->thread_safe = true;

    pthread_mutex_init(&analyzer->mutex, NULL);
    pthread_mutex_init(&analyzer->alloc_mutex, NULL);

    // Initialize alert buffer
    analyzer->alert_capacity = config->alert_buffer_size;
    analyzer->alerts = calloc(analyzer->alert_capacity, sizeof(resource_alert_t));

    // Initialize thread stats
    analyzer->thread_capacity = 64;
    analyzer->thread_stats = calloc(analyzer->thread_capacity,
                                    sizeof(thread_cpu_stats_t));

    // Initialize intentional allocations list
    analyzer->intentional_capacity = 1000;
    analyzer->intentional_allocs = calloc(analyzer->intentional_capacity,
                                          sizeof(void*));

    // Initialize history buffers
    for (int i = 0; i < RESOURCE_TYPE_COUNT; i++) {
        analyzer->history[i].capacity = MAX_HISTORY_ENTRIES;
        analyzer->history[i].entries = calloc(MAX_HISTORY_ENTRIES,
                                              sizeof(history_entry_t));
    }

    // Initialize default thresholds
    for (int i = 0; i < RESOURCE_TYPE_COUNT; i++) {
        analyzer->thresholds[i].resource = (resource_type_t)i;
        analyzer->thresholds[i].warning_threshold = 0.70;
        analyzer->thresholds[i].critical_threshold = 0.85;
        analyzer->thresholds[i].emergency_threshold = 0.95;
        analyzer->thresholds[i].enabled = true;
    }

    return analyzer;
}

void resource_analyzer_destroy(resource_analyzer_t* analyzer) {
    if (!analyzer) return;

    // Free allocation table
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry) {
            alloc_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
    }

    free(analyzer->alerts);
    free(analyzer->thread_stats);
    free(analyzer->gpu_stats);
    free(analyzer->disk_stats);
    free(analyzer->network_stats);
    free(analyzer->intentional_allocs);

    for (int i = 0; i < RESOURCE_TYPE_COUNT; i++) {
        free(analyzer->history[i].entries);
    }

    pthread_mutex_destroy(&analyzer->mutex);
    pthread_mutex_destroy(&analyzer->alloc_mutex);

    free(analyzer);
}

bool resource_analyzer_reset(resource_analyzer_t* analyzer) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Reset memory stats
    memset(&analyzer->memory_stats, 0, sizeof(memory_stats_t));

    // Clear allocation table
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry) {
            alloc_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
        analyzer->alloc_table[i] = NULL;
    }
    analyzer->alloc_count = 0;

    // Reset alerts
    analyzer->alert_count = 0;

    // Reset history
    for (int i = 0; i < RESOURCE_TYPE_COUNT; i++) {
        analyzer->history[i].count = 0;
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Memory Tracking
// ============================================================================

void resource_track_allocation(resource_analyzer_t* analyzer,
                               void* ptr,
                               size_t size,
                               const char* file,
                               int line,
                               const char* function) {
    if (!analyzer || !ptr) return;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    size_t hash = hash_pointer(ptr);

    alloc_entry_t* entry = calloc(1, sizeof(alloc_entry_t));
    if (entry) {
        entry->data.address = ptr;
        entry->data.size = size;
        entry->data.file = file;
        entry->data.line = line;
        entry->data.function = function;
        entry->data.timestamp_ns = get_timestamp_ns();
#ifdef __APPLE__
        entry->data.thread_id = pthread_mach_thread_np(pthread_self());
#else
        entry->data.thread_id = (uint32_t)pthread_self();
#endif
        entry->data.is_freed = false;

        entry->next = analyzer->alloc_table[hash];
        analyzer->alloc_table[hash] = entry;
        analyzer->alloc_count++;

        // Update stats
        analyzer->memory_stats.current_allocated += size;
        analyzer->memory_stats.total_allocated += size;
        analyzer->memory_stats.allocation_count++;

        if (analyzer->memory_stats.current_allocated >
            analyzer->memory_stats.peak_allocated) {
            analyzer->memory_stats.peak_allocated =
                analyzer->memory_stats.current_allocated;
        }
    }

    pthread_mutex_unlock(&analyzer->alloc_mutex);
}

void resource_track_free(resource_analyzer_t* analyzer, void* ptr) {
    if (!analyzer || !ptr) return;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    size_t hash = hash_pointer(ptr);
    alloc_entry_t* entry = analyzer->alloc_table[hash];
    alloc_entry_t* prev = NULL;

    while (entry) {
        if (entry->data.address == ptr && !entry->data.is_freed) {
            entry->data.is_freed = true;

            // Update stats
            analyzer->memory_stats.current_allocated -= entry->data.size;
            analyzer->memory_stats.total_freed += entry->data.size;
            analyzer->memory_stats.free_count++;

            // Remove from table (optional - can keep for leak detection)
            if (prev) {
                prev->next = entry->next;
            } else {
                analyzer->alloc_table[hash] = entry->next;
            }
            analyzer->alloc_count--;
            free(entry);
            break;
        }
        prev = entry;
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->alloc_mutex);
}

bool resource_get_memory_stats(resource_analyzer_t* analyzer,
                               memory_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    *stats = analyzer->memory_stats;

    // Get system memory info
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        stats->virtual_memory_used = info.virtual_size;
        stats->resident_memory = info.resident_size;
    }
#elif defined(__linux__)
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        unsigned long vsize, rss;
        if (fscanf(f, "%lu %lu", &vsize, &rss) == 2) {
            long page_size = sysconf(_SC_PAGESIZE);
            stats->virtual_memory_used = vsize * page_size;
            stats->resident_memory = rss * page_size;
        }
        fclose(f);
    }
#endif

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return true;
}

bool resource_get_allocations(resource_analyzer_t* analyzer,
                              memory_allocation_t** allocations,
                              size_t* count) {
    if (!analyzer || !allocations || !count) return false;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    *count = analyzer->alloc_count;
    if (*count == 0) {
        *allocations = NULL;
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return true;
    }

    *allocations = calloc(*count, sizeof(memory_allocation_t));
    if (!*allocations) {
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE && idx < *count; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry && idx < *count) {
            (*allocations)[idx++] = entry->data;
            entry = entry->next;
        }
    }

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return true;
}

bool resource_find_allocation(resource_analyzer_t* analyzer,
                              void* address,
                              memory_allocation_t* allocation) {
    if (!analyzer || !address || !allocation) return false;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    size_t hash = hash_pointer(address);
    alloc_entry_t* entry = analyzer->alloc_table[hash];

    bool found = false;
    while (entry) {
        if (entry->data.address == address) {
            *allocation = entry->data;
            found = true;
            break;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return found;
}

double resource_calculate_fragmentation(resource_analyzer_t* analyzer) {
    if (!analyzer) return 0.0;

    // Simplified fragmentation estimation
    // Real implementation would analyze actual heap structure
    pthread_mutex_lock(&analyzer->alloc_mutex);

    double frag = 0.0;
    if (analyzer->memory_stats.allocation_count >
        analyzer->memory_stats.free_count) {
        size_t live_allocs = analyzer->memory_stats.allocation_count -
                            analyzer->memory_stats.free_count;
        if (live_allocs > 0 && analyzer->memory_stats.peak_allocated > 0) {
            // Estimate fragmentation as ratio of live allocations to peak
            frag = 1.0 - ((double)analyzer->memory_stats.current_allocated /
                         (double)analyzer->memory_stats.peak_allocated);
            if (frag < 0) frag = 0;
            if (frag > 1) frag = 1;
        }
    }

    analyzer->memory_stats.fragmentation_ratio = frag;

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return frag;
}

// ============================================================================
// CPU Monitoring
// ============================================================================

bool resource_get_cpu_stats(resource_analyzer_t* analyzer,
                            cpu_stats_t* stats) {
    if (!analyzer || !stats) return false;

    memset(stats, 0, sizeof(cpu_stats_t));

#ifdef __APPLE__
    // Get CPU load
    host_cpu_load_info_data_t cpu_info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        (host_info_t)&cpu_info, &count) == KERN_SUCCESS) {
        unsigned long total = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total += cpu_info.cpu_ticks[i];
        }

        if (total > 0) {
            stats->user_percent = 100.0 * cpu_info.cpu_ticks[CPU_STATE_USER] / total;
            stats->system_percent = 100.0 * cpu_info.cpu_ticks[CPU_STATE_SYSTEM] / total;
            stats->idle_percent = 100.0 * cpu_info.cpu_ticks[CPU_STATE_IDLE] / total;
            stats->overall_usage_percent = 100.0 - stats->idle_percent;
        }
    }

    // Get core count
    int cores;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &size, NULL, 0) == 0) {
        stats->num_cores = cores;
    }

    // Get load averages
    double load[3];
    if (getloadavg(load, 3) == 3) {
        stats->load_average_1min = load[0];
        stats->load_average_5min = load[1];
        stats->load_average_15min = load[2];
    }

#elif defined(__linux__)
    FILE* f = fopen("/proc/stat", "r");
    if (f) {
        char line[256];
        if (fgets(line, sizeof(line), f)) {
            unsigned long long user, nice, system, idle, iowait;
            if (sscanf(line, "cpu %llu %llu %llu %llu %llu",
                       &user, &nice, &system, &idle, &iowait) == 5) {
                unsigned long long total = user + nice + system + idle + iowait;
                if (total > 0) {
                    stats->user_percent = 100.0 * (user + nice) / total;
                    stats->system_percent = 100.0 * system / total;
                    stats->idle_percent = 100.0 * idle / total;
                    stats->iowait_percent = 100.0 * iowait / total;
                    stats->overall_usage_percent = 100.0 - stats->idle_percent;
                }
            }
        }
        fclose(f);
    }

    stats->num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    // Load averages
    double load[3];
    if (getloadavg(load, 3) == 3) {
        stats->load_average_1min = load[0];
        stats->load_average_5min = load[1];
        stats->load_average_15min = load[2];
    }
#endif

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->cpu_stats = *stats;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool resource_get_thread_cpu_stats(resource_analyzer_t* analyzer,
                                   thread_cpu_stats_t** stats,
                                   size_t* count) {
    if (!analyzer || !stats || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->thread_count;
    if (*count == 0) {
        *stats = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *stats = calloc(*count, sizeof(thread_cpu_stats_t));
    if (*stats) {
        memcpy(*stats, analyzer->thread_stats,
               *count * sizeof(thread_cpu_stats_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *stats != NULL;
}

bool resource_get_thread_cpu_by_id(resource_analyzer_t* analyzer,
                                   uint32_t thread_id,
                                   thread_cpu_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->thread_count; i++) {
        if (analyzer->thread_stats[i].thread_id == thread_id) {
            *stats = analyzer->thread_stats[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool resource_set_cpu_affinity(int cpu_core) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
#else
    (void)cpu_core;
    return false;  // Not supported on macOS
#endif
}

size_t resource_get_cpu_core_count(void) {
#ifdef __APPLE__
    int cores;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &size, NULL, 0) == 0) {
        return cores;
    }
    return 1;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

// ============================================================================
// GPU Monitoring
// ============================================================================

size_t resource_get_gpu_count(resource_analyzer_t* analyzer) {
    if (!analyzer) return 0;
    return analyzer->gpu_count;
}

bool resource_get_gpu_stats(resource_analyzer_t* analyzer,
                            int device_id,
                            gpu_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->gpu_count; i++) {
        if (analyzer->gpu_stats[i].device_id == device_id) {
            *stats = analyzer->gpu_stats[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool resource_get_all_gpu_stats(resource_analyzer_t* analyzer,
                                gpu_stats_t** stats,
                                size_t* count) {
    if (!analyzer || !stats || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->gpu_count;
    if (*count == 0) {
        *stats = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *stats = calloc(*count, sizeof(gpu_stats_t));
    if (*stats) {
        memcpy(*stats, analyzer->gpu_stats, *count * sizeof(gpu_stats_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *stats != NULL;
}

bool resource_gpu_available(void) {
#ifdef __APPLE__
    // Check for Metal support
    return true;  // All modern Macs have Metal
#elif defined(__linux__)
    // Check for NVIDIA GPU
    FILE* f = popen("nvidia-smi -L 2>/dev/null | wc -l", "r");
    if (f) {
        int count = 0;
        if (fscanf(f, "%d", &count) == 1 && count > 0) {
            pclose(f);
            return true;
        }
        pclose(f);
    }
    return false;
#else
    return false;
#endif
}

// ============================================================================
// Disk I/O Monitoring
// ============================================================================

bool resource_get_disk_stats(resource_analyzer_t* analyzer,
                             const char* device,
                             disk_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->disk_count; i++) {
        if (strcmp(analyzer->disk_stats[i].device_name, device) == 0) {
            *stats = analyzer->disk_stats[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool resource_get_all_disk_stats(resource_analyzer_t* analyzer,
                                 disk_stats_t** stats,
                                 size_t* count) {
    if (!analyzer || !stats || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->disk_count;
    if (*count == 0) {
        *stats = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *stats = calloc(*count, sizeof(disk_stats_t));
    if (*stats) {
        memcpy(*stats, analyzer->disk_stats, *count * sizeof(disk_stats_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *stats != NULL;
}

// ============================================================================
// Network Monitoring
// ============================================================================

bool resource_get_network_stats(resource_analyzer_t* analyzer,
                                const char* interface,
                                network_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    bool found = false;
    for (size_t i = 0; i < analyzer->network_count; i++) {
        if (strcmp(analyzer->network_stats[i].interface_name, interface) == 0) {
            *stats = analyzer->network_stats[i];
            found = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool resource_get_all_network_stats(resource_analyzer_t* analyzer,
                                    network_stats_t** stats,
                                    size_t* count) {
    if (!analyzer || !stats || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->network_count;
    if (*count == 0) {
        *stats = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *stats = calloc(*count, sizeof(network_stats_t));
    if (*stats) {
        memcpy(*stats, analyzer->network_stats, *count * sizeof(network_stats_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *stats != NULL;
}

// ============================================================================
// File Descriptor Tracking
// ============================================================================

bool resource_get_fd_stats(resource_analyzer_t* analyzer,
                           fd_stats_t* stats) {
    if (!analyzer || !stats) return false;

    memset(stats, 0, sizeof(fd_stats_t));

#ifdef __APPLE__
    // Count open file descriptors
    int max_fd = getdtablesize();
    stats->max_fds = max_fd;

    for (int fd = 0; fd < max_fd; fd++) {
        struct stat st;
        if (fstat(fd, &st) == 0) {
            stats->open_fds++;
            if (S_ISSOCK(st.st_mode)) {
                stats->socket_count++;
            } else if (S_ISFIFO(st.st_mode)) {
                stats->pipe_count++;
            } else if (S_ISREG(st.st_mode)) {
                stats->file_count++;
            }
        }
    }
#elif defined(__linux__)
    // Get max FDs
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        stats->max_fds = rl.rlim_cur;
    }

    // Count open FDs
    DIR* dir = opendir("/proc/self/fd");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] != '.') {
                stats->open_fds++;
            }
        }
        closedir(dir);
    }
#endif

    if (stats->max_fds > 0) {
        stats->usage_percent = 100.0 * stats->open_fds / stats->max_fds;
    }

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->fd_stats = *stats;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool resource_list_open_fds(resource_analyzer_t* analyzer,
                            int** fds,
                            size_t* count) {
    if (!analyzer || !fds || !count) return false;

    *count = 0;
    *fds = NULL;

#ifdef __linux__
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return false;

    // Count first
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] != '.') {
            (*count)++;
        }
    }

    if (*count == 0) {
        closedir(dir);
        return true;
    }

    *fds = calloc(*count, sizeof(int));
    if (!*fds) {
        closedir(dir);
        return false;
    }

    rewinddir(dir);
    size_t idx = 0;
    while ((entry = readdir(dir)) != NULL && idx < *count) {
        if (entry->d_name[0] != '.') {
            (*fds)[idx++] = atoi(entry->d_name);
        }
    }

    closedir(dir);
#endif

    return true;
}

// ============================================================================
// Resource Leak Detection
// ============================================================================

bool resource_detect_leaks(resource_analyzer_t* analyzer,
                           potential_leak_t** leaks,
                           size_t* count) {
    if (!analyzer || !leaks || !count) return false;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    uint64_t now = get_timestamp_ns();
    size_t potential_count = 0;

    // Count potential leaks (old, unfreed allocations)
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry) {
            if (!entry->data.is_freed) {
                uint64_t age_ms = (now - entry->data.timestamp_ns) / 1000000;
                if (age_ms > 60000) {  // Older than 1 minute
                    // Check if intentionally long-lived
                    bool intentional = false;
                    for (size_t j = 0; j < analyzer->intentional_count; j++) {
                        if (analyzer->intentional_allocs[j] == entry->data.address) {
                            intentional = true;
                            break;
                        }
                    }
                    if (!intentional) {
                        potential_count++;
                    }
                }
            }
            entry = entry->next;
        }
    }

    *count = potential_count;
    if (potential_count == 0) {
        *leaks = NULL;
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return true;
    }

    *leaks = calloc(potential_count, sizeof(potential_leak_t));
    if (!*leaks) {
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE && idx < potential_count; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry && idx < potential_count) {
            if (!entry->data.is_freed) {
                uint64_t age_ms = (now - entry->data.timestamp_ns) / 1000000;
                if (age_ms > 60000) {
                    bool intentional = false;
                    for (size_t j = 0; j < analyzer->intentional_count; j++) {
                        if (analyzer->intentional_allocs[j] == entry->data.address) {
                            intentional = true;
                            break;
                        }
                    }
                    if (!intentional) {
                        (*leaks)[idx].address = entry->data.address;
                        (*leaks)[idx].size = entry->data.size;
                        (*leaks)[idx].allocation_site = entry->data.file;
                        (*leaks)[idx].age_ms = age_ms;
                        (*leaks)[idx].leak_probability =
                            fmin(1.0, age_ms / 300000.0);  // Max at 5 min
                        (*leaks)[idx].suggested_fix =
                            "Review allocation and ensure proper cleanup";
                        idx++;
                    }
                }
            }
            entry = entry->next;
        }
    }

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return true;
}

bool resource_check_unfreed_allocations(resource_analyzer_t* analyzer,
                                        memory_allocation_t** unfreed,
                                        size_t* count) {
    if (!analyzer || !unfreed || !count) return false;

    pthread_mutex_lock(&analyzer->alloc_mutex);

    *count = analyzer->alloc_count;
    if (*count == 0) {
        *unfreed = NULL;
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return true;
    }

    *unfreed = calloc(*count, sizeof(memory_allocation_t));
    if (!*unfreed) {
        pthread_mutex_unlock(&analyzer->alloc_mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE && idx < *count; i++) {
        alloc_entry_t* entry = analyzer->alloc_table[i];
        while (entry && idx < *count) {
            if (!entry->data.is_freed) {
                (*unfreed)[idx++] = entry->data;
            }
            entry = entry->next;
        }
    }
    *count = idx;

    pthread_mutex_unlock(&analyzer->alloc_mutex);

    return true;
}

void resource_mark_intentional(resource_analyzer_t* analyzer, void* ptr) {
    if (!analyzer || !ptr) return;

    pthread_mutex_lock(&analyzer->mutex);

    if (analyzer->intentional_count < analyzer->intentional_capacity) {
        analyzer->intentional_allocs[analyzer->intentional_count++] = ptr;
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

bool resource_get_leak_summary(resource_analyzer_t* analyzer,
                               size_t* total_leaked_bytes,
                               size_t* leak_count) {
    if (!analyzer || !total_leaked_bytes || !leak_count) return false;

    potential_leak_t* leaks;
    size_t count;

    if (!resource_detect_leaks(analyzer, &leaks, &count)) {
        return false;
    }

    *leak_count = count;
    *total_leaked_bytes = 0;

    for (size_t i = 0; i < count; i++) {
        *total_leaked_bytes += leaks[i].size;
    }

    resource_free_leak_results(leaks, count);

    return true;
}

void resource_free_leak_results(potential_leak_t* leaks, size_t count) {
    (void)count;
    free(leaks);
}

// ============================================================================
// Threshold-Based Alerting
// ============================================================================

bool resource_set_threshold(resource_analyzer_t* analyzer,
                            const resource_threshold_t* threshold) {
    if (!analyzer || !threshold ||
        threshold->resource >= RESOURCE_TYPE_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->thresholds[threshold->resource] = *threshold;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool resource_get_threshold(resource_analyzer_t* analyzer,
                            resource_type_t resource,
                            resource_threshold_t* threshold) {
    if (!analyzer || !threshold || resource >= RESOURCE_TYPE_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);
    *threshold = analyzer->thresholds[resource];
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

static void add_alert(resource_analyzer_t* analyzer,
                      resource_type_t resource,
                      alert_severity_t severity,
                      double current_value,
                      double threshold_value,
                      const char* message) {
    if (analyzer->alert_count >= analyzer->alert_capacity) return;

    resource_alert_t* alert = &analyzer->alerts[analyzer->alert_count++];
    alert->resource = resource;
    alert->severity = severity;
    alert->current_value = current_value;
    alert->threshold_value = threshold_value;
    alert->timestamp_ns = get_timestamp_ns();
    alert->acknowledged = false;
    strncpy(alert->message, message, sizeof(alert->message) - 1);

    if (analyzer->alert_callback) {
        analyzer->alert_callback(alert, analyzer->alert_user_data);
    }
}

size_t resource_check_thresholds(resource_analyzer_t* analyzer) {
    if (!analyzer) return 0;

    pthread_mutex_lock(&analyzer->mutex);

    size_t alerts_generated = 0;
    double memory_usage = 0;
    double cpu_usage = 0;

    // Check memory usage
    memory_stats_t mem_stats;
    if (resource_get_memory_stats(analyzer, &mem_stats)) {
        size_t total = resource_get_total_memory();
        if (total > 0) {
            memory_usage = (double)mem_stats.resident_memory / (double)total;
        }
    }

    resource_threshold_t* mem_thresh = &analyzer->thresholds[RESOURCE_MEMORY];
    if (mem_thresh->enabled && memory_usage > 0) {
        if (memory_usage >= mem_thresh->emergency_threshold) {
            add_alert(analyzer, RESOURCE_MEMORY, ALERT_EMERGENCY,
                     memory_usage, mem_thresh->emergency_threshold,
                     "Memory usage is at emergency level");
            alerts_generated++;
        } else if (memory_usage >= mem_thresh->critical_threshold) {
            add_alert(analyzer, RESOURCE_MEMORY, ALERT_CRITICAL,
                     memory_usage, mem_thresh->critical_threshold,
                     "Memory usage is critical");
            alerts_generated++;
        } else if (memory_usage >= mem_thresh->warning_threshold) {
            add_alert(analyzer, RESOURCE_MEMORY, ALERT_WARNING,
                     memory_usage, mem_thresh->warning_threshold,
                     "Memory usage is elevated");
            alerts_generated++;
        }
    }

    // Check CPU usage
    cpu_stats_t cpu_stats;
    if (resource_get_cpu_stats(analyzer, &cpu_stats)) {
        cpu_usage = cpu_stats.overall_usage_percent / 100.0;
    }

    resource_threshold_t* cpu_thresh = &analyzer->thresholds[RESOURCE_CPU];
    if (cpu_thresh->enabled && cpu_usage > 0) {
        if (cpu_usage >= cpu_thresh->emergency_threshold) {
            add_alert(analyzer, RESOURCE_CPU, ALERT_EMERGENCY,
                     cpu_usage, cpu_thresh->emergency_threshold,
                     "CPU usage is at emergency level");
            alerts_generated++;
        } else if (cpu_usage >= cpu_thresh->critical_threshold) {
            add_alert(analyzer, RESOURCE_CPU, ALERT_CRITICAL,
                     cpu_usage, cpu_thresh->critical_threshold,
                     "CPU usage is critical");
            alerts_generated++;
        } else if (cpu_usage >= cpu_thresh->warning_threshold) {
            add_alert(analyzer, RESOURCE_CPU, ALERT_WARNING,
                     cpu_usage, cpu_thresh->warning_threshold,
                     "CPU usage is elevated");
            alerts_generated++;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return alerts_generated;
}

bool resource_get_alerts(resource_analyzer_t* analyzer,
                         resource_alert_t** alerts,
                         size_t* count) {
    if (!analyzer || !alerts || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->alert_count;
    if (*count == 0) {
        *alerts = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *alerts = calloc(*count, sizeof(resource_alert_t));
    if (*alerts) {
        memcpy(*alerts, analyzer->alerts, *count * sizeof(resource_alert_t));
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return *alerts != NULL;
}

void resource_clear_alerts(resource_analyzer_t* analyzer) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->alert_count = 0;
    pthread_mutex_unlock(&analyzer->mutex);
}

void resource_acknowledge_alert(resource_analyzer_t* analyzer,
                                uint64_t alert_timestamp) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < analyzer->alert_count; i++) {
        if (analyzer->alerts[i].timestamp_ns == alert_timestamp) {
            analyzer->alerts[i].acknowledged = true;
            break;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

void resource_set_alert_callback(resource_analyzer_t* analyzer,
                                 resource_alert_callback_t callback,
                                 void* user_data) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->alert_callback = callback;
    analyzer->alert_user_data = user_data;
    pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Resource History and Trends
// ============================================================================

bool resource_start_recording(resource_analyzer_t* analyzer,
                              resource_type_t resource,
                              double interval_ms) {
    if (!analyzer || resource >= RESOURCE_TYPE_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->history[resource].recording = true;
    analyzer->history[resource].interval_ms = interval_ms;
    analyzer->history[resource].count = 0;
    pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

void resource_stop_recording(resource_analyzer_t* analyzer,
                             resource_type_t resource) {
    if (!analyzer || resource >= RESOURCE_TYPE_COUNT) return;

    pthread_mutex_lock(&analyzer->mutex);
    analyzer->history[resource].recording = false;
    pthread_mutex_unlock(&analyzer->mutex);
}

bool resource_get_history(resource_analyzer_t* analyzer,
                          resource_type_t resource,
                          double** values,
                          uint64_t** timestamps,
                          size_t* count) {
    if (!analyzer || !values || !timestamps || !count ||
        resource >= RESOURCE_TYPE_COUNT) return false;

    pthread_mutex_lock(&analyzer->mutex);

    resource_history_t* hist = &analyzer->history[resource];
    *count = hist->count;

    if (*count == 0) {
        *values = NULL;
        *timestamps = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *values = calloc(*count, sizeof(double));
    *timestamps = calloc(*count, sizeof(uint64_t));

    if (*values && *timestamps) {
        for (size_t i = 0; i < *count; i++) {
            (*values)[i] = hist->entries[i].value;
            (*timestamps)[i] = hist->entries[i].timestamp_ns;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return (*values != NULL && *timestamps != NULL);
}

double resource_get_trend(resource_analyzer_t* analyzer,
                          resource_type_t resource) {
    if (!analyzer || resource >= RESOURCE_TYPE_COUNT) return 0.0;

    pthread_mutex_lock(&analyzer->mutex);

    resource_history_t* hist = &analyzer->history[resource];
    double trend = 0.0;

    if (hist->count >= 2) {
        // Simple linear regression to get slope
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

        for (size_t i = 0; i < hist->count; i++) {
            double x = (double)i;
            double y = hist->entries[i].value;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        double n = (double)hist->count;
        double denom = n * sum_xx - sum_x * sum_x;

        if (fabs(denom) > 1e-10) {
            trend = (n * sum_xy - sum_x * sum_y) / denom;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);

    return trend;
}

// ============================================================================
// Export and Reporting
// ============================================================================

char* resource_export_json(resource_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    size_t buffer_size = 16384;
    char* json = malloc(buffer_size);
    if (!json) return NULL;

    memory_stats_t mem_stats;
    cpu_stats_t cpu_stats;
    fd_stats_t fd_stats;

    resource_get_memory_stats(analyzer, &mem_stats);
    resource_get_cpu_stats(analyzer, &cpu_stats);
    resource_get_fd_stats(analyzer, &fd_stats);

    snprintf(json, buffer_size,
        "{\n"
        "  \"memory\": {\n"
        "    \"current_allocated\": %zu,\n"
        "    \"peak_allocated\": %zu,\n"
        "    \"total_allocated\": %zu,\n"
        "    \"total_freed\": %zu,\n"
        "    \"allocation_count\": %llu,\n"
        "    \"free_count\": %llu,\n"
        "    \"fragmentation_ratio\": %.4f,\n"
        "    \"resident_memory\": %zu,\n"
        "    \"virtual_memory\": %zu\n"
        "  },\n"
        "  \"cpu\": {\n"
        "    \"overall_usage_percent\": %.2f,\n"
        "    \"user_percent\": %.2f,\n"
        "    \"system_percent\": %.2f,\n"
        "    \"idle_percent\": %.2f,\n"
        "    \"num_cores\": %zu,\n"
        "    \"load_average_1min\": %.2f,\n"
        "    \"load_average_5min\": %.2f,\n"
        "    \"load_average_15min\": %.2f\n"
        "  },\n"
        "  \"file_descriptors\": {\n"
        "    \"open\": %u,\n"
        "    \"max\": %u,\n"
        "    \"usage_percent\": %.2f\n"
        "  }\n"
        "}",
        mem_stats.current_allocated,
        mem_stats.peak_allocated,
        mem_stats.total_allocated,
        mem_stats.total_freed,
        (unsigned long long)mem_stats.allocation_count,
        (unsigned long long)mem_stats.free_count,
        mem_stats.fragmentation_ratio,
        mem_stats.resident_memory,
        mem_stats.virtual_memory_used,
        cpu_stats.overall_usage_percent,
        cpu_stats.user_percent,
        cpu_stats.system_percent,
        cpu_stats.idle_percent,
        cpu_stats.num_cores,
        cpu_stats.load_average_1min,
        cpu_stats.load_average_5min,
        cpu_stats.load_average_15min,
        fd_stats.open_fds,
        fd_stats.max_fds,
        fd_stats.usage_percent);

    return json;
}

bool resource_export_to_file(resource_analyzer_t* analyzer,
                             const char* filename) {
    if (!analyzer || !filename) return false;

    char* json = resource_export_json(analyzer);
    if (!json) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(json);
        return false;
    }

    fputs(json, f);
    fclose(f);
    free(json);

    return true;
}

char* resource_generate_report(resource_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    size_t buffer_size = 8192;
    char* report = malloc(buffer_size);
    if (!report) return NULL;

    memory_stats_t mem_stats;
    cpu_stats_t cpu_stats;
    fd_stats_t fd_stats;

    resource_get_memory_stats(analyzer, &mem_stats);
    resource_get_cpu_stats(analyzer, &cpu_stats);
    resource_get_fd_stats(analyzer, &fd_stats);

    snprintf(report, buffer_size,
        "=== Resource Usage Report ===\n\n"
        "Memory:\n"
        "  Current: %s\n"
        "  Peak: %s\n"
        "  Allocations: %llu\n"
        "  Frees: %llu\n"
        "  Fragmentation: %.1f%%\n\n"
        "CPU:\n"
        "  Overall Usage: %.1f%%\n"
        "  User: %.1f%%\n"
        "  System: %.1f%%\n"
        "  Cores: %zu\n"
        "  Load (1/5/15): %.2f / %.2f / %.2f\n\n"
        "File Descriptors:\n"
        "  Open: %u / %u (%.1f%%)\n",
        resource_format_bytes(mem_stats.current_allocated),
        resource_format_bytes(mem_stats.peak_allocated),
        (unsigned long long)mem_stats.allocation_count,
        (unsigned long long)mem_stats.free_count,
        mem_stats.fragmentation_ratio * 100.0,
        cpu_stats.overall_usage_percent,
        cpu_stats.user_percent,
        cpu_stats.system_percent,
        cpu_stats.num_cores,
        cpu_stats.load_average_1min,
        cpu_stats.load_average_5min,
        cpu_stats.load_average_15min,
        fd_stats.open_fds,
        fd_stats.max_fds,
        fd_stats.usage_percent);

    return report;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* resource_type_name(resource_type_t type) {
    switch (type) {
        case RESOURCE_MEMORY: return "memory";
        case RESOURCE_CPU: return "cpu";
        case RESOURCE_GPU: return "gpu";
        case RESOURCE_DISK: return "disk";
        case RESOURCE_NETWORK: return "network";
        case RESOURCE_FILE_DESCRIPTORS: return "file_descriptors";
        case RESOURCE_THREADS: return "threads";
        default: return "unknown";
    }
}

const char* resource_alert_severity_name(alert_severity_t severity) {
    switch (severity) {
        case ALERT_INFO: return "info";
        case ALERT_WARNING: return "warning";
        case ALERT_CRITICAL: return "critical";
        case ALERT_EMERGENCY: return "emergency";
        default: return "unknown";
    }
}

char* resource_format_bytes(size_t bytes) {
    char* buffer = malloc(64);
    if (!buffer) return NULL;

    if (bytes < 1024) {
        snprintf(buffer, 64, "%zu B", bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buffer, 64, "%.2f KB", bytes / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buffer, 64, "%.2f MB", bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buffer, 64, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }

    return buffer;
}

const char* resource_get_last_error(void) {
    return tls_error[0] ? tls_error : "No error";
}

size_t resource_get_page_size(void) {
#ifdef _SC_PAGESIZE
    return sysconf(_SC_PAGESIZE);
#else
    return 4096;
#endif
}

size_t resource_get_total_memory(void) {
#ifdef __APPLE__
    int64_t mem;
    size_t size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, NULL, 0) == 0) {
        return (size_t)mem;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.totalram * si.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

size_t resource_get_available_memory(void) {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_size_t page_size;
    host_page_size(host, &page_size);

    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        return (vm_stats.free_count + vm_stats.inactive_count) * page_size;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.freeram * si.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}
