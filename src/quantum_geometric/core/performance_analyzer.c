#include "quantum_geometric/core/performance_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <mach/thread_act.h>
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#endif

// ============================================================================
// Internal Structures
// ============================================================================

// Thread-local storage key for error messages
static pthread_key_t error_key;
static pthread_once_t error_key_once = PTHREAD_ONCE_INIT;

// Profiling region internal structure
typedef struct profiling_region_internal {
    profiling_region_t data;
    uint64_t* time_samples;
    size_t sample_count;
    size_t sample_capacity;
    uint64_t current_start_ns;
    uint64_t current_cycles_start;
    bool is_active;
    struct profiling_region_internal* next;
} profiling_region_internal_t;

// Sample buffer
typedef struct {
    perf_sample_t* samples;
    size_t count;
    size_t capacity;
} sample_buffer_t;

// Flame graph node internal
typedef struct flame_node {
    char* function_name;
    uint64_t sample_count;
    uint64_t self_samples;
    struct flame_node** children;
    size_t num_children;
    size_t children_capacity;
    struct flame_node* parent;
} flame_node_t;

// Cycle region tracking
typedef struct cycle_region {
    char name[64];
    uint64_t start_cycles;
    bool is_active;
    struct cycle_region* next;
} cycle_region_t;

// Main analyzer structure
struct performance_analyzer {
    perf_analyzer_config_t config;

    // Profiling regions hash table
    profiling_region_internal_t* regions[256];
    size_t region_count;

    // Sample buffers per metric type
    sample_buffer_t sample_buffers[METRIC_TYPE_COUNT];

    // Cache monitoring state
    bool cache_monitoring_active;
#ifdef __linux__
    int perf_fds[CACHE_LEVEL_COUNT][2];  // [level][references/misses]
#endif
    cache_stats_t cache_stats[CACHE_LEVEL_COUNT];

    // Bandwidth monitoring
    bool bandwidth_monitoring_active;
    uint64_t bandwidth_start_time;
    uint64_t bandwidth_bytes_read;
    uint64_t bandwidth_bytes_written;

    // FLOPS counting
    bool flops_counting_active;
    uint64_t flops_start_time;
    uint64_t total_flops;

    // Flame graph
    bool flamegraph_active;
    flame_node_t* flamegraph_root;
    flame_node_t* current_flame_node;

    // Cycle regions
    cycle_region_t* cycle_regions;

    // Region stack for nested regions
    profiling_region_internal_t* region_stack[64];
    size_t region_stack_depth;

    // Thread safety
    pthread_mutex_t mutex;
    bool thread_safe;

    // Timing calibration
    uint64_t timer_overhead_ns;
    double ns_per_tick;
};

// ============================================================================
// Error Handling
// ============================================================================

static void init_error_key(void) {
    pthread_key_create(&error_key, free);
}

static void set_error(const char* message) {
    pthread_once(&error_key_once, init_error_key);
    char* error = pthread_getspecific(error_key);
    if (!error) {
        error = malloc(512);
        pthread_setspecific(error_key, error);
    }
    if (error) {
        strncpy(error, message, 511);
        error[511] = '\0';
    }
}

const char* perf_get_last_error(void) {
    pthread_once(&error_key_once, init_error_key);
    char* error = pthread_getspecific(error_key);
    return error ? error : "No error";
}

// ============================================================================
// High-Resolution Timing Implementation
// ============================================================================

#ifdef __APPLE__
static mach_timebase_info_data_t timebase_info = {0, 0};

static void init_timebase(void) {
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
}

uint64_t perf_get_timestamp_ns(void) {
    init_timebase();
    uint64_t ticks = mach_absolute_time();
    return (ticks * timebase_info.numer) / timebase_info.denom;
}

uint64_t perf_get_cpu_frequency(void) {
    uint64_t freq = 0;
    size_t size = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) != 0) {
        // Fallback: estimate from calibration
        freq = 2400000000ULL;  // 2.4 GHz default
    }
    return freq;
}

#elif defined(__linux__)
uint64_t perf_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t perf_get_cpu_frequency(void) {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (f) {
        unsigned long freq_khz;
        if (fscanf(f, "%lu", &freq_khz) == 1) {
            fclose(f);
            return freq_khz * 1000ULL;
        }
        fclose(f);
    }
    // Fallback
    return 2400000000ULL;
}
#else
uint64_t perf_get_timestamp_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t perf_get_cpu_frequency(void) {
    return 2400000000ULL;
}
#endif

// ============================================================================
// CPU Cycle Counter
// ============================================================================

uint64_t perf_read_cpu_cycles(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile ("mrs %0, cntvct_el0" : "=r" (val));
    return val;
#else
    // Fallback to timestamp
    return perf_get_timestamp_ns();
#endif
}

// ============================================================================
// Hash Function for Region Names
// ============================================================================

static uint8_t hash_string(const char* str) {
    uint32_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + (unsigned char)*str++;
    }
    return (uint8_t)(hash & 0xFF);
}

// ============================================================================
// Initialization
// ============================================================================

perf_analyzer_config_t perf_analyzer_default_config(void) {
    return (perf_analyzer_config_t){
        .default_precision = TIMER_PRECISION_NANOSECOND,
        .max_samples = 100000,
        .max_regions = 1000,
        .enable_cpu_cycles = true,
        .enable_cache_counters = false,  // Requires elevated privileges
        .enable_memory_bandwidth = true,
        .enable_flops_counting = true,
        .enable_flamegraph = false,
        .sample_buffer_size = 10000,
        .sample_rate_hz = 1000.0,
        .thread_safe = true
    };
}

performance_analyzer_t* perf_analyzer_create(void) {
    perf_analyzer_config_t config = perf_analyzer_default_config();
    return perf_analyzer_create_with_config(&config);
}

performance_analyzer_t* perf_analyzer_create_with_config(
    const perf_analyzer_config_t* config) {

    if (!config) {
        set_error("NULL configuration provided");
        return NULL;
    }

    performance_analyzer_t* analyzer = calloc(1, sizeof(performance_analyzer_t));
    if (!analyzer) {
        set_error("Failed to allocate analyzer");
        return NULL;
    }

    analyzer->config = *config;
    analyzer->thread_safe = config->thread_safe;

    if (config->thread_safe) {
        pthread_mutex_init(&analyzer->mutex, NULL);
    }

    // Initialize sample buffers
    for (int i = 0; i < METRIC_TYPE_COUNT; i++) {
        analyzer->sample_buffers[i].capacity = config->sample_buffer_size;
        analyzer->sample_buffers[i].samples = calloc(
            config->sample_buffer_size, sizeof(perf_sample_t));
        if (!analyzer->sample_buffers[i].samples) {
            perf_analyzer_destroy(analyzer);
            set_error("Failed to allocate sample buffers");
            return NULL;
        }
    }

    // Calibrate timer overhead
    uint64_t start = perf_get_timestamp_ns();
    for (int i = 0; i < 1000; i++) {
        perf_get_timestamp_ns();
    }
    uint64_t end = perf_get_timestamp_ns();
    analyzer->timer_overhead_ns = (end - start) / 1000;

    // Calculate nanoseconds per tick for cycle counter
    uint64_t cycles_start = perf_read_cpu_cycles();
    uint64_t time_start = perf_get_timestamp_ns();

    // Spin for a bit
    volatile int x = 0;
    for (int i = 0; i < 1000000; i++) x++;

    uint64_t cycles_end = perf_read_cpu_cycles();
    uint64_t time_end = perf_get_timestamp_ns();

    uint64_t cycles = cycles_end - cycles_start;
    uint64_t nanos = time_end - time_start;

    if (cycles > 0) {
        analyzer->ns_per_tick = (double)nanos / (double)cycles;
    } else {
        analyzer->ns_per_tick = 1.0 / 2.4;  // Assume 2.4 GHz
    }

#ifdef __linux__
    // Initialize perf_event file descriptors
    for (int i = 0; i < CACHE_LEVEL_COUNT; i++) {
        analyzer->perf_fds[i][0] = -1;
        analyzer->perf_fds[i][1] = -1;
    }
#endif

    return analyzer;
}

void perf_analyzer_destroy(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    // Free profiling regions
    for (int i = 0; i < 256; i++) {
        profiling_region_internal_t* region = analyzer->regions[i];
        while (region) {
            profiling_region_internal_t* next = region->next;
            free(region->time_samples);
            free(region);
            region = next;
        }
    }

    // Free sample buffers
    for (int i = 0; i < METRIC_TYPE_COUNT; i++) {
        free(analyzer->sample_buffers[i].samples);
    }

    // Free cycle regions
    cycle_region_t* cr = analyzer->cycle_regions;
    while (cr) {
        cycle_region_t* next = cr->next;
        free(cr);
        cr = next;
    }

    // Free flame graph
    if (analyzer->flamegraph_root) {
        perf_free_flamegraph_data((call_stack_entry_t*)analyzer->flamegraph_root);
    }

#ifdef __linux__
    // Close perf_event file descriptors
    for (int i = 0; i < CACHE_LEVEL_COUNT; i++) {
        if (analyzer->perf_fds[i][0] >= 0) close(analyzer->perf_fds[i][0]);
        if (analyzer->perf_fds[i][1] >= 0) close(analyzer->perf_fds[i][1]);
    }
#endif

    if (analyzer->thread_safe) {
        pthread_mutex_destroy(&analyzer->mutex);
    }

    free(analyzer);
}

bool perf_analyzer_reset(performance_analyzer_t* analyzer) {
    if (!analyzer) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Reset all profiling regions
    for (int i = 0; i < 256; i++) {
        profiling_region_internal_t* region = analyzer->regions[i];
        while (region) {
            region->data.call_count = 0;
            region->data.total_time_ns = 0;
            region->data.min_time_ns = UINT64_MAX;
            region->data.max_time_ns = 0;
            region->data.avg_time_ns = 0;
            region->data.std_dev_ns = 0;
            region->data.total_cpu_cycles = 0;
            region->data.self_time_ns = 0;
            region->data.children_time_ns = 0;
            region->sample_count = 0;
            region = region->next;
        }
    }

    // Reset sample buffers
    for (int i = 0; i < METRIC_TYPE_COUNT; i++) {
        analyzer->sample_buffers[i].count = 0;
    }

    // Reset cache stats
    for (int i = 0; i < CACHE_LEVEL_COUNT; i++) {
        memset(&analyzer->cache_stats[i], 0, sizeof(cache_stats_t));
    }

    // Reset FLOPS counter
    analyzer->total_flops = 0;

    // Reset bandwidth counters
    analyzer->bandwidth_bytes_read = 0;
    analyzer->bandwidth_bytes_written = 0;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Timer Functions
// ============================================================================

perf_timer_t perf_timer_create(const char* name, timer_precision_t precision) {
    perf_timer_t timer = {0};
    timer.name = name;
    timer.precision = precision;
    timer.is_running = false;
    return timer;
}

void perf_timer_start(perf_timer_t* timer) {
    if (!timer) return;
    timer->start_time_ns = perf_get_timestamp_ns();
    timer->cpu_cycles_start = perf_read_cpu_cycles();
    timer->is_running = true;
}

uint64_t perf_timer_stop(perf_timer_t* timer) {
    if (!timer || !timer->is_running) return 0;

    timer->end_time_ns = perf_get_timestamp_ns();
    timer->cpu_cycles_end = perf_read_cpu_cycles();
    timer->elapsed_ns = timer->end_time_ns - timer->start_time_ns;
    timer->is_running = false;

    return timer->elapsed_ns;
}

uint64_t perf_timer_elapsed_ns(const perf_timer_t* timer) {
    if (!timer) return 0;
    if (timer->is_running) {
        return perf_get_timestamp_ns() - timer->start_time_ns;
    }
    return timer->elapsed_ns;
}

// ============================================================================
// CPU Cycle Functions
// ============================================================================

void perf_cycles_begin(performance_analyzer_t* analyzer, const char* region) {
    if (!analyzer || !region) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Find or create cycle region
    cycle_region_t* cr = analyzer->cycle_regions;
    while (cr && strcmp(cr->name, region) != 0) {
        cr = cr->next;
    }

    if (!cr) {
        cr = calloc(1, sizeof(cycle_region_t));
        if (cr) {
            strncpy(cr->name, region, sizeof(cr->name) - 1);
            cr->next = analyzer->cycle_regions;
            analyzer->cycle_regions = cr;
        }
    }

    if (cr) {
        cr->start_cycles = perf_read_cpu_cycles();
        cr->is_active = true;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

uint64_t perf_cycles_end(performance_analyzer_t* analyzer, const char* region) {
    if (!analyzer || !region) return 0;

    uint64_t end_cycles = perf_read_cpu_cycles();
    uint64_t result = 0;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    cycle_region_t* cr = analyzer->cycle_regions;
    while (cr && strcmp(cr->name, region) != 0) {
        cr = cr->next;
    }

    if (cr && cr->is_active) {
        result = end_cycles - cr->start_cycles;
        cr->is_active = false;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return result;
}

double perf_get_cycles_per_op(uint64_t cycles, uint64_t operations) {
    if (operations == 0) return 0.0;
    return (double)cycles / (double)operations;
}

// ============================================================================
// Profiling Region Functions
// ============================================================================

static profiling_region_internal_t* find_or_create_region(
    performance_analyzer_t* analyzer,
    const char* name,
    const char* file,
    int line) {

    uint8_t hash = hash_string(name);
    profiling_region_internal_t* region = analyzer->regions[hash];

    while (region && strcmp(region->data.name, name) != 0) {
        region = region->next;
    }

    if (!region) {
        region = calloc(1, sizeof(profiling_region_internal_t));
        if (!region) return NULL;

        region->data.name = strdup(name);
        region->data.file = file;
        region->data.line = line;
        region->data.min_time_ns = UINT64_MAX;
        region->sample_capacity = 1000;
        region->time_samples = calloc(region->sample_capacity, sizeof(uint64_t));

        region->next = analyzer->regions[hash];
        analyzer->regions[hash] = region;
        analyzer->region_count++;
    }

    return region;
}

void perf_region_begin(performance_analyzer_t* analyzer,
                       const char* name,
                       const char* file,
                       int line) {
    if (!analyzer || !name) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    profiling_region_internal_t* region = find_or_create_region(
        analyzer, name, file, line);

    if (region) {
        region->current_start_ns = perf_get_timestamp_ns();
        region->current_cycles_start = perf_read_cpu_cycles();
        region->is_active = true;

        // Push onto stack
        if (analyzer->region_stack_depth < 64) {
            analyzer->region_stack[analyzer->region_stack_depth++] = region;
        }
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

void perf_region_end(performance_analyzer_t* analyzer, const char* name) {
    if (!analyzer || !name) return;

    uint64_t end_time = perf_get_timestamp_ns();
    uint64_t end_cycles = perf_read_cpu_cycles();

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    uint8_t hash = hash_string(name);
    profiling_region_internal_t* region = analyzer->regions[hash];

    while (region && strcmp(region->data.name, name) != 0) {
        region = region->next;
    }

    if (region && region->is_active) {
        uint64_t elapsed = end_time - region->current_start_ns;
        uint64_t cycles = end_cycles - region->current_cycles_start;

        region->data.call_count++;
        region->data.total_time_ns += elapsed;
        region->data.total_cpu_cycles += cycles;

        if (elapsed < region->data.min_time_ns) {
            region->data.min_time_ns = elapsed;
        }
        if (elapsed > region->data.max_time_ns) {
            region->data.max_time_ns = elapsed;
        }

        // Store sample if we have room
        if (region->sample_count < region->sample_capacity) {
            region->time_samples[region->sample_count++] = elapsed;
        }

        // Update self time
        region->data.self_time_ns = region->data.total_time_ns -
                                    region->data.children_time_ns;

        region->is_active = false;

        // Pop from stack and update parent's children time
        if (analyzer->region_stack_depth > 0) {
            analyzer->region_stack_depth--;
            if (analyzer->region_stack_depth > 0) {
                profiling_region_internal_t* parent =
                    analyzer->region_stack[analyzer->region_stack_depth - 1];
                parent->data.children_time_ns += elapsed;
            }
        }
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

bool perf_get_region_stats(performance_analyzer_t* analyzer,
                           const char* name,
                           profiling_region_t* region) {
    if (!analyzer || !name || !region) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    uint8_t hash = hash_string(name);
    profiling_region_internal_t* r = analyzer->regions[hash];

    while (r && strcmp(r->data.name, name) != 0) {
        r = r->next;
    }

    bool found = false;
    if (r) {
        // Calculate statistics
        r->data.avg_time_ns = (double)r->data.total_time_ns /
                              (double)(r->data.call_count > 0 ? r->data.call_count : 1);

        // Calculate standard deviation
        if (r->sample_count > 1) {
            double sum_sq_diff = 0;
            for (size_t i = 0; i < r->sample_count; i++) {
                double diff = (double)r->time_samples[i] - r->data.avg_time_ns;
                sum_sq_diff += diff * diff;
            }
            r->data.std_dev_ns = sqrt(sum_sq_diff / (r->sample_count - 1));
        }

        *region = r->data;
        found = true;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return found;
}

bool perf_get_all_regions(performance_analyzer_t* analyzer,
                          profiling_region_t** regions,
                          size_t* count) {
    if (!analyzer || !regions || !count) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->region_count;
    if (*count == 0) {
        *regions = NULL;
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *regions = calloc(*count, sizeof(profiling_region_t));
    if (!*regions) {
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (int i = 0; i < 256 && idx < *count; i++) {
        profiling_region_internal_t* r = analyzer->regions[i];
        while (r && idx < *count) {
            // Calculate avg and std_dev
            r->data.avg_time_ns = (double)r->data.total_time_ns /
                                  (double)(r->data.call_count > 0 ? r->data.call_count : 1);

            if (r->sample_count > 1) {
                double sum_sq_diff = 0;
                for (size_t j = 0; j < r->sample_count; j++) {
                    double diff = (double)r->time_samples[j] - r->data.avg_time_ns;
                    sum_sq_diff += diff * diff;
                }
                r->data.std_dev_ns = sqrt(sum_sq_diff / (r->sample_count - 1));
            }

            (*regions)[idx++] = r->data;
            r = r->next;
        }
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Sample Collection
// ============================================================================

void perf_record_sample(performance_analyzer_t* analyzer,
                        perf_metric_type_t type,
                        double value,
                        const char* label) {
    if (!analyzer || type >= METRIC_TYPE_COUNT) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    sample_buffer_t* buffer = &analyzer->sample_buffers[type];

    if (buffer->count < buffer->capacity) {
        perf_sample_t* sample = &buffer->samples[buffer->count++];
        sample->timestamp_ns = perf_get_timestamp_ns();
        sample->value = value;
        sample->metric_type = type;
#ifdef __APPLE__
        sample->thread_id = pthread_mach_thread_np(pthread_self());
#else
        sample->thread_id = (uint32_t)pthread_self();
#endif
        sample->label = label;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

bool perf_get_samples(performance_analyzer_t* analyzer,
                      perf_metric_type_t type,
                      perf_sample_t** samples,
                      size_t* count) {
    if (!analyzer || !samples || !count || type >= METRIC_TYPE_COUNT) {
        return false;
    }

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    sample_buffer_t* buffer = &analyzer->sample_buffers[type];
    *count = buffer->count;

    if (*count == 0) {
        *samples = NULL;
    } else {
        *samples = calloc(*count, sizeof(perf_sample_t));
        if (*samples) {
            memcpy(*samples, buffer->samples, *count * sizeof(perf_sample_t));
        }
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return (*count == 0 || *samples != NULL);
}

void perf_clear_samples(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    for (int i = 0; i < METRIC_TYPE_COUNT; i++) {
        analyzer->sample_buffers[i].count = 0;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Statistics and Aggregation
// ============================================================================

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

double perf_get_percentile(const double* values, size_t count, double percentile) {
    if (!values || count == 0 || percentile < 0 || percentile > 100) {
        return 0.0;
    }

    // Copy and sort
    double* sorted = malloc(count * sizeof(double));
    if (!sorted) return 0.0;

    memcpy(sorted, values, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_doubles);

    double index = (percentile / 100.0) * (count - 1);
    size_t lower = (size_t)floor(index);
    size_t upper = (size_t)ceil(index);

    double result;
    if (lower == upper) {
        result = sorted[lower];
    } else {
        double fraction = index - lower;
        result = sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
    }

    free(sorted);
    return result;
}

bool perf_compute_stats(performance_analyzer_t* analyzer,
                        perf_metric_type_t type,
                        aggregated_stats_t* stats) {
    if (!analyzer || !stats || type >= METRIC_TYPE_COUNT) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    sample_buffer_t* buffer = &analyzer->sample_buffers[type];
    size_t count = buffer->count;

    if (count == 0) {
        memset(stats, 0, sizeof(aggregated_stats_t));
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    // Extract values
    double* values = malloc(count * sizeof(double));
    if (!values) {
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    for (size_t i = 0; i < count; i++) {
        values[i] = buffer->samples[i].value;
    }

    // Calculate statistics
    stats->count = count;
    stats->sum = 0;
    stats->min = values[0];
    stats->max = values[0];

    for (size_t i = 0; i < count; i++) {
        stats->sum += values[i];
        if (values[i] < stats->min) stats->min = values[i];
        if (values[i] > stats->max) stats->max = values[i];
    }

    stats->mean = stats->sum / count;

    // Variance and std dev
    double sum_sq_diff = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - stats->mean;
        sum_sq_diff += diff * diff;
    }
    stats->variance = count > 1 ? sum_sq_diff / (count - 1) : 0;
    stats->std_dev = sqrt(stats->variance);

    // Percentiles
    stats->p50 = perf_get_percentile(values, count, 50.0);
    stats->median = stats->p50;
    stats->p90 = perf_get_percentile(values, count, 90.0);
    stats->p95 = perf_get_percentile(values, count, 95.0);
    stats->p99 = perf_get_percentile(values, count, 99.0);

    free(values);

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool perf_compute_region_stats(performance_analyzer_t* analyzer,
                               const char* region_name,
                               aggregated_stats_t* stats) {
    if (!analyzer || !region_name || !stats) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    uint8_t hash = hash_string(region_name);
    profiling_region_internal_t* region = analyzer->regions[hash];

    while (region && strcmp(region->data.name, region_name) != 0) {
        region = region->next;
    }

    if (!region || region->sample_count == 0) {
        memset(stats, 0, sizeof(aggregated_stats_t));
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return region != NULL;
    }

    // Convert samples to doubles
    double* values = malloc(region->sample_count * sizeof(double));
    if (!values) {
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    for (size_t i = 0; i < region->sample_count; i++) {
        values[i] = (double)region->time_samples[i];
    }

    size_t count = region->sample_count;

    // Calculate statistics
    stats->count = count;
    stats->sum = 0;
    stats->min = values[0];
    stats->max = values[0];

    for (size_t i = 0; i < count; i++) {
        stats->sum += values[i];
        if (values[i] < stats->min) stats->min = values[i];
        if (values[i] > stats->max) stats->max = values[i];
    }

    stats->mean = stats->sum / count;

    double sum_sq_diff = 0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - stats->mean;
        sum_sq_diff += diff * diff;
    }
    stats->variance = count > 1 ? sum_sq_diff / (count - 1) : 0;
    stats->std_dev = sqrt(stats->variance);

    stats->p50 = perf_get_percentile(values, count, 50.0);
    stats->median = stats->p50;
    stats->p90 = perf_get_percentile(values, count, 90.0);
    stats->p95 = perf_get_percentile(values, count, 95.0);
    stats->p99 = perf_get_percentile(values, count, 99.0);

    free(values);

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

// ============================================================================
// Cache Monitoring (Linux perf_event)
// ============================================================================

#ifdef __linux__
static int perf_event_open(struct perf_event_attr* attr, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}
#endif

bool perf_cache_monitoring_start(performance_analyzer_t* analyzer) {
    if (!analyzer) return false;

#ifdef __linux__
    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Try to open L1 cache events
    struct perf_event_attr pe = {0};
    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    // L1 data cache references
    pe.config = (PERF_COUNT_HW_CACHE_L1D) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    analyzer->perf_fds[CACHE_L1_DATA][0] = perf_event_open(&pe, 0, -1, -1, 0);

    // L1 data cache misses
    pe.config = (PERF_COUNT_HW_CACHE_L1D) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    analyzer->perf_fds[CACHE_L1_DATA][1] = perf_event_open(&pe, 0, -1, -1, 0);

    // Enable counters
    for (int i = 0; i < CACHE_LEVEL_COUNT; i++) {
        if (analyzer->perf_fds[i][0] >= 0) {
            ioctl(analyzer->perf_fds[i][0], PERF_EVENT_IOC_RESET, 0);
            ioctl(analyzer->perf_fds[i][0], PERF_EVENT_IOC_ENABLE, 0);
        }
        if (analyzer->perf_fds[i][1] >= 0) {
            ioctl(analyzer->perf_fds[i][1], PERF_EVENT_IOC_RESET, 0);
            ioctl(analyzer->perf_fds[i][1], PERF_EVENT_IOC_ENABLE, 0);
        }
    }

    analyzer->cache_monitoring_active = true;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
    return true;
#else
    set_error("Cache monitoring requires Linux perf_event");
    return false;
#endif
}

void perf_cache_monitoring_stop(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

#ifdef __linux__
    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Read and store final values
    for (int i = 0; i < CACHE_LEVEL_COUNT; i++) {
        if (analyzer->perf_fds[i][0] >= 0) {
            read(analyzer->perf_fds[i][0], &analyzer->cache_stats[i].references,
                 sizeof(uint64_t));
            ioctl(analyzer->perf_fds[i][0], PERF_EVENT_IOC_DISABLE, 0);
        }
        if (analyzer->perf_fds[i][1] >= 0) {
            read(analyzer->perf_fds[i][1], &analyzer->cache_stats[i].misses,
                 sizeof(uint64_t));
            ioctl(analyzer->perf_fds[i][1], PERF_EVENT_IOC_DISABLE, 0);
        }

        // Calculate derived stats
        analyzer->cache_stats[i].hits =
            analyzer->cache_stats[i].references - analyzer->cache_stats[i].misses;
        if (analyzer->cache_stats[i].references > 0) {
            analyzer->cache_stats[i].hit_rate =
                (double)analyzer->cache_stats[i].hits /
                (double)analyzer->cache_stats[i].references;
            analyzer->cache_stats[i].miss_rate = 1.0 - analyzer->cache_stats[i].hit_rate;
        }
    }

    analyzer->cache_monitoring_active = false;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
#endif
}

bool perf_get_cache_stats(performance_analyzer_t* analyzer,
                          cache_level_t level,
                          cache_stats_t* stats) {
    if (!analyzer || !stats || level >= CACHE_LEVEL_COUNT) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    *stats = analyzer->cache_stats[level];
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool perf_get_all_cache_stats(performance_analyzer_t* analyzer,
                              cache_stats_t stats[CACHE_LEVEL_COUNT]) {
    if (!analyzer || !stats) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    memcpy(stats, analyzer->cache_stats, sizeof(analyzer->cache_stats));
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

void perf_reset_cache_counters(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    memset(analyzer->cache_stats, 0, sizeof(analyzer->cache_stats));
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Memory Bandwidth Measurement
// ============================================================================

bool perf_bandwidth_monitoring_start(performance_analyzer_t* analyzer) {
    if (!analyzer) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    analyzer->bandwidth_start_time = perf_get_timestamp_ns();
    analyzer->bandwidth_bytes_read = 0;
    analyzer->bandwidth_bytes_written = 0;
    analyzer->bandwidth_monitoring_active = true;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

void perf_bandwidth_monitoring_stop(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    analyzer->bandwidth_monitoring_active = false;
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

bool perf_get_memory_bandwidth(performance_analyzer_t* analyzer,
                               memory_bandwidth_t* bandwidth) {
    if (!analyzer || !bandwidth) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    uint64_t end_time = perf_get_timestamp_ns();
    double elapsed_sec = (double)(end_time - analyzer->bandwidth_start_time) / 1e9;

    bandwidth->bytes_read = analyzer->bandwidth_bytes_read;
    bandwidth->bytes_written = analyzer->bandwidth_bytes_written;
    bandwidth->measurement_time_sec = elapsed_sec;

    if (elapsed_sec > 0) {
        bandwidth->read_bandwidth_gbps =
            (double)bandwidth->bytes_read / (elapsed_sec * 1e9);
        bandwidth->write_bandwidth_gbps =
            (double)bandwidth->bytes_written / (elapsed_sec * 1e9);
        bandwidth->total_bandwidth_gbps =
            bandwidth->read_bandwidth_gbps + bandwidth->write_bandwidth_gbps;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

bool perf_benchmark_memory_bandwidth(size_t buffer_size,
                                     memory_bandwidth_t* result) {
    if (!result || buffer_size < 1024) return false;

    // Allocate aligned buffer
    void* buffer = NULL;
#ifdef _POSIX_VERSION
    if (posix_memalign(&buffer, 64, buffer_size) != 0) {
        return false;
    }
#else
    buffer = malloc(buffer_size);
    if (!buffer) return false;
#endif

    // Initialize
    memset(buffer, 0, buffer_size);

    // Benchmark read
    uint64_t start = perf_get_timestamp_ns();
    volatile uint64_t sum = 0;
    uint64_t* ptr = (uint64_t*)buffer;
    size_t count = buffer_size / sizeof(uint64_t);

    for (size_t i = 0; i < count; i++) {
        sum += ptr[i];
    }

    uint64_t read_time = perf_get_timestamp_ns() - start;

    // Benchmark write
    start = perf_get_timestamp_ns();

    for (size_t i = 0; i < count; i++) {
        ptr[i] = i;
    }

    uint64_t write_time = perf_get_timestamp_ns() - start;

    // Calculate results
    result->bytes_read = buffer_size;
    result->bytes_written = buffer_size;
    result->measurement_time_sec = (double)(read_time + write_time) / 1e9;

    result->read_bandwidth_gbps =
        (double)buffer_size / ((double)read_time / 1e9) / 1e9;
    result->write_bandwidth_gbps =
        (double)buffer_size / ((double)write_time / 1e9) / 1e9;
    result->total_bandwidth_gbps =
        result->read_bandwidth_gbps + result->write_bandwidth_gbps;

    free(buffer);

    // Prevent optimization
    if (sum == 0) {
        result->read_bandwidth_gbps = 0;
    }

    return true;
}

// ============================================================================
// FLOPS Measurement
// ============================================================================

bool perf_flops_counting_start(performance_analyzer_t* analyzer) {
    if (!analyzer) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    analyzer->flops_start_time = perf_get_timestamp_ns();
    analyzer->total_flops = 0;
    analyzer->flops_counting_active = true;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

void perf_flops_counting_stop(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    analyzer->flops_counting_active = false;
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

void perf_record_flops(performance_analyzer_t* analyzer, uint64_t flops) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    analyzer->total_flops += flops;
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

bool perf_get_flops_stats(performance_analyzer_t* analyzer,
                          flops_stats_t* stats) {
    if (!analyzer || !stats) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    uint64_t end_time = perf_get_timestamp_ns();
    double elapsed_sec = (double)(end_time - analyzer->flops_start_time) / 1e9;

    stats->total_flops = analyzer->total_flops;
    stats->measurement_time_sec = elapsed_sec;

    if (elapsed_sec > 0) {
        stats->gflops = (double)analyzer->total_flops / (elapsed_sec * 1e9);
    } else {
        stats->gflops = 0;
    }

    stats->theoretical_peak_gflops = perf_get_theoretical_peak_gflops();

    if (stats->theoretical_peak_gflops > 0) {
        stats->efficiency = (stats->gflops / stats->theoretical_peak_gflops) * 100.0;
    } else {
        stats->efficiency = 0;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return true;
}

double perf_get_theoretical_peak_gflops(void) {
    // Get CPU info and estimate peak FLOPS
    uint64_t freq = perf_get_cpu_frequency();

#ifdef __APPLE__
    int cores = 0;
    size_t size = sizeof(cores);
    sysctlbyname("hw.physicalcpu", &cores, &size, NULL, 0);

    // Apple Silicon: estimate based on architecture
    // M1/M2: ~3.5 TFLOPS for GPU, ~500 GFLOPS for CPU
    if (cores >= 8) {
        return 500.0;  // High-performance Apple Silicon
    }
#elif defined(__linux__)
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
#else
    int cores = 4;
#endif

    // Estimate: freq_ghz * cores * fma_per_cycle * simd_width
    double freq_ghz = (double)freq / 1e9;
    double fma_per_cycle = 2.0;  // FMA = 2 FLOPS
    double simd_width = 8.0;     // AVX-512 or similar

    return freq_ghz * cores * fma_per_cycle * simd_width;
}

// ============================================================================
// Export Functions
// ============================================================================

char* perf_export_to_json(performance_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Estimate buffer size
    size_t buffer_size = 4096 + analyzer->region_count * 512;
    char* json = malloc(buffer_size);
    if (!json) {
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    size_t offset = 0;
    offset += snprintf(json + offset, buffer_size - offset,
                       "{\n  \"profiling_regions\": [\n");

    bool first_region = true;
    for (int i = 0; i < 256; i++) {
        profiling_region_internal_t* r = analyzer->regions[i];
        while (r) {
            if (!first_region) {
                offset += snprintf(json + offset, buffer_size - offset, ",\n");
            }
            first_region = false;

            double avg_time_ns = (double)r->data.total_time_ns /
                                 (double)(r->data.call_count > 0 ? r->data.call_count : 1);

            offset += snprintf(json + offset, buffer_size - offset,
                "    {\n"
                "      \"name\": \"%s\",\n"
                "      \"file\": \"%s\",\n"
                "      \"line\": %d,\n"
                "      \"call_count\": %llu,\n"
                "      \"total_time_ns\": %llu,\n"
                "      \"avg_time_ns\": %.2f,\n"
                "      \"min_time_ns\": %llu,\n"
                "      \"max_time_ns\": %llu,\n"
                "      \"total_cpu_cycles\": %llu\n"
                "    }",
                r->data.name,
                r->data.file ? r->data.file : "unknown",
                r->data.line,
                (unsigned long long)r->data.call_count,
                (unsigned long long)r->data.total_time_ns,
                avg_time_ns,
                (unsigned long long)r->data.min_time_ns,
                (unsigned long long)r->data.max_time_ns,
                (unsigned long long)r->data.total_cpu_cycles);

            r = r->next;
        }
    }

    offset += snprintf(json + offset, buffer_size - offset,
                       "\n  ],\n  \"timer_overhead_ns\": %llu\n}",
                       (unsigned long long)analyzer->timer_overhead_ns);

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return json;
}

char* perf_export_to_csv(performance_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    size_t buffer_size = 4096 + analyzer->region_count * 256;
    char* csv = malloc(buffer_size);
    if (!csv) {
        if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
        return NULL;
    }

    size_t offset = 0;
    offset += snprintf(csv + offset, buffer_size - offset,
        "name,file,line,call_count,total_time_ns,avg_time_ns,min_time_ns,"
        "max_time_ns,total_cpu_cycles\n");

    for (int i = 0; i < 256; i++) {
        profiling_region_internal_t* r = analyzer->regions[i];
        while (r) {
            double avg_time_ns = (double)r->data.total_time_ns /
                                 (double)(r->data.call_count > 0 ? r->data.call_count : 1);

            offset += snprintf(csv + offset, buffer_size - offset,
                "%s,%s,%d,%llu,%llu,%.2f,%llu,%llu,%llu\n",
                r->data.name,
                r->data.file ? r->data.file : "unknown",
                r->data.line,
                (unsigned long long)r->data.call_count,
                (unsigned long long)r->data.total_time_ns,
                avg_time_ns,
                (unsigned long long)r->data.min_time_ns,
                (unsigned long long)r->data.max_time_ns,
                (unsigned long long)r->data.total_cpu_cycles);

            r = r->next;
        }
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return csv;
}

bool perf_export_data(performance_analyzer_t* analyzer,
                      const char* filename,
                      export_format_t format) {
    if (!analyzer || !filename) return false;

    char* data = NULL;

    switch (format) {
        case EXPORT_FORMAT_JSON:
            data = perf_export_to_json(analyzer);
            break;
        case EXPORT_FORMAT_CSV:
            data = perf_export_to_csv(analyzer);
            break;
        default:
            set_error("Unsupported export format");
            return false;
    }

    if (!data) return false;

    FILE* f = fopen(filename, "w");
    if (!f) {
        free(data);
        set_error("Failed to open file for writing");
        return false;
    }

    fputs(data, f);
    fclose(f);
    free(data);

    return true;
}

bool perf_import_data(performance_analyzer_t* analyzer,
                      const char* filename,
                      export_format_t format) {
    (void)analyzer;
    (void)filename;
    (void)format;
    set_error("Import not yet implemented");
    return false;
}

// ============================================================================
// Flame Graph Functions
// ============================================================================

bool perf_flamegraph_start(performance_analyzer_t* analyzer) {
    if (!analyzer) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Create root node
    analyzer->flamegraph_root = calloc(1, sizeof(flame_node_t));
    if (analyzer->flamegraph_root) {
        analyzer->flamegraph_root->function_name = strdup("root");
        analyzer->flamegraph_root->children_capacity = 16;
        analyzer->flamegraph_root->children =
            calloc(16, sizeof(flame_node_t*));
        analyzer->current_flame_node = analyzer->flamegraph_root;
        analyzer->flamegraph_active = true;
    }

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return analyzer->flamegraph_root != NULL;
}

void perf_flamegraph_stop(performance_analyzer_t* analyzer) {
    if (!analyzer) return;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);
    analyzer->flamegraph_active = false;
    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);
}

bool perf_get_flamegraph_data(performance_analyzer_t* analyzer,
                              call_stack_entry_t** root) {
    if (!analyzer || !root) return false;

    if (analyzer->thread_safe) pthread_mutex_lock(&analyzer->mutex);

    // Return the root cast to public type
    *root = (call_stack_entry_t*)analyzer->flamegraph_root;

    if (analyzer->thread_safe) pthread_mutex_unlock(&analyzer->mutex);

    return *root != NULL;
}

static void free_flame_node(flame_node_t* node) {
    if (!node) return;

    for (size_t i = 0; i < node->num_children; i++) {
        free_flame_node(node->children[i]);
    }

    free(node->function_name);
    free(node->children);
    free(node);
}

void perf_free_flamegraph_data(call_stack_entry_t* root) {
    free_flame_node((flame_node_t*)root);
}

bool perf_export_flamegraph_svg(performance_analyzer_t* analyzer,
                                const char* filename,
                                size_t width,
                                size_t height) {
    (void)analyzer;
    (void)filename;
    (void)width;
    (void)height;
    set_error("Flame graph SVG export not yet implemented");
    return false;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* perf_metric_type_name(perf_metric_type_t type) {
    switch (type) {
        case METRIC_WALL_TIME: return "wall_time";
        case METRIC_CPU_TIME: return "cpu_time";
        case METRIC_CPU_CYCLES: return "cpu_cycles";
        case METRIC_INSTRUCTIONS: return "instructions";
        case METRIC_CACHE_MISSES: return "cache_misses";
        case METRIC_CACHE_REFERENCES: return "cache_references";
        case METRIC_BRANCH_MISSES: return "branch_misses";
        case METRIC_PAGE_FAULTS: return "page_faults";
        case METRIC_CONTEXT_SWITCHES: return "context_switches";
        case METRIC_MEMORY_BANDWIDTH: return "memory_bandwidth";
        case METRIC_FLOPS: return "flops";
        default: return "unknown";
    }
}

const char* perf_cache_level_name(cache_level_t level) {
    switch (level) {
        case CACHE_L1_DATA: return "L1_data";
        case CACHE_L1_INSTRUCTION: return "L1_instruction";
        case CACHE_L2: return "L2";
        case CACHE_L3: return "L3";
        default: return "unknown";
    }
}

char* perf_format_duration(uint64_t nanoseconds) {
    char* buffer = malloc(64);
    if (!buffer) return NULL;

    if (nanoseconds < 1000) {
        snprintf(buffer, 64, "%llu ns", (unsigned long long)nanoseconds);
    } else if (nanoseconds < 1000000) {
        snprintf(buffer, 64, "%.2f µs", nanoseconds / 1000.0);
    } else if (nanoseconds < 1000000000) {
        snprintf(buffer, 64, "%.2f ms", nanoseconds / 1000000.0);
    } else {
        snprintf(buffer, 64, "%.2f s", nanoseconds / 1000000000.0);
    }

    return buffer;
}

char* perf_format_bytes(uint64_t bytes) {
    char* buffer = malloc(64);
    if (!buffer) return NULL;

    if (bytes < 1024) {
        snprintf(buffer, 64, "%llu B", (unsigned long long)bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buffer, 64, "%.2f KB", bytes / 1024.0);
    } else if (bytes < 1024 * 1024 * 1024) {
        snprintf(buffer, 64, "%.2f MB", bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buffer, 64, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }

    return buffer;
}

bool perf_hw_counters_available(void) {
#ifdef __linux__
    // Try to open a basic perf event
    struct perf_event_attr pe = {0};
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;

    int fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd >= 0) {
        close(fd);
        return true;
    }
    return false;
#else
    return false;
#endif
}
