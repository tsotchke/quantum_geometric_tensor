/**
 * @file runtime_analyzer.c
 * @brief Implementation of Runtime Performance Analysis
 *
 * Provides nanosecond-precision timing, statistical analysis,
 * histogram generation, regression detection, and bottleneck analysis.
 */

#include "quantum_geometric/core/runtime_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <pthread.h>
#elif defined(__linux__)
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// ============================================================================
// Constants
// ============================================================================

#define HASH_TABLE_SIZE 4096
#define MAX_ACTIVE_TIMINGS 1024
#define MAX_SPANS 10000
#define TREND_WINDOW_SIZE 100
#define MIN_SAMPLES_FOR_STATS 10

// ============================================================================
// Internal Structures
// ============================================================================

// Hash table entry for operations
typedef struct op_entry {
    runtime_operation_t data;
    double* recent_samples;           // Circular buffer for trend analysis
    size_t sample_write_idx;
    size_t sample_count;
    runtime_stats_t baseline_stats;   // Baseline for regression detection
    bool has_baseline;
    struct op_entry* next;
} op_entry_t;

// Active timing entry
typedef struct {
    uint64_t handle;
    char operation_name[RUNTIME_MAX_OPERATION_NAME];
    runtime_op_category_t category;
    uint64_t start_ns;
    uint32_t thread_id;
    bool in_use;
} active_timing_t;

// Span stack for hierarchical timing
typedef struct {
    runtime_span_t* spans;
    size_t count;
    size_t capacity;
    uint32_t next_span_id;
    uint32_t current_depth;
    uint32_t stack[RUNTIME_MAX_STACK_DEPTH];  // Parent span ID stack
    size_t stack_size;
} span_tracker_t;

// Main analyzer structure
struct runtime_analyzer {
    runtime_analyzer_config_t config;

    // Operation tracking (hash table)
    op_entry_t* operations[HASH_TABLE_SIZE];
    size_t operation_count;

    // Active timings
    active_timing_t active_timings[MAX_ACTIVE_TIMINGS];
    atomic_uint_fast64_t next_handle;

    // Span tracking
    span_tracker_t spans;

    // State
    bool enabled;
    uint64_t start_time_ns;

    // Thread safety
    pthread_mutex_t mutex;
    pthread_mutex_t timing_mutex;

    // Error handling
    char last_error[256];
};

// Thread-local error storage
static __thread char tls_error[256] = {0};

// ============================================================================
// Platform-Specific Timing
// ============================================================================

#ifdef __APPLE__
static mach_timebase_info_data_t g_timebase = {0, 0};
static pthread_once_t g_timebase_once = PTHREAD_ONCE_INIT;

static void init_timebase(void) {
    mach_timebase_info(&g_timebase);
}
#endif

uint64_t runtime_get_timestamp_ns(void) {
#ifdef __APPLE__
    pthread_once(&g_timebase_once, init_timebase);
    return (mach_absolute_time() * g_timebase.numer) / g_timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static uint32_t get_thread_id(void) {
#ifdef __APPLE__
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
    return (uint32_t)tid;
#else
    return (uint32_t)syscall(SYS_gettid);
#endif
}

// ============================================================================
// Hash Function
// ============================================================================

static size_t hash_string(const char* str) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_TABLE_SIZE;
}

// ============================================================================
// Error Handling
// ============================================================================

static void set_error(runtime_analyzer_t* analyzer, const char* msg) {
    if (analyzer) {
        strncpy(analyzer->last_error, msg, sizeof(analyzer->last_error) - 1);
        analyzer->last_error[sizeof(analyzer->last_error) - 1] = '\0';
    }
    strncpy(tls_error, msg, sizeof(tls_error) - 1);
    tls_error[sizeof(tls_error) - 1] = '\0';
}

const char* runtime_get_last_error(runtime_analyzer_t* analyzer) {
    if (analyzer && analyzer->last_error[0]) {
        return analyzer->last_error;
    }
    return tls_error;
}

// ============================================================================
// Configuration
// ============================================================================

runtime_analyzer_config_t runtime_analyzer_default_config(void) {
    return (runtime_analyzer_config_t){
        .enable_histogram = true,
        .enable_hierarchical = true,
        .enable_regression_detection = true,
        .enable_bottleneck_analysis = true,
        .max_operations = RUNTIME_MAX_OPERATIONS,
        .histogram_bins = 50,
        .regression_threshold = 10.0,  // 10% regression threshold
        .outlier_threshold = 3.0,      // 3 standard deviations
        .warmup_samples = 5
    };
}

// ============================================================================
// Initialization and Destruction
// ============================================================================

runtime_analyzer_t* runtime_analyzer_create(void) {
    runtime_analyzer_config_t config = runtime_analyzer_default_config();
    return runtime_analyzer_create_with_config(&config);
}

runtime_analyzer_t* runtime_analyzer_create_with_config(
    const runtime_analyzer_config_t* config) {

    if (!config) {
        set_error(NULL, "NULL config provided");
        return NULL;
    }

    runtime_analyzer_t* analyzer = calloc(1, sizeof(runtime_analyzer_t));
    if (!analyzer) {
        set_error(NULL, "Failed to allocate analyzer");
        return NULL;
    }

    analyzer->config = *config;
    analyzer->enabled = true;
    analyzer->start_time_ns = runtime_get_timestamp_ns();

    pthread_mutex_init(&analyzer->mutex, NULL);
    pthread_mutex_init(&analyzer->timing_mutex, NULL);

    atomic_init(&analyzer->next_handle, 1);

    // Initialize span tracking
    if (config->enable_hierarchical) {
        analyzer->spans.capacity = MAX_SPANS;
        analyzer->spans.spans = calloc(MAX_SPANS, sizeof(runtime_span_t));
        analyzer->spans.next_span_id = 1;
    }

    return analyzer;
}

void runtime_analyzer_destroy(runtime_analyzer_t* analyzer) {
    if (!analyzer) return;

    pthread_mutex_lock(&analyzer->mutex);

    // Free operation hash table
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            op_entry_t* next = entry->next;
            free(entry->recent_samples);
            free(entry);
            entry = next;
        }
    }

    // Free spans
    free(analyzer->spans.spans);

    pthread_mutex_unlock(&analyzer->mutex);
    pthread_mutex_destroy(&analyzer->mutex);
    pthread_mutex_destroy(&analyzer->timing_mutex);

    free(analyzer);
}

bool runtime_analyzer_reset(runtime_analyzer_t* analyzer) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Clear all operations
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            op_entry_t* next = entry->next;
            free(entry->recent_samples);
            free(entry);
            entry = next;
        }
        analyzer->operations[i] = NULL;
    }
    analyzer->operation_count = 0;

    // Reset spans
    analyzer->spans.count = 0;
    analyzer->spans.stack_size = 0;
    analyzer->spans.current_depth = 0;
    analyzer->spans.next_span_id = 1;

    // Reset active timings
    memset(analyzer->active_timings, 0, sizeof(analyzer->active_timings));

    analyzer->start_time_ns = runtime_get_timestamp_ns();

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

void runtime_analyzer_enable(runtime_analyzer_t* analyzer, bool enable) {
    if (analyzer) {
        analyzer->enabled = enable;
    }
}

bool runtime_analyzer_is_enabled(runtime_analyzer_t* analyzer) {
    return analyzer ? analyzer->enabled : false;
}

// ============================================================================
// Operation Lookup/Create
// ============================================================================

static op_entry_t* find_or_create_operation(runtime_analyzer_t* analyzer,
                                             const char* name,
                                             runtime_op_category_t category) {
    size_t idx = hash_string(name);

    // Search existing
    op_entry_t* entry = analyzer->operations[idx];
    while (entry) {
        if (strcmp(entry->data.name, name) == 0) {
            return entry;
        }
        entry = entry->next;
    }

    // Create new
    if (analyzer->operation_count >= analyzer->config.max_operations) {
        set_error(analyzer, "Maximum operations reached");
        return NULL;
    }

    entry = calloc(1, sizeof(op_entry_t));
    if (!entry) {
        set_error(analyzer, "Failed to allocate operation entry");
        return NULL;
    }

    strncpy(entry->data.name, name, RUNTIME_MAX_OPERATION_NAME - 1);
    entry->data.category = category;
    entry->data.stats.min_ns = UINT64_MAX;
    entry->data.first_seen_ns = runtime_get_timestamp_ns();
    entry->data.trend = RUNTIME_TREND_UNKNOWN;

    // Allocate sample buffer for trend analysis
    entry->recent_samples = calloc(TREND_WINDOW_SIZE, sizeof(double));

    // Initialize histogram
    if (analyzer->config.enable_histogram) {
        entry->data.histogram.num_bins = analyzer->config.histogram_bins;
        entry->data.histogram.min_observed_ns = UINT64_MAX;
    }

    // Insert into hash table
    entry->next = analyzer->operations[idx];
    analyzer->operations[idx] = entry;
    analyzer->operation_count++;

    return entry;
}

// ============================================================================
// Statistics Update
// ============================================================================

static void update_stats(op_entry_t* entry, uint64_t duration_ns) {
    runtime_stats_t* stats = &entry->data.stats;

    stats->count++;
    stats->total_ns += duration_ns;

    if (duration_ns < stats->min_ns) {
        stats->min_ns = duration_ns;
    }
    if (duration_ns > stats->max_ns) {
        stats->max_ns = duration_ns;
    }

    // Online mean and variance (Welford's algorithm)
    double delta = (double)duration_ns - stats->mean_ns;
    stats->mean_ns += delta / (double)stats->count;
    double delta2 = (double)duration_ns - stats->mean_ns;
    stats->variance_ns += delta * delta2;

    if (stats->count > 1) {
        stats->std_dev_ns = sqrt(stats->variance_ns / (double)(stats->count - 1));
    }

    // Store sample for trend analysis
    if (entry->recent_samples) {
        entry->recent_samples[entry->sample_write_idx] = (double)duration_ns;
        entry->sample_write_idx = (entry->sample_write_idx + 1) % TREND_WINDOW_SIZE;
        if (entry->sample_count < TREND_WINDOW_SIZE) {
            entry->sample_count++;
        }
    }

    entry->data.last_seen_ns = runtime_get_timestamp_ns();
}

static void update_histogram(runtime_histogram_t* hist, uint64_t duration_ns) {
    if (hist->total_samples == 0) {
        hist->min_observed_ns = duration_ns;
        hist->max_observed_ns = duration_ns;
    } else {
        if (duration_ns < hist->min_observed_ns) {
            hist->min_observed_ns = duration_ns;
        }
        if (duration_ns > hist->max_observed_ns) {
            hist->max_observed_ns = duration_ns;
        }
    }

    hist->total_samples++;

    // Dynamic binning based on observed range
    if (hist->total_samples <= 10) {
        // Not enough samples for binning yet
        return;
    }

    // Calculate bin edges if needed
    if (hist->bin_edges_ns[0] == 0) {
        uint64_t range = hist->max_observed_ns - hist->min_observed_ns;
        uint64_t bin_width = range / hist->num_bins;
        if (bin_width == 0) bin_width = 1;

        for (size_t i = 0; i <= hist->num_bins; i++) {
            hist->bin_edges_ns[i] = hist->min_observed_ns + i * bin_width;
        }
    }

    // Find bin
    if (duration_ns < hist->bin_edges_ns[0]) {
        hist->outliers_low++;
    } else if (duration_ns >= hist->bin_edges_ns[hist->num_bins]) {
        hist->outliers_high++;
    } else {
        for (size_t i = 0; i < hist->num_bins; i++) {
            if (duration_ns >= hist->bin_edges_ns[i] &&
                duration_ns < hist->bin_edges_ns[i + 1]) {
                hist->bin_counts[i]++;
                break;
            }
        }
    }
}

// ============================================================================
// Basic Timing Operations
// ============================================================================

uint64_t runtime_start_timing(runtime_analyzer_t* analyzer,
                               const char* operation_name,
                               runtime_op_category_t category) {
    if (!analyzer || !analyzer->enabled || !operation_name) {
        return 0;
    }

    uint64_t handle = atomic_fetch_add(&analyzer->next_handle, 1);

    pthread_mutex_lock(&analyzer->timing_mutex);

    // Find free slot
    active_timing_t* slot = NULL;
    for (size_t i = 0; i < MAX_ACTIVE_TIMINGS; i++) {
        if (!analyzer->active_timings[i].in_use) {
            slot = &analyzer->active_timings[i];
            break;
        }
    }

    if (!slot) {
        pthread_mutex_unlock(&analyzer->timing_mutex);
        set_error(analyzer, "No free timing slots");
        return 0;
    }

    slot->handle = handle;
    strncpy(slot->operation_name, operation_name, RUNTIME_MAX_OPERATION_NAME - 1);
    slot->category = category;
    slot->start_ns = runtime_get_timestamp_ns();
    slot->thread_id = get_thread_id();
    slot->in_use = true;

    pthread_mutex_unlock(&analyzer->timing_mutex);

    return handle;
}

void runtime_stop_timing(runtime_analyzer_t* analyzer, uint64_t handle) {
    if (!analyzer || handle == 0) return;

    uint64_t end_ns = runtime_get_timestamp_ns();

    pthread_mutex_lock(&analyzer->timing_mutex);

    // Find the timing
    active_timing_t* slot = NULL;
    for (size_t i = 0; i < MAX_ACTIVE_TIMINGS; i++) {
        if (analyzer->active_timings[i].in_use &&
            analyzer->active_timings[i].handle == handle) {
            slot = &analyzer->active_timings[i];
            break;
        }
    }

    if (!slot) {
        pthread_mutex_unlock(&analyzer->timing_mutex);
        return;
    }

    uint64_t duration_ns = end_ns - slot->start_ns;
    char name[RUNTIME_MAX_OPERATION_NAME];
    strncpy(name, slot->operation_name, RUNTIME_MAX_OPERATION_NAME);
    runtime_op_category_t category = slot->category;

    slot->in_use = false;

    pthread_mutex_unlock(&analyzer->timing_mutex);

    // Record the timing
    pthread_mutex_lock(&analyzer->mutex);

    op_entry_t* entry = find_or_create_operation(analyzer, name, category);
    if (entry) {
        update_stats(entry, duration_ns);
        if (analyzer->config.enable_histogram) {
            update_histogram(&entry->data.histogram, duration_ns);
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

void runtime_record_duration(runtime_analyzer_t* analyzer,
                             const char* operation_name,
                             runtime_op_category_t category,
                             uint64_t duration_ns) {
    if (!analyzer || !analyzer->enabled || !operation_name) return;

    pthread_mutex_lock(&analyzer->mutex);

    op_entry_t* entry = find_or_create_operation(analyzer, operation_name, category);
    if (entry) {
        update_stats(entry, duration_ns);
        if (analyzer->config.enable_histogram) {
            update_histogram(&entry->data.histogram, duration_ns);
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

// ============================================================================
// Scoped Timing
// ============================================================================

runtime_scoped_timer_t runtime_scoped_begin(runtime_analyzer_t* analyzer,
                                             const char* operation_name,
                                             runtime_op_category_t category) {
    runtime_scoped_timer_t timer = {
        .analyzer = analyzer,
        .handle = runtime_start_timing(analyzer, operation_name, category)
    };
    return timer;
}

void runtime_scoped_end(runtime_scoped_timer_t* timer) {
    if (timer && timer->handle) {
        runtime_stop_timing(timer->analyzer, timer->handle);
        timer->handle = 0;
    }
}

// ============================================================================
// Hierarchical/Span Timing
// ============================================================================

uint32_t runtime_begin_span(runtime_analyzer_t* analyzer,
                            const char* name,
                            runtime_op_category_t category) {
    if (!analyzer || !analyzer->enabled || !name) return 0;
    if (!analyzer->config.enable_hierarchical) return 0;

    pthread_mutex_lock(&analyzer->mutex);

    span_tracker_t* tracker = &analyzer->spans;

    if (tracker->count >= tracker->capacity) {
        pthread_mutex_unlock(&analyzer->mutex);
        set_error(analyzer, "Span buffer full");
        return 0;
    }

    uint32_t span_id = tracker->next_span_id++;
    runtime_span_t* span = &tracker->spans[tracker->count++];

    strncpy(span->name, name, RUNTIME_MAX_OPERATION_NAME - 1);
    span->start_ns = runtime_get_timestamp_ns();
    span->category = category;
    span->depth = tracker->current_depth;
    span->span_id = span_id;
    span->parent_id = (tracker->stack_size > 0) ?
                      tracker->stack[tracker->stack_size - 1] : 0;

    // Push onto stack
    if (tracker->stack_size < RUNTIME_MAX_STACK_DEPTH) {
        tracker->stack[tracker->stack_size++] = span_id;
        tracker->current_depth++;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return span_id;
}

void runtime_end_span(runtime_analyzer_t* analyzer, uint32_t span_id) {
    if (!analyzer || span_id == 0) return;

    uint64_t end_ns = runtime_get_timestamp_ns();

    pthread_mutex_lock(&analyzer->mutex);

    span_tracker_t* tracker = &analyzer->spans;

    // Find the span
    for (size_t i = 0; i < tracker->count; i++) {
        if (tracker->spans[i].span_id == span_id && tracker->spans[i].end_ns == 0) {
            tracker->spans[i].end_ns = end_ns;
            break;
        }
    }

    // Pop from stack
    if (tracker->stack_size > 0 && tracker->stack[tracker->stack_size - 1] == span_id) {
        tracker->stack_size--;
        if (tracker->current_depth > 0) {
            tracker->current_depth--;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

bool runtime_get_spans(runtime_analyzer_t* analyzer,
                       runtime_span_t** spans,
                       size_t* count) {
    if (!analyzer || !spans || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->spans.count;
    if (*count == 0) {
        *spans = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *spans = calloc(*count, sizeof(runtime_span_t));
    if (!*spans) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    memcpy(*spans, analyzer->spans.spans, *count * sizeof(runtime_span_t));

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

// ============================================================================
// Statistics Retrieval
// ============================================================================

bool runtime_get_operation_stats(runtime_analyzer_t* analyzer,
                                 const char* operation_name,
                                 runtime_stats_t* stats) {
    if (!analyzer || !operation_name || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    size_t idx = hash_string(operation_name);
    op_entry_t* entry = analyzer->operations[idx];

    while (entry) {
        if (strcmp(entry->data.name, operation_name) == 0) {
            *stats = entry->data.stats;

            // Calculate throughput
            uint64_t elapsed_ns = runtime_get_timestamp_ns() - analyzer->start_time_ns;
            if (elapsed_ns > 0) {
                stats->throughput_per_sec = (double)stats->count *
                                            1000000000.0 / (double)elapsed_ns;
            }

            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    set_error(analyzer, "Operation not found");
    return false;
}

bool runtime_get_all_stats(runtime_analyzer_t* analyzer,
                           runtime_operation_t** operations,
                           size_t* count) {
    if (!analyzer || !operations || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    *count = analyzer->operation_count;
    if (*count == 0) {
        *operations = NULL;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    *operations = calloc(*count, sizeof(runtime_operation_t));
    if (!*operations) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t idx = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE && idx < *count; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry && idx < *count) {
            (*operations)[idx++] = entry->data;
            entry = entry->next;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool runtime_get_category_stats(runtime_analyzer_t* analyzer,
                                runtime_op_category_t category,
                                runtime_stats_t* stats) {
    if (!analyzer || !stats) return false;

    pthread_mutex_lock(&analyzer->mutex);

    memset(stats, 0, sizeof(runtime_stats_t));
    stats->min_ns = UINT64_MAX;

    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            if (entry->data.category == category) {
                stats->count += entry->data.stats.count;
                stats->total_ns += entry->data.stats.total_ns;
                if (entry->data.stats.min_ns < stats->min_ns) {
                    stats->min_ns = entry->data.stats.min_ns;
                }
                if (entry->data.stats.max_ns > stats->max_ns) {
                    stats->max_ns = entry->data.stats.max_ns;
                }
            }
            entry = entry->next;
        }
    }

    if (stats->count > 0) {
        stats->mean_ns = (double)stats->total_ns / (double)stats->count;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool runtime_get_histogram(runtime_analyzer_t* analyzer,
                           const char* operation_name,
                           runtime_histogram_t* histogram) {
    if (!analyzer || !operation_name || !histogram) return false;

    pthread_mutex_lock(&analyzer->mutex);

    size_t idx = hash_string(operation_name);
    op_entry_t* entry = analyzer->operations[idx];

    while (entry) {
        if (strcmp(entry->data.name, operation_name) == 0) {
            *histogram = entry->data.histogram;
            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return false;
}

// ============================================================================
// Sorting helpers for top-N queries
// ============================================================================

static int compare_by_mean_desc(const void* a, const void* b) {
    const runtime_operation_t* op_a = (const runtime_operation_t*)a;
    const runtime_operation_t* op_b = (const runtime_operation_t*)b;
    if (op_a->stats.mean_ns > op_b->stats.mean_ns) return -1;
    if (op_a->stats.mean_ns < op_b->stats.mean_ns) return 1;
    return 0;
}

static int compare_by_count_desc(const void* a, const void* b) {
    const runtime_operation_t* op_a = (const runtime_operation_t*)a;
    const runtime_operation_t* op_b = (const runtime_operation_t*)b;
    if (op_a->stats.count > op_b->stats.count) return -1;
    if (op_a->stats.count < op_b->stats.count) return 1;
    return 0;
}

bool runtime_get_slowest_operations(runtime_analyzer_t* analyzer,
                                    size_t n,
                                    runtime_operation_t** operations,
                                    size_t* count) {
    runtime_operation_t* all_ops = NULL;
    size_t all_count = 0;

    if (!runtime_get_all_stats(analyzer, &all_ops, &all_count)) {
        return false;
    }

    qsort(all_ops, all_count, sizeof(runtime_operation_t), compare_by_mean_desc);

    *count = (n < all_count) ? n : all_count;
    *operations = calloc(*count, sizeof(runtime_operation_t));
    if (!*operations) {
        free(all_ops);
        return false;
    }

    memcpy(*operations, all_ops, *count * sizeof(runtime_operation_t));
    free(all_ops);

    return true;
}

bool runtime_get_hottest_operations(runtime_analyzer_t* analyzer,
                                    size_t n,
                                    runtime_operation_t** operations,
                                    size_t* count) {
    runtime_operation_t* all_ops = NULL;
    size_t all_count = 0;

    if (!runtime_get_all_stats(analyzer, &all_ops, &all_count)) {
        return false;
    }

    qsort(all_ops, all_count, sizeof(runtime_operation_t), compare_by_count_desc);

    *count = (n < all_count) ? n : all_count;
    *operations = calloc(*count, sizeof(runtime_operation_t));
    if (!*operations) {
        free(all_ops);
        return false;
    }

    memcpy(*operations, all_ops, *count * sizeof(runtime_operation_t));
    free(all_ops);

    return true;
}

// ============================================================================
// Trend Analysis
// ============================================================================

runtime_trend_t runtime_get_trend(runtime_analyzer_t* analyzer,
                                  const char* operation_name) {
    if (!analyzer || !operation_name) return RUNTIME_TREND_UNKNOWN;

    pthread_mutex_lock(&analyzer->mutex);

    size_t idx = hash_string(operation_name);
    op_entry_t* entry = analyzer->operations[idx];

    while (entry) {
        if (strcmp(entry->data.name, operation_name) == 0) {
            if (entry->sample_count < MIN_SAMPLES_FOR_STATS) {
                pthread_mutex_unlock(&analyzer->mutex);
                return RUNTIME_TREND_UNKNOWN;
            }

            // Calculate trend using linear regression on recent samples
            size_t n = entry->sample_count;
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

            for (size_t i = 0; i < n; i++) {
                size_t sample_idx = (entry->sample_write_idx + TREND_WINDOW_SIZE - n + i)
                                    % TREND_WINDOW_SIZE;
                double x = (double)i;
                double y = entry->recent_samples[sample_idx];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            }

            double mean_x = sum_x / n;
            double mean_y = sum_y / n;
            double slope = (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x);

            // Calculate variance for volatility detection
            double variance = 0;
            for (size_t i = 0; i < n; i++) {
                size_t sample_idx = (entry->sample_write_idx + TREND_WINDOW_SIZE - n + i)
                                    % TREND_WINDOW_SIZE;
                double diff = entry->recent_samples[sample_idx] - mean_y;
                variance += diff * diff;
            }
            variance /= n;
            double std_dev = sqrt(variance);
            double cv = std_dev / mean_y;  // Coefficient of variation

            pthread_mutex_unlock(&analyzer->mutex);

            // Classify trend
            if (cv > 0.5) {
                return RUNTIME_TREND_VOLATILE;
            }

            double relative_slope = slope / mean_y * 100.0;  // % change per sample

            if (relative_slope > 1.0) {
                return RUNTIME_TREND_DEGRADING;
            } else if (relative_slope < -1.0) {
                return RUNTIME_TREND_IMPROVING;
            }
            return RUNTIME_TREND_STABLE;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return RUNTIME_TREND_UNKNOWN;
}

// ============================================================================
// Regression Detection
// ============================================================================

bool runtime_set_baseline(runtime_analyzer_t* analyzer,
                          const char* operation_name) {
    if (!analyzer || !operation_name) return false;

    pthread_mutex_lock(&analyzer->mutex);

    size_t idx = hash_string(operation_name);
    op_entry_t* entry = analyzer->operations[idx];

    while (entry) {
        if (strcmp(entry->data.name, operation_name) == 0) {
            entry->baseline_stats = entry->data.stats;
            entry->has_baseline = true;
            pthread_mutex_unlock(&analyzer->mutex);
            return true;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return false;
}

bool runtime_set_all_baselines(runtime_analyzer_t* analyzer) {
    if (!analyzer) return false;

    pthread_mutex_lock(&analyzer->mutex);

    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            entry->baseline_stats = entry->data.stats;
            entry->has_baseline = true;
            entry = entry->next;
        }
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

void runtime_clear_baseline(runtime_analyzer_t* analyzer,
                            const char* operation_name) {
    if (!analyzer || !operation_name) return;

    pthread_mutex_lock(&analyzer->mutex);

    size_t idx = hash_string(operation_name);
    op_entry_t* entry = analyzer->operations[idx];

    while (entry) {
        if (strcmp(entry->data.name, operation_name) == 0) {
            entry->has_baseline = false;
            break;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&analyzer->mutex);
}

bool runtime_detect_regressions(runtime_analyzer_t* analyzer,
                                runtime_regression_t** regressions,
                                size_t* count) {
    if (!analyzer || !regressions || !count) return false;

    pthread_mutex_lock(&analyzer->mutex);

    // Count operations with baselines
    size_t max_regressions = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            if (entry->has_baseline) {
                max_regressions++;
            }
            entry = entry->next;
        }
    }

    if (max_regressions == 0) {
        *regressions = NULL;
        *count = 0;
        pthread_mutex_unlock(&analyzer->mutex);
        return true;
    }

    runtime_regression_t* results = calloc(max_regressions, sizeof(runtime_regression_t));
    if (!results) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    size_t regression_count = 0;
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            if (entry->has_baseline && entry->data.stats.count >= MIN_SAMPLES_FOR_STATS) {
                double baseline_mean = entry->baseline_stats.mean_ns;
                double current_mean = entry->data.stats.mean_ns;

                if (baseline_mean > 0) {
                    double regression_pct = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

                    // Calculate significance using t-test approximation
                    double pooled_std = sqrt((entry->baseline_stats.variance_ns +
                                              entry->data.stats.variance_ns) / 2.0);
                    double t_stat = 0;
                    if (pooled_std > 0) {
                        t_stat = fabs(current_mean - baseline_mean) /
                                 (pooled_std / sqrt((double)entry->data.stats.count));
                    }

                    // Consider significant if |t| > 2 (approximately 95% confidence)
                    bool significant = (t_stat > 2.0) &&
                                      (fabs(regression_pct) > analyzer->config.regression_threshold);

                    if (significant || fabs(regression_pct) > 50.0) {
                        results[regression_count].operation_name = entry->data.name;
                        results[regression_count].baseline_mean_ns = baseline_mean;
                        results[regression_count].current_mean_ns = current_mean;
                        results[regression_count].regression_percent = regression_pct;
                        results[regression_count].confidence = fmin(1.0, t_stat / 10.0);
                        results[regression_count].is_significant = significant;
                        regression_count++;
                    }
                }
            }
            entry = entry->next;
        }
    }

    *regressions = results;
    *count = regression_count;

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

// ============================================================================
// Bottleneck Analysis
// ============================================================================

bool runtime_analyze_bottlenecks(runtime_analyzer_t* analyzer,
                                 runtime_bottleneck_info_t** bottlenecks,
                                 size_t* count) {
    if (!analyzer || !bottlenecks || !count) return false;

    // Get category statistics
    runtime_stats_t category_stats[RUNTIME_OP_CATEGORY_COUNT];
    for (int i = 0; i < RUNTIME_OP_CATEGORY_COUNT; i++) {
        runtime_get_category_stats(analyzer, (runtime_op_category_t)i, &category_stats[i]);
    }

    // Find dominant categories
    uint64_t total_time = 0;
    for (int i = 0; i < RUNTIME_OP_CATEGORY_COUNT; i++) {
        total_time += category_stats[i].total_ns;
    }

    if (total_time == 0) {
        *bottlenecks = NULL;
        *count = 0;
        return true;
    }

    // Allocate worst case
    runtime_bottleneck_info_t* results = calloc(RUNTIME_OP_CATEGORY_COUNT,
                                                 sizeof(runtime_bottleneck_info_t));
    if (!results) return false;

    size_t bottleneck_count = 0;

    // Analyze each category
    for (int i = 0; i < RUNTIME_OP_CATEGORY_COUNT; i++) {
        double pct = (double)category_stats[i].total_ns / (double)total_time * 100.0;

        if (pct > 30.0) {  // >30% of time in one category is a potential bottleneck
            runtime_bottleneck_info_t* info = &results[bottleneck_count];
            info->severity = pct / 100.0;

            switch ((runtime_op_category_t)i) {
                case RUNTIME_OP_MATRIX_MULTIPLY:
                case RUNTIME_OP_TENSOR_CONTRACTION:
                case RUNTIME_OP_DECOMPOSITION:
                    info->type = BOTTLENECK_CPU_BOUND;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in compute-heavy operations", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Consider GPU acceleration or SIMD optimization");
                    break;

                case RUNTIME_OP_MEMORY_ALLOCATION:
                    info->type = BOTTLENECK_MEMORY_BOUND;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in memory operations", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Use memory pooling or reduce allocations");
                    break;

                case RUNTIME_OP_IO:
                    info->type = BOTTLENECK_IO_BOUND;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in I/O operations", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Consider async I/O or buffering");
                    break;

                case RUNTIME_OP_COMMUNICATION:
                    info->type = BOTTLENECK_NETWORK_BOUND;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in communication", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Reduce message count or use collective operations");
                    break;

                case RUNTIME_OP_SYNCHRONIZATION:
                    info->type = BOTTLENECK_SYNCHRONIZATION;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in synchronization", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Reduce lock contention or use lock-free algorithms");
                    break;

                case RUNTIME_OP_GPU_KERNEL:
                    info->type = BOTTLENECK_GPU_COMPUTE;
                    snprintf(info->description, sizeof(info->description),
                             "%.1f%% of time in GPU kernels", pct);
                    snprintf(info->suggestion, sizeof(info->suggestion),
                             "Optimize kernel occupancy or reduce memory transfers");
                    break;

                default:
                    continue;  // Skip categories without bottleneck classification
            }

            bottleneck_count++;
        }
    }

    *bottlenecks = results;
    *count = bottleneck_count;
    return true;
}

// ============================================================================
// Export and Reporting
// ============================================================================

char* runtime_export_json(runtime_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    // Estimate size needed
    size_t buf_size = 4096 + analyzer->operation_count * 512;
    char* json = malloc(buf_size);
    if (!json) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    char* p = json;
    p += sprintf(p, "{\n  \"operations\": [\n");

    bool first = true;
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            if (!first) {
                p += sprintf(p, ",\n");
            }
            first = false;

            p += sprintf(p,
                "    {\n"
                "      \"name\": \"%s\",\n"
                "      \"category\": \"%s\",\n"
                "      \"count\": %llu,\n"
                "      \"total_ns\": %llu,\n"
                "      \"mean_ns\": %.2f,\n"
                "      \"min_ns\": %llu,\n"
                "      \"max_ns\": %llu,\n"
                "      \"std_dev_ns\": %.2f\n"
                "    }",
                entry->data.name,
                runtime_category_name(entry->data.category),
                (unsigned long long)entry->data.stats.count,
                (unsigned long long)entry->data.stats.total_ns,
                entry->data.stats.mean_ns,
                (unsigned long long)entry->data.stats.min_ns,
                (unsigned long long)entry->data.stats.max_ns,
                entry->data.stats.std_dev_ns);

            entry = entry->next;
        }
    }

    p += sprintf(p, "\n  ]\n}\n");

    pthread_mutex_unlock(&analyzer->mutex);
    return json;
}

bool runtime_export_to_file(runtime_analyzer_t* analyzer,
                            const char* filename) {
    if (!analyzer || !filename) return false;

    char* json = runtime_export_json(analyzer);
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

char* runtime_generate_report(runtime_analyzer_t* analyzer) {
    if (!analyzer) return NULL;

    size_t buf_size = 8192 + analyzer->operation_count * 256;
    char* report = malloc(buf_size);
    if (!report) return NULL;

    pthread_mutex_lock(&analyzer->mutex);

    char* p = report;
    p += sprintf(p, "=== Runtime Analysis Report ===\n\n");

    uint64_t elapsed_ns = runtime_get_timestamp_ns() - analyzer->start_time_ns;
    p += sprintf(p, "Total elapsed time: %.3f ms\n", elapsed_ns / 1000000.0);
    p += sprintf(p, "Operations tracked: %zu\n\n", analyzer->operation_count);

    p += sprintf(p, "Top Operations by Time:\n");
    p += sprintf(p, "%-40s %12s %12s %12s\n", "Operation", "Count", "Mean (µs)", "Total (ms)");
    p += sprintf(p, "%-40s %12s %12s %12s\n", "----------------------------------------",
                 "------------", "------------", "------------");

    // Get sorted operations
    size_t count = 0;
    runtime_operation_t* ops = NULL;

    // Temporarily unlock for the call
    pthread_mutex_unlock(&analyzer->mutex);
    runtime_get_slowest_operations(analyzer, 10, &ops, &count);
    pthread_mutex_lock(&analyzer->mutex);

    if (ops) {
        for (size_t i = 0; i < count; i++) {
            p += sprintf(p, "%-40s %12llu %12.2f %12.2f\n",
                        ops[i].name,
                        (unsigned long long)ops[i].stats.count,
                        ops[i].stats.mean_ns / 1000.0,
                        ops[i].stats.total_ns / 1000000.0);
        }
        free(ops);
    }

    pthread_mutex_unlock(&analyzer->mutex);
    return report;
}

bool runtime_export_trace_json(runtime_analyzer_t* analyzer,
                               const char* filename) {
    if (!analyzer || !filename) return false;

    pthread_mutex_lock(&analyzer->mutex);

    FILE* f = fopen(filename, "w");
    if (!f) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    fprintf(f, "{\"traceEvents\":[\n");

    for (size_t i = 0; i < analyzer->spans.count; i++) {
        runtime_span_t* span = &analyzer->spans.spans[i];
        if (i > 0) fprintf(f, ",\n");

        uint64_t duration_us = (span->end_ns - span->start_ns) / 1000;
        uint64_t start_us = span->start_ns / 1000;

        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\","
                   "\"ts\":%llu,\"dur\":%llu,\"pid\":1,\"tid\":%u}",
                span->name,
                runtime_category_name(span->category),
                (unsigned long long)start_us,
                (unsigned long long)duration_us,
                span->span_id);
    }

    fprintf(f, "\n]}\n");
    fclose(f);

    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

bool runtime_export_flamegraph(runtime_analyzer_t* analyzer,
                               const char* filename) {
    // Folded stacks format for flamegraph.pl
    if (!analyzer || !filename) return false;

    pthread_mutex_lock(&analyzer->mutex);

    FILE* f = fopen(filename, "w");
    if (!f) {
        pthread_mutex_unlock(&analyzer->mutex);
        return false;
    }

    // Output each operation as a stack sample
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        op_entry_t* entry = analyzer->operations[i];
        while (entry) {
            // Format: stack;stack;... count
            fprintf(f, "%s;%s %llu\n",
                    runtime_category_name(entry->data.category),
                    entry->data.name,
                    (unsigned long long)(entry->data.stats.total_ns / 1000));  // microseconds
            entry = entry->next;
        }
    }

    fclose(f);
    pthread_mutex_unlock(&analyzer->mutex);
    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* runtime_category_name(runtime_op_category_t category) {
    static const char* names[] = {
        "gate_application",
        "state_preparation",
        "measurement",
        "tensor_contraction",
        "matrix_multiply",
        "fft",
        "decomposition",
        "error_correction",
        "memory_allocation",
        "io",
        "communication",
        "gpu_kernel",
        "synchronization",
        "custom"
    };

    if (category >= 0 && category < RUNTIME_OP_CATEGORY_COUNT) {
        return names[category];
    }
    return "unknown";
}

const char* runtime_trend_name(runtime_trend_t trend) {
    static const char* names[] = {
        "improving",
        "stable",
        "degrading",
        "volatile",
        "unknown"
    };

    if (trend >= 0 && trend <= RUNTIME_TREND_UNKNOWN) {
        return names[trend];
    }
    return "unknown";
}

const char* runtime_bottleneck_name(runtime_bottleneck_t type) {
    static const char* names[] = {
        "none",
        "cpu_bound",
        "memory_bound",
        "memory_latency",
        "gpu_compute",
        "gpu_memory",
        "io_bound",
        "network_bound",
        "synchronization",
        "cache_miss"
    };

    if (type >= 0 && type <= BOTTLENECK_CACHE_MISS) {
        return names[type];
    }
    return "unknown";
}

char* runtime_format_duration(uint64_t duration_ns) {
    char* buf = malloc(64);
    if (!buf) return NULL;

    if (duration_ns < 1000) {
        sprintf(buf, "%llu ns", (unsigned long long)duration_ns);
    } else if (duration_ns < 1000000) {
        sprintf(buf, "%.2f µs", duration_ns / 1000.0);
    } else if (duration_ns < 1000000000) {
        sprintf(buf, "%.2f ms", duration_ns / 1000000.0);
    } else {
        sprintf(buf, "%.2f s", duration_ns / 1000000000.0);
    }

    return buf;
}

void runtime_free_operations(runtime_operation_t* operations, size_t count) {
    (void)count;
    free(operations);
}

void runtime_free_regressions(runtime_regression_t* regressions, size_t count) {
    (void)count;
    free(regressions);
}

void runtime_free_bottlenecks(runtime_bottleneck_info_t* bottlenecks, size_t count) {
    (void)count;
    free(bottlenecks);
}

void runtime_free_strings(char** strings, size_t count) {
    if (!strings) return;
    for (size_t i = 0; i < count; i++) {
        free(strings[i]);
    }
    free(strings);
}

bool runtime_get_optimization_suggestions(runtime_analyzer_t* analyzer,
                                          char*** suggestions,
                                          size_t* count) {
    if (!analyzer || !suggestions || !count) return false;

    runtime_bottleneck_info_t* bottlenecks = NULL;
    size_t bottleneck_count = 0;

    if (!runtime_analyze_bottlenecks(analyzer, &bottlenecks, &bottleneck_count)) {
        return false;
    }

    if (bottleneck_count == 0) {
        *suggestions = NULL;
        *count = 0;
        return true;
    }

    *suggestions = calloc(bottleneck_count, sizeof(char*));
    if (!*suggestions) {
        free(bottlenecks);
        return false;
    }

    for (size_t i = 0; i < bottleneck_count; i++) {
        (*suggestions)[i] = strdup(bottlenecks[i].suggestion);
    }

    *count = bottleneck_count;
    free(bottlenecks);
    return true;
}
