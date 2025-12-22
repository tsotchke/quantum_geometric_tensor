#include "quantum_geometric/distributed/suggestion_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <net/if_dl.h>
#else
#include <sys/sysinfo.h>
#endif

// Static variables for throughput/latency tracking
static struct {
    struct timeval last_sample_time;
    uint64_t operations_count;
    double accumulated_latency;
    size_t latency_samples;
    uint64_t last_net_bytes;
    bool initialized;
} g_metrics_state = {0};

// Internal suggestion tracker implementation
struct SuggestionTrackerImpl {
    // Tracked suggestions
    SuggestionTrackerEntry* entries;
    size_t num_entries;
    size_t capacity;

    // Baseline metrics for comparison
    SuggestionPerformanceMetrics baseline;
    SuggestionPerformanceMetrics current;

    // Effectiveness analyzer
    EffectivenessAnalyzer* analyzer;

    // Configuration
    SuggestionTrackerConfig config;

    // Statistics
    size_t total_tracked;
    size_t total_implemented;
    size_t total_reverted;
    double avg_effectiveness;
};

// Forward declarations for static functions
static void capture_current_metrics(SuggestionPerformanceMetrics* metrics);
static double compute_performance_improvement(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current);
static double compute_resource_savings(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current);
static double compute_stability_impact(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current);
static void update_effectiveness_score(SuggestionTrackerEntry* entry);
static void update_confidence_score(SuggestionTrackerEntry* entry);
static SuggestionTrackerEntry* find_entry(
    SuggestionTracker* tracker,
    const Suggestion* suggestion);
static void remove_oldest_entry(SuggestionTracker* tracker);
static void cleanup_entry(SuggestionTrackerEntry* entry);
static int compare_effectiveness_desc(const void* a, const void* b);
static int compare_effectiveness_asc(const void* a, const void* b);
static bool suggestions_equal(const Suggestion* a, const Suggestion* b);

// Initialize suggestion tracker
SuggestionTracker* init_suggestion_tracker(const SuggestionTrackerConfig* config) {
    SuggestionTracker* tracker = calloc(1, sizeof(SuggestionTracker));
    if (!tracker) return NULL;

    // Set default configuration if not provided
    if (config) {
        tracker->config = *config;
    } else {
        tracker->config.max_tracked = TRACKER_MAX_SUGGESTIONS;
        tracker->config.effectiveness_window_sec = TRACKER_EFFECTIVENESS_WINDOW;
        tracker->config.min_samples_for_analysis = TRACKER_MIN_SAMPLES;
        tracker->config.confidence_threshold = TRACKER_CONFIDENCE_THRESHOLD;
        tracker->config.enable_auto_reversion = false;
        tracker->config.reversion_threshold = -0.1;  // 10% degradation threshold
    }

    // Allocate entries array
    tracker->capacity = tracker->config.max_tracked;
    tracker->entries = calloc(tracker->capacity, sizeof(SuggestionTrackerEntry));
    if (!tracker->entries) {
        free(tracker);
        return NULL;
    }

    // Initialize effectiveness analyzer
    tracker->analyzer = init_effectiveness_analyzer();

    // Capture initial baseline
    capture_current_metrics(&tracker->baseline);
    tracker->current = tracker->baseline;

    // Initialize statistics
    tracker->num_entries = 0;
    tracker->total_tracked = 0;
    tracker->total_implemented = 0;
    tracker->total_reverted = 0;
    tracker->avg_effectiveness = 0.0;

    return tracker;
}

// Capture current system performance metrics
static void capture_current_metrics(SuggestionPerformanceMetrics* metrics) {
    if (!metrics) return;

    memset(metrics, 0, sizeof(SuggestionPerformanceMetrics));
    metrics->timestamp = time(NULL);

#ifdef __APPLE__
    // Get CPU usage on macOS
    host_cpu_load_info_data_t cpu_info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                       (host_info_t)&cpu_info, &count) == KERN_SUCCESS) {
        natural_t total = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total += cpu_info.cpu_ticks[i];
        }
        if (total > 0) {
            natural_t idle = cpu_info.cpu_ticks[CPU_STATE_IDLE];
            metrics->cpu_usage = 1.0 - ((double)idle / (double)total);
        }
    }

    // Get memory usage on macOS
    vm_size_t page_size;
    vm_statistics64_data_t vm_stats;
    mach_port_t mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);

    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        uint64_t used = ((uint64_t)vm_stats.active_count +
                        (uint64_t)vm_stats.wire_count) * page_size;
        uint64_t total_mem = ((uint64_t)vm_stats.active_count +
                             (uint64_t)vm_stats.inactive_count +
                             (uint64_t)vm_stats.wire_count +
                             (uint64_t)vm_stats.free_count) * page_size;
        if (total_mem > 0) {
            metrics->memory_usage = (double)used / (double)total_mem;
        }
    }
#else
    // Get CPU usage on Linux (simplified)
    FILE* fp = fopen("/proc/stat", "r");
    if (fp) {
        unsigned long user, nice, system, idle;
        if (fscanf(fp, "cpu %lu %lu %lu %lu", &user, &nice, &system, &idle) == 4) {
            unsigned long total = user + nice + system + idle;
            if (total > 0) {
                metrics->cpu_usage = 1.0 - ((double)idle / (double)total);
            }
        }
        fclose(fp);
    }

    // Get memory usage on Linux
    struct sysinfo info;
    if (sysinfo(&info) == 0 && info.totalram > 0) {
        metrics->memory_usage = 1.0 - ((double)info.freeram / (double)info.totalram);
    }
#endif

    // Calculate throughput (operations per second)
    struct timeval now;
    gettimeofday(&now, NULL);

    if (!g_metrics_state.initialized) {
        g_metrics_state.last_sample_time = now;
        g_metrics_state.operations_count = 0;
        g_metrics_state.accumulated_latency = 0.0;
        g_metrics_state.latency_samples = 0;
        g_metrics_state.last_net_bytes = 0;
        g_metrics_state.initialized = true;
        metrics->throughput = 0.0;
        metrics->latency = 0.0;
    } else {
        double elapsed = (double)(now.tv_sec - g_metrics_state.last_sample_time.tv_sec) +
                        (double)(now.tv_usec - g_metrics_state.last_sample_time.tv_usec) / 1000000.0;

        if (elapsed > 0.001) {  // At least 1ms elapsed
            metrics->throughput = (double)g_metrics_state.operations_count / elapsed;
            g_metrics_state.operations_count = 0;
            g_metrics_state.last_sample_time = now;
        }

        // Average latency from samples
        if (g_metrics_state.latency_samples > 0) {
            metrics->latency = g_metrics_state.accumulated_latency / (double)g_metrics_state.latency_samples;
            g_metrics_state.accumulated_latency = 0.0;
            g_metrics_state.latency_samples = 0;
        } else {
            metrics->latency = 0.0;
        }
    }

    // GPU usage
    metrics->gpu_usage = 0.0;
#ifdef __APPLE__
    // Query Metal GPU usage via IOKit
    io_iterator_t iterator;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AppleGPUWrangler"),
                                     &iterator) == KERN_SUCCESS) {
        io_service_t device;
        while ((device = IOIteratorNext(iterator)) != 0) {
            CFMutableDictionaryRef properties = NULL;
            if (IORegistryEntryCreateCFProperties(device, &properties,
                                                   kCFAllocatorDefault, 0) == KERN_SUCCESS) {
                CFNumberRef utilization = CFDictionaryGetValue(properties, CFSTR("PerformanceStatistics"));
                if (utilization) {
                    // GPU utilization is tracked but structure varies by GPU
                    // Set a reasonable default based on system activity
                    metrics->gpu_usage = metrics->cpu_usage * 0.5;  // Estimate based on CPU
                }
                CFRelease(properties);
            }
            IOObjectRelease(device);
        }
        IOObjectRelease(iterator);
    }
#else
    // Linux: try reading from nvidia-smi or sysfs
    FILE* gpu_fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (gpu_fp) {
        int gpu_util = 0;
        if (fscanf(gpu_fp, "%d", &gpu_util) == 1) {
            metrics->gpu_usage = (double)gpu_util / 100.0;
        }
        pclose(gpu_fp);
    }
#endif

    // Network bandwidth
    metrics->network_bandwidth = 0.0;
#ifdef __APPLE__
    struct ifaddrs* ifaddr = NULL;
    if (getifaddrs(&ifaddr) == 0) {
        uint64_t total_bytes = 0;
        for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_LINK) {
                struct if_data* if_data = (struct if_data*)ifa->ifa_data;
                if (if_data) {
                    total_bytes += if_data->ifi_ibytes + if_data->ifi_obytes;
                }
            }
        }
        freeifaddrs(ifaddr);

        if (g_metrics_state.last_net_bytes > 0 && total_bytes > g_metrics_state.last_net_bytes) {
            double elapsed = (double)(now.tv_sec - g_metrics_state.last_sample_time.tv_sec) +
                            (double)(now.tv_usec - g_metrics_state.last_sample_time.tv_usec) / 1000000.0;
            if (elapsed > 0.001) {
                metrics->network_bandwidth = (double)(total_bytes - g_metrics_state.last_net_bytes) / elapsed;
            }
        }
        g_metrics_state.last_net_bytes = total_bytes;
    }
#else
    // Linux: read from /proc/net/dev
    FILE* net_fp = fopen("/proc/net/dev", "r");
    if (net_fp) {
        char line[512];
        uint64_t total_bytes = 0;
        while (fgets(line, sizeof(line), net_fp)) {
            char iface[32];
            uint64_t rx_bytes, tx_bytes;
            if (sscanf(line, " %31[^:]: %lu %*u %*u %*u %*u %*u %*u %*u %lu",
                      iface, &rx_bytes, &tx_bytes) == 3) {
                total_bytes += rx_bytes + tx_bytes;
            }
        }
        fclose(net_fp);

        if (g_metrics_state.last_net_bytes > 0 && total_bytes > g_metrics_state.last_net_bytes) {
            double elapsed = (double)(now.tv_sec - g_metrics_state.last_sample_time.tv_sec) +
                            (double)(now.tv_usec - g_metrics_state.last_sample_time.tv_usec) / 1000000.0;
            if (elapsed > 0.001) {
                metrics->network_bandwidth = (double)(total_bytes - g_metrics_state.last_net_bytes) / elapsed;
            }
        }
        g_metrics_state.last_net_bytes = total_bytes;
    }
#endif
}

// Record an operation for throughput tracking
void suggestion_tracker_record_operation(double latency_ms) {
    g_metrics_state.operations_count++;
    if (latency_ms > 0.0) {
        g_metrics_state.accumulated_latency += latency_ms;
        g_metrics_state.latency_samples++;
    }
}

// Compute performance improvement
static double compute_performance_improvement(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current) {

    if (!baseline || !current) return 0.0;

    double improvement = 0.0;
    int count = 0;

    // Throughput improvement (higher is better)
    if (baseline->throughput > 0) {
        improvement += (current->throughput - baseline->throughput) / baseline->throughput;
        count++;
    }

    // Latency improvement (lower is better)
    if (baseline->latency > 0) {
        improvement += (baseline->latency - current->latency) / baseline->latency;
        count++;
    }

    return count > 0 ? improvement / count : 0.0;
}

// Compute resource savings
static double compute_resource_savings(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current) {

    if (!baseline || !current) return 0.0;

    double savings = 0.0;
    int count = 0;

    // CPU savings (lower usage is better)
    if (baseline->cpu_usage > 0) {
        savings += (baseline->cpu_usage - current->cpu_usage) / baseline->cpu_usage;
        count++;
    }

    // Memory savings (lower usage is better)
    if (baseline->memory_usage > 0) {
        savings += (baseline->memory_usage - current->memory_usage) / baseline->memory_usage;
        count++;
    }

    // GPU savings
    if (baseline->gpu_usage > 0) {
        savings += (baseline->gpu_usage - current->gpu_usage) / baseline->gpu_usage;
        count++;
    }

    return count > 0 ? savings / count : 0.0;
}

// Compute stability impact
static double compute_stability_impact(
    const SuggestionPerformanceMetrics* baseline,
    const SuggestionPerformanceMetrics* current) {

    if (!baseline || !current) return 0.0;

    // Stability is measured by variance in metrics
    // Positive means more stable, negative means less stable
    // For simplicity, we use latency variance as a proxy

    double baseline_variance = baseline->latency * 0.1;  // Assumed 10% variance
    double current_variance = current->latency * 0.1;

    if (baseline_variance > 0) {
        return (baseline_variance - current_variance) / baseline_variance;
    }

    return 0.0;
}

// Update effectiveness score based on impacts
static void update_effectiveness_score(SuggestionTrackerEntry* entry) {
    if (!entry || entry->num_impacts == 0) return;

    double total_effectiveness = 0.0;
    double weight_sum = 0.0;

    // Use exponential weighting - more recent impacts count more
    for (size_t i = 0; i < entry->num_impacts; i++) {
        double age_factor = exp(-0.1 * (double)(entry->num_impacts - 1 - i));

        // Combine impact metrics
        double impact = entry->impacts[i].performance_improvement * 0.5 +
                       entry->impacts[i].resource_savings * 0.3 +
                       entry->impacts[i].stability_impact * 0.2;

        total_effectiveness += impact * age_factor;
        weight_sum += age_factor;
    }

    entry->cumulative_effectiveness = weight_sum > 0 ?
        total_effectiveness / weight_sum : 0.0;
}

// Update confidence score based on number of samples
static void update_confidence_score(SuggestionTrackerEntry* entry) {
    if (!entry) return;

    // Confidence increases with more samples up to a maximum
    double sample_factor = 1.0 - exp(-0.1 * (double)entry->num_impacts);

    // Consider consistency of impacts
    if (entry->num_impacts >= 2) {
        double variance = 0.0;
        double mean = entry->cumulative_effectiveness;

        for (size_t i = 0; i < entry->num_impacts; i++) {
            double impact = entry->impacts[i].performance_improvement * 0.5 +
                           entry->impacts[i].resource_savings * 0.3 +
                           entry->impacts[i].stability_impact * 0.2;
            double diff = impact - mean;
            variance += diff * diff;
        }
        variance /= entry->num_impacts;

        // Lower variance means higher confidence
        double consistency_factor = exp(-variance);
        entry->confidence_score = sample_factor * consistency_factor;
    } else {
        entry->confidence_score = sample_factor * 0.5;  // Low confidence with few samples
    }
}

// Check if two suggestions are equal
static bool suggestions_equal(const Suggestion* a, const Suggestion* b) {
    if (!a || !b) return false;
    if (a->type != b->type) return false;
    if (a->description && b->description) {
        return strcmp(a->description, b->description) == 0;
    }
    return a->description == b->description;
}

// Find entry for a suggestion
static SuggestionTrackerEntry* find_entry(
    SuggestionTracker* tracker,
    const Suggestion* suggestion) {

    if (!tracker || !suggestion) return NULL;

    for (size_t i = 0; i < tracker->num_entries; i++) {
        if (suggestions_equal(&tracker->entries[i].suggestion, suggestion)) {
            return &tracker->entries[i];
        }
    }

    return NULL;
}

// Remove oldest entry to make room
static void remove_oldest_entry(SuggestionTracker* tracker) {
    if (!tracker || tracker->num_entries == 0) return;

    // Find oldest entry (by implementation time or tracking time)
    size_t oldest_idx = 0;
    time_t oldest_time = tracker->entries[0].status.implementation_time;
    if (oldest_time == 0) {
        oldest_time = tracker->entries[0].suggestion.timestamp;
    }

    for (size_t i = 1; i < tracker->num_entries; i++) {
        time_t entry_time = tracker->entries[i].status.implementation_time;
        if (entry_time == 0) {
            entry_time = tracker->entries[i].suggestion.timestamp;
        }

        if (entry_time < oldest_time) {
            oldest_time = entry_time;
            oldest_idx = i;
        }
    }

    // Clean up and remove
    cleanup_entry(&tracker->entries[oldest_idx]);

    // Shift remaining entries
    for (size_t i = oldest_idx; i < tracker->num_entries - 1; i++) {
        tracker->entries[i] = tracker->entries[i + 1];
    }

    tracker->num_entries--;
}

// Clean up an entry
static void cleanup_entry(SuggestionTrackerEntry* entry) {
    if (!entry) return;

    free(entry->suggestion.description);
    entry->suggestion.description = NULL;

    free(entry->status.implementation_details);
    entry->status.implementation_details = NULL;

    free(entry->status.reversion_reason);
    entry->status.reversion_reason = NULL;

    free(entry->impacts);
    entry->impacts = NULL;
    entry->num_impacts = 0;
    entry->impact_capacity = 0;
}

// Comparison function for sorting by effectiveness (descending)
static int compare_effectiveness_desc(const void* a, const void* b) {
    const SuggestionTrackerEntry* ea = (const SuggestionTrackerEntry*)a;
    const SuggestionTrackerEntry* eb = (const SuggestionTrackerEntry*)b;

    if (eb->cumulative_effectiveness > ea->cumulative_effectiveness) return 1;
    if (eb->cumulative_effectiveness < ea->cumulative_effectiveness) return -1;
    return 0;
}

// Comparison function for sorting by effectiveness (ascending)
static int compare_effectiveness_asc(const void* a, const void* b) {
    return -compare_effectiveness_desc(a, b);
}

// Start tracking a suggestion
void tracker_track_suggestion(
    SuggestionTracker* tracker,
    const Suggestion* suggestion) {

    if (!tracker || !suggestion) return;

    // Check if already tracking
    if (find_entry(tracker, suggestion)) return;

    // Make room if needed
    if (tracker->num_entries >= tracker->capacity) {
        remove_oldest_entry(tracker);
    }

    // Create new entry
    SuggestionTrackerEntry* entry = &tracker->entries[tracker->num_entries++];
    memset(entry, 0, sizeof(SuggestionTrackerEntry));

    // Copy suggestion
    entry->suggestion = *suggestion;
    if (suggestion->description) {
        entry->suggestion.description = strdup(suggestion->description);
    }

    // Initialize status
    entry->status.is_implemented = false;
    entry->status.implementation_time = 0;
    entry->status.implementation_details = NULL;
    entry->status.is_reverted = false;
    entry->status.reversion_reason = NULL;

    // Initialize impacts array
    entry->impact_capacity = TRACKER_MAX_IMPACTS;
    entry->impacts = calloc(entry->impact_capacity, sizeof(SuggestionImpact));
    entry->num_impacts = 0;

    entry->cumulative_effectiveness = 0.0;
    entry->confidence_score = 0.0;

    tracker->total_tracked++;
}

// Record suggestion implementation
void tracker_record_implementation(
    SuggestionTracker* tracker,
    const Suggestion* suggestion,
    const char* details) {

    if (!tracker || !suggestion) return;

    SuggestionTrackerEntry* entry = find_entry(tracker, suggestion);
    if (!entry) {
        // Start tracking if not already
        tracker_track_suggestion(tracker, suggestion);
        entry = find_entry(tracker, suggestion);
        if (!entry) return;
    }

    // Update status
    entry->status.is_implemented = true;
    entry->status.implementation_time = time(NULL);

    free(entry->status.implementation_details);
    entry->status.implementation_details = details ? strdup(details) : NULL;

    // Reset impacts for fresh measurement
    entry->num_impacts = 0;

    // Capture new baseline
    tracker_update_baseline(tracker);

    tracker->total_implemented++;
}

// Measure current impact
void tracker_measure_impact(
    SuggestionTracker* tracker,
    const Suggestion* suggestion) {

    if (!tracker || !suggestion) return;

    SuggestionTrackerEntry* entry = find_entry(tracker, suggestion);
    if (!entry || !entry->status.is_implemented) return;

    // Capture current metrics
    capture_current_metrics(&tracker->current);

    // Compute impact
    SuggestionImpact impact;
    impact.performance_improvement = compute_performance_improvement(
        &tracker->baseline, &tracker->current);
    impact.resource_savings = compute_resource_savings(
        &tracker->baseline, &tracker->current);
    impact.stability_impact = compute_stability_impact(
        &tracker->baseline, &tracker->current);
    impact.measurement_time = time(NULL);

    // Store impact (circular buffer if full)
    if (entry->num_impacts >= entry->impact_capacity) {
        // Shift impacts to make room
        memmove(entry->impacts, entry->impacts + 1,
               (entry->impact_capacity - 1) * sizeof(SuggestionImpact));
        entry->num_impacts = entry->impact_capacity - 1;
    }

    entry->impacts[entry->num_impacts++] = impact;

    // Update scores
    update_effectiveness_score(entry);
    update_confidence_score(entry);
}

// Analyze effectiveness of all tracked suggestions
void tracker_analyze_effectiveness(
    SuggestionTracker* tracker,
    EffectivenessReport* report) {

    if (!tracker || !report) return;

    // Reset report
    memset(report, 0, sizeof(EffectivenessReport));

    // Analyze each tracked suggestion
    for (size_t i = 0; i < tracker->num_entries; i++) {
        SuggestionTrackerEntry* entry = &tracker->entries[i];

        if (!entry->status.is_implemented ||
            entry->num_impacts < tracker->config.min_samples_for_analysis) {
            continue;
        }

        // Create a TrackedSuggestion for the analyzer
        TrackedSuggestion tracked;
        memset(&tracked, 0, sizeof(TrackedSuggestion));

        // Map our suggestion to EffOptimizationSuggestion
        tracked.suggestion.type = (EffSuggestionType)entry->suggestion.type;
        tracked.suggestion.priority = entry->suggestion.impact_score;
        tracked.suggestion.expected_improvement = entry->suggestion.impact_score;
        tracked.suggestion.description = entry->suggestion.description;
        tracked.suggestion.context = NULL;

        // Map impacts
        tracked.capacity = entry->num_impacts;
        tracked.num_impacts = entry->num_impacts;
        tracked.impacts = calloc(entry->num_impacts, sizeof(ImpactMeasurement));

        if (tracked.impacts) {
            for (size_t j = 0; j < entry->num_impacts; j++) {
                tracked.impacts[j].timestamp = entry->impacts[j].measurement_time;
                tracked.impacts[j].value = entry->impacts[j].performance_improvement;
                tracked.impacts[j].metric = METRIC_THROUGHPUT;
                tracked.impacts[j].is_positive =
                    entry->impacts[j].performance_improvement > 0;
            }
        }

        tracked.cumulative_impact = entry->cumulative_effectiveness;
        tracked.is_active = entry->status.is_implemented &&
                           !entry->status.is_reverted;

        // Use analyzer for detailed analysis
        if (tracker->analyzer) {
            analyze_impact_patterns(tracker->analyzer, &tracked, report);
            check_for_regressions(tracker->analyzer, &tracked, report);
        }

        free(tracked.impacts);
    }

    // Compute overall effectiveness
    double total_eff = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < tracker->num_entries; i++) {
        if (tracker->entries[i].status.is_implemented &&
            tracker->entries[i].confidence_score >= tracker->config.confidence_threshold) {
            total_eff += tracker->entries[i].cumulative_effectiveness;
            count++;
        }
    }
    report->overall_effectiveness = count > 0 ? total_eff / count : 0.0;
}

// Record suggestion reversion
void tracker_record_reversion(
    SuggestionTracker* tracker,
    const Suggestion* suggestion,
    const char* reason) {

    if (!tracker || !suggestion) return;

    SuggestionTrackerEntry* entry = find_entry(tracker, suggestion);
    if (!entry) return;

    // Update status
    entry->status.is_implemented = false;
    entry->status.is_reverted = true;

    free(entry->status.reversion_reason);
    entry->status.reversion_reason = reason ? strdup(reason) : NULL;

    // Update confidence (reverted suggestions have lower confidence)
    entry->confidence_score *= 0.5;

    tracker->total_reverted++;
}

// Get tracked suggestion entry
const SuggestionTrackerEntry* tracker_get_entry(
    const SuggestionTracker* tracker,
    const Suggestion* suggestion) {

    if (!tracker || !suggestion) return NULL;

    for (size_t i = 0; i < tracker->num_entries; i++) {
        if (suggestions_equal(&tracker->entries[i].suggestion, suggestion)) {
            return &tracker->entries[i];
        }
    }

    return NULL;
}

// Get number of tracked suggestions
size_t tracker_get_count(const SuggestionTracker* tracker) {
    return tracker ? tracker->num_entries : 0;
}

// Get suggestions sorted by effectiveness
void tracker_get_by_effectiveness(
    const SuggestionTracker* tracker,
    SuggestionTrackerEntry** entries,
    size_t* num_entries,
    bool descending) {

    if (!tracker || !entries || !num_entries) return;

    *num_entries = tracker->num_entries;
    if (*num_entries == 0) {
        *entries = NULL;
        return;
    }

    // Create sorted copy
    *entries = calloc(*num_entries, sizeof(SuggestionTrackerEntry));
    if (!*entries) {
        *num_entries = 0;
        return;
    }

    memcpy(*entries, tracker->entries, *num_entries * sizeof(SuggestionTrackerEntry));

    qsort(*entries, *num_entries, sizeof(SuggestionTrackerEntry),
          descending ? compare_effectiveness_desc : compare_effectiveness_asc);
}

// Update baseline metrics
void tracker_update_baseline(SuggestionTracker* tracker) {
    if (!tracker) return;
    capture_current_metrics(&tracker->baseline);
}

// Check if suggestion should be reverted
bool tracker_should_revert(
    const SuggestionTracker* tracker,
    const Suggestion* suggestion) {

    if (!tracker || !suggestion) return false;
    if (!tracker->config.enable_auto_reversion) return false;

    const SuggestionTrackerEntry* entry = tracker_get_entry(tracker, suggestion);
    if (!entry) return false;

    // Check if we have enough samples
    if (entry->num_impacts < tracker->config.min_samples_for_analysis) {
        return false;
    }

    // Check if effectiveness is below threshold
    if (entry->cumulative_effectiveness < tracker->config.reversion_threshold &&
        entry->confidence_score >= tracker->config.confidence_threshold) {
        return true;
    }

    return false;
}

// Clean up suggestion tracker
void cleanup_suggestion_tracker(SuggestionTracker* tracker) {
    if (!tracker) return;

    // Clean up all entries
    for (size_t i = 0; i < tracker->num_entries; i++) {
        cleanup_entry(&tracker->entries[i]);
    }
    free(tracker->entries);

    // Clean up analyzer
    if (tracker->analyzer) {
        cleanup_effectiveness_analyzer(tracker->analyzer);
    }

    free(tracker);
}
