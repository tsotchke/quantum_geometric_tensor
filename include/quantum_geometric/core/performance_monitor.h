#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <stdint.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Hardware performance counters
uint64_t get_page_faults(void);
uint64_t get_cache_misses(void);
uint64_t get_tlb_misses(void);

// Initialize performance monitoring
void init_performance_monitor(void);

// Get current performance metrics
PerformanceMetrics get_performance_metrics(void);

// Update performance metrics
void update_performance_metrics(PerformanceMetrics* metrics);

// Reset performance counters
void reset_performance_counters(void);

// Clean up performance monitoring
void cleanup_performance_monitor(void);

#endif // PERFORMANCE_MONITOR_H
