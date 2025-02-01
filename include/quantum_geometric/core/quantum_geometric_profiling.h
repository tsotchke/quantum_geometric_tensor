#ifndef QUANTUM_GEOMETRIC_PROFILING_H
#define QUANTUM_GEOMETRIC_PROFILING_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include <time.h>

// Maximum number of profiling spans
#define QGT_MAX_PROFILING_SPANS 1024
#define QGT_MAX_NAME_LENGTH 256

// Profiling categories
typedef enum {
    PROFILING_CATEGORY_NONE = 0,
    PROFILING_CATEGORY_COMPUTATION = 1,
    PROFILING_CATEGORY_MEMORY = 2,
    PROFILING_CATEGORY_IO = 3,
    PROFILING_CATEGORY_NETWORK = 4
} profiling_category_t;

// Profiling span structure
typedef struct {
    char name[QGT_MAX_NAME_LENGTH];
    profiling_category_t category;
    clock_t start_time;
    clock_t end_time;
    double duration;
} profiling_span_t;

// Profiling context structure
typedef struct {
    profiling_span_t spans[QGT_MAX_PROFILING_SPANS];
    size_t active_spans;
    size_t total_spans;
} profiling_context_t;

// Profiling statistics structure
typedef struct {
    double total_time;
    double computation_time;
    double memory_time;
    double io_time;
    double network_time;
    double other_time;
    size_t span_count;
} profiling_stats_t;

// Initialize profiling system
qgt_error_t geometric_init_profiling(void);

// Cleanup profiling system
void geometric_cleanup_profiling(void);

// Start profiling span
profiling_span_t* geometric_start_span(const char* name, profiling_category_t category);

// End profiling span
void geometric_end_span(profiling_span_t* span);

// Get profiling statistics
void geometric_get_profiling_stats(profiling_stats_t* stats);

// Reset profiling data
qgt_error_t geometric_reset_profiling(void);

// Convenience macros for profiling
#define QGT_PROFILE_START(name, category) \
    profiling_span_t* span = geometric_start_span(name, category)

#define QGT_PROFILE_END() \
    geometric_end_span(span)

#define QGT_PROFILE_SCOPE(name, category) \
    profiling_span_t* span = geometric_start_span(name, category); \
    __attribute__((cleanup(geometric_end_span))) profiling_span_t* _cleanup_span __attribute__((unused)) = span

#endif // QUANTUM_GEOMETRIC_PROFILING_H
