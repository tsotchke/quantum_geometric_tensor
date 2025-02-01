#include "quantum_geometric/core/quantum_geometric_profiling.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Global profiling context
static profiling_context_t* global_profiling_context = NULL;

// Initialize profiling system
qgt_error_t geometric_init_profiling(void) {
    if (global_profiling_context) {
        return QGT_ERROR_ALREADY_INITIALIZED;
    }
    
    global_profiling_context = (profiling_context_t*)calloc(1, sizeof(profiling_context_t));
    if (!global_profiling_context) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Initialize profiling data structures
    global_profiling_context->active_spans = 0;
    global_profiling_context->total_spans = 0;
    
    return QGT_SUCCESS;
}

// Cleanup profiling system
void geometric_cleanup_profiling(void) {
    if (global_profiling_context) {
        free(global_profiling_context);
        global_profiling_context = NULL;
    }
}

// Start profiling span
profiling_span_t* geometric_start_span(const char* name, profiling_category_t category) {
    if (!global_profiling_context || !name) {
        return NULL;
    }
    
    if (global_profiling_context->active_spans >= QGT_MAX_PROFILING_SPANS) {
        return NULL;
    }
    
    profiling_span_t* span = &global_profiling_context->spans[global_profiling_context->active_spans++];
    span->category = category;
    strncpy(span->name, name, QGT_MAX_NAME_LENGTH - 1);
    span->name[QGT_MAX_NAME_LENGTH - 1] = '\0';
    span->start_time = clock();
    span->end_time = 0;
    span->duration = 0;
    
    return span;
}

// End profiling span
void geometric_end_span(profiling_span_t* span) {
    if (!span) {
        return;
    }
    
    span->end_time = clock();
    span->duration = (double)(span->end_time - span->start_time) / CLOCKS_PER_SEC;
    global_profiling_context->active_spans--;
    global_profiling_context->total_spans++;
}

// Get profiling statistics
void geometric_get_profiling_stats(profiling_stats_t* stats) {
    if (!stats || !global_profiling_context) {
        return;
    }
    
    memset(stats, 0, sizeof(profiling_stats_t));
    
    // Compute statistics per category
    for (size_t i = 0; i < global_profiling_context->total_spans; i++) {
        profiling_span_t* span = &global_profiling_context->spans[i];
        stats->total_time += span->duration;
        
        switch (span->category) {
            case PROFILING_CATEGORY_COMPUTATION:
                stats->computation_time += span->duration;
                break;
                
            case PROFILING_CATEGORY_MEMORY:
                stats->memory_time += span->duration;
                break;
                
            case PROFILING_CATEGORY_IO:
                stats->io_time += span->duration;
                break;
                
            case PROFILING_CATEGORY_NETWORK:
                stats->network_time += span->duration;
                break;
                
            default:
                stats->other_time += span->duration;
                break;
        }
    }
    
    stats->span_count = global_profiling_context->total_spans;
}

// Reset profiling data
qgt_error_t geometric_reset_profiling(void) {
    if (!global_profiling_context) {
        return QGT_ERROR_NOT_INITIALIZED;
    }
    
    global_profiling_context->active_spans = 0;
    global_profiling_context->total_spans = 0;
    
    return QGT_SUCCESS;
}
