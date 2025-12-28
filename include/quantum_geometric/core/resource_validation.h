/**
 * @file resource_validation.h
 * @brief Resource Validation for Quantum Operations
 *
 * Provides resource validation including:
 * - Memory allocation validation
 * - Handle validity checking
 * - Resource limit enforcement
 * - Lifecycle tracking
 * - Leak detection
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef RESOURCE_VALIDATION_H
#define RESOURCE_VALIDATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define RESOURCE_VALIDATION_MAX_TRACKED 65536
#define RESOURCE_VALIDATION_MAX_NAME 128

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    RESOURCE_TYPE_MEMORY,
    RESOURCE_TYPE_HANDLE,
    RESOURCE_TYPE_FILE,
    RESOURCE_TYPE_SOCKET,
    RESOURCE_TYPE_THREAD,
    RESOURCE_TYPE_MUTEX,
    RESOURCE_TYPE_GPU_MEMORY,
    RESOURCE_TYPE_QUANTUM_STATE,
    RESOURCE_TYPE_CUSTOM
} resource_type_t;

typedef enum {
    RESOURCE_STATE_UNALLOCATED,
    RESOURCE_STATE_ALLOCATED,
    RESOURCE_STATE_IN_USE,
    RESOURCE_STATE_RELEASED,
    RESOURCE_STATE_INVALID
} resource_state_t;

typedef enum {
    RESOURCE_VALID,
    RESOURCE_NULL,
    RESOURCE_INVALID,
    RESOURCE_FREED,
    RESOURCE_CORRUPTED,
    RESOURCE_OUT_OF_BOUNDS
} resource_validation_result_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    void* address;
    size_t size;
    resource_type_t type;
    resource_state_t state;
    char name[RESOURCE_VALIDATION_MAX_NAME];
    uint64_t allocation_time_ns;
    uint64_t last_access_ns;
    const char* allocation_file;
    int allocation_line;
} resource_info_t;

typedef struct {
    uint64_t total_allocated;
    uint64_t total_freed;
    uint64_t current_allocated;
    uint64_t peak_allocated;
    uint64_t allocation_count;
    uint64_t free_count;
    uint64_t leak_count;
    uint64_t invalid_access_count;
} resource_validation_stats_t;

typedef struct {
    bool enable_tracking;
    bool enable_leak_detection;
    bool enable_bounds_checking;
    bool track_allocation_source;
    size_t max_tracked_resources;
} resource_validation_config_t;

typedef struct resource_validator resource_validator_t;

// ============================================================================
// API Functions
// ============================================================================

resource_validator_t* resource_validator_create(void);
resource_validator_t* resource_validator_create_with_config(
    const resource_validation_config_t* config);
resource_validation_config_t resource_validator_default_config(void);
void resource_validator_destroy(resource_validator_t* validator);
bool resource_validator_reset(resource_validator_t* validator);

bool resource_track_allocation(resource_validator_t* validator,
                                void* address,
                                size_t size,
                                resource_type_t type,
                                const char* name);

bool resource_track_allocation_ex(resource_validator_t* validator,
                                   void* address,
                                   size_t size,
                                   resource_type_t type,
                                   const char* name,
                                   const char* file,
                                   int line);

bool resource_track_free(resource_validator_t* validator,
                          void* address);

resource_validation_result_t resource_validate(resource_validator_t* validator,
                                               const void* address);

resource_validation_result_t resource_validate_bounds(resource_validator_t* validator,
                                                      const void* address,
                                                      size_t offset,
                                                      size_t size);

bool resource_get_info(resource_validator_t* validator,
                        const void* address,
                        resource_info_t* info);

bool resource_detect_leaks(resource_validator_t* validator,
                            resource_info_t** leaks,
                            size_t* count);

bool resource_get_stats(resource_validator_t* validator,
                         resource_validation_stats_t* stats);

char* resource_validation_generate_report(resource_validator_t* validator);
char* resource_validation_export_json(resource_validator_t* validator);

const char* resource_type_name(resource_type_t type);
const char* resource_state_name(resource_state_t state);
const char* resource_validation_result_name(resource_validation_result_t result);
void resource_free_info_array(resource_info_t* info, size_t count);
const char* resource_validator_get_last_error(resource_validator_t* validator);

// Convenience macros for tracking with source location
#define RESOURCE_TRACK(v, addr, size, type, name) \
    resource_track_allocation_ex(v, addr, size, type, name, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif // RESOURCE_VALIDATION_H
