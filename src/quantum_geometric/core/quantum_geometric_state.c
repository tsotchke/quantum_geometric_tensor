#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_handling.h"
#include "quantum_geometric/core/numerical_backend.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_scheduler.h"
#include <stdlib.h>
#include <string.h>

// geometric_create_state() - Canonical implementation in quantum_geometric_operations.c
// (removed: uses simple calloc, canonical uses pool allocation with GPU cleanup)

// geometric_destroy_state() - Canonical implementation in quantum_geometric_operations.c
// (removed: uses simple free, canonical uses pool_free with GPU buffer cleanup)

// Error handling functions
qgt_error_t report_error(error_handler_t* handler,
                        error_type_t type,
                        error_severity_t severity,
                        const char* message) {
    if (!handler) return QGT_ERROR_INVALID_PARAMETER;
    fprintf(stderr, "Error: %s\n", message);
    return QGT_SUCCESS;
}

// Queue management for scheduler
bool reorder_queue(execution_priority_t min_priority) {
    (void)min_priority; // Unused parameter
    return true;
}

// Pipeline implementation functions - REMOVED
// Broken stubs deleted. Canonical implementations are in:
// - quantum_pipeline_impl.c:quantum_pipeline_create_impl (lines 29-96)
// - quantum_pipeline_impl.c:quantum_pipeline_destroy_impl (lines 245-265)
// - quantum_pipeline_impl.c:quantum_pipeline_train_impl (lines 98-161)
// - quantum_pipeline_impl.c:quantum_pipeline_save_impl (lines 238-243)

// validate_hierarchical_matrix() - Canonical implementation in hierarchical_matrix.c
// (removed: duplicate)
