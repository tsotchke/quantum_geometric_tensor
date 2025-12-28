/**
 * @file recovery_analyzer.h
 * @brief Error recovery analysis for quantum systems
 */

#ifndef RECOVERY_ANALYZER_H
#define RECOVERY_ANALYZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { RECOVERY_NONE, RECOVERY_PARTIAL, RECOVERY_FULL, RECOVERY_FAILED } RecoveryStatus;
typedef enum { RECOVERY_RETRY, RECOVERY_CHECKPOINT, RECOVERY_REDUNDANCY, RECOVERY_CORRECTION, RECOVERY_RESTART } RecoveryStrategy;

typedef struct { char* id; uint64_t timestamp; void* state; size_t size; bool valid; } RecoveryPoint;
typedef struct { size_t total; size_t successful; size_t failed; double avg_time_ms; double rate; } RecoveryMetrics;
typedef struct { RecoveryStrategy strategy; size_t max_retries; uint64_t delay_ms; bool auto_checkpoint; size_t interval; } RecoveryConfig;
typedef struct { RecoveryPoint* checkpoints; size_t num; size_t max; RecoveryConfig config; RecoveryMetrics metrics; } RecoveryContext;

int recovery_context_create(RecoveryContext** ctx, RecoveryConfig* config);
void recovery_context_destroy(RecoveryContext* ctx);
int recovery_create_checkpoint(RecoveryContext* ctx, void* state, size_t size, char** id);
int recovery_restore_checkpoint(RecoveryContext* ctx, const char* id, void** state, size_t* size);
int recovery_analyze_failure(RecoveryContext* ctx, int error_code, RecoveryStrategy* suggested);
int recovery_execute(RecoveryContext* ctx, RecoveryStrategy strategy, RecoveryStatus* status);
int recovery_get_metrics(RecoveryContext* ctx, RecoveryMetrics* metrics);
void recovery_clear_checkpoints(RecoveryContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // RECOVERY_ANALYZER_H
