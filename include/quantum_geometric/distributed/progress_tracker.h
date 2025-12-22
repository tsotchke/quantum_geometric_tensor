#ifndef PROGRESS_TRACKER_H
#define PROGRESS_TRACKER_H

/**
 * @file progress_tracker.h
 * @brief Progress tracking for distributed training plans
 *
 * Provides tracking of multi-step training plans, issue detection,
 * and progress analytics integration.
 */

#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include "quantum_geometric/distributed/progress_analytics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define TRACKER_MAX_PLANS 100
#define TRACKER_MAX_STEPS 1000
#define TRACKER_MAX_ISSUES 100
#define TRACKER_STALL_TIMEOUT 3600  // 1 hour
#define TRACKER_UPDATE_INTERVAL 60  // 1 minute

// Progress status
typedef enum {
    PROGRESS_STATUS_PENDING,
    PROGRESS_STATUS_ON_TRACK,
    PROGRESS_STATUS_BEHIND,
    PROGRESS_STATUS_STALLED,
    PROGRESS_STATUS_BLOCKED,
    PROGRESS_STATUS_COMPLETED,
    PROGRESS_STATUS_FAILED
} ProgressStatus;

// Implementation issue
typedef struct {
    char* description;
    time_t detection_time;
    double severity;
    bool is_blocking;
    bool is_resolved;
    char* resolution;
    int step_index;
} ImplementationIssue;

// Step progress
typedef struct {
    size_t step_index;
    char* step_name;
    double expected_progress;
    double actual_progress;
    time_t start_time;
    time_t last_update;
    ProgressStatus status;
    ImplementationIssue* active_issue;
} StepProgress;

// Action plan (simplified for tracking)
typedef struct {
    char* name;
    char* description;
    size_t num_steps;
    char** step_names;
    double* step_weights;  // Relative weight of each step
    double priority;
} TrackerActionPlan;

// Tracked plan progress
typedef struct {
    TrackerActionPlan* plan;
    StepProgress* steps;
    size_t num_steps;
    size_t current_step;
    double overall_progress;
    ProgressStatus overall_status;
    time_t start_time;
    time_t last_update;
    double expected_duration;
    double estimated_remaining;
} TrackedPlanProgress;

// Monitoring configuration
typedef struct {
    size_t max_tracked_plans;
    size_t update_interval_sec;
    double alert_threshold;
    double stall_timeout_sec;
    bool enable_analytics;
    bool enable_alerts;
} TrackerConfig;

// Progress tracker (opaque)
typedef struct ProgressTrackerImpl ProgressTracker;

// Initialize progress tracker
ProgressTracker* init_progress_tracker(const TrackerConfig* config);

// Create and track a new plan
int tracker_add_plan(
    ProgressTracker* tracker,
    const char* name,
    const char** step_names,
    size_t num_steps);

// Update step progress
void tracker_update_step(
    ProgressTracker* tracker,
    int plan_id,
    size_t step_index,
    double progress);

// Complete current step and move to next
void tracker_complete_step(
    ProgressTracker* tracker,
    int plan_id,
    size_t step_index);

// Report an issue
void tracker_report_issue(
    ProgressTracker* tracker,
    int plan_id,
    size_t step_index,
    const char* description,
    double severity,
    bool is_blocking);

// Resolve an issue
void tracker_resolve_issue(
    ProgressTracker* tracker,
    int plan_id,
    size_t step_index,
    const char* resolution);

// Get plan progress
const TrackedPlanProgress* tracker_get_plan_progress(
    const ProgressTracker* tracker,
    int plan_id);

// Get overall status
ProgressStatus tracker_get_status(
    const ProgressTracker* tracker,
    int plan_id);

// Get number of active plans
size_t tracker_get_active_count(const ProgressTracker* tracker);

// Get number of active issues
size_t tracker_get_issue_count(const ProgressTracker* tracker);

// Periodic update (check for stalls, update analytics)
void tracker_periodic_update(ProgressTracker* tracker);

// Complete a plan
void tracker_complete_plan(ProgressTracker* tracker, int plan_id);

// Cancel a plan
void tracker_cancel_plan(ProgressTracker* tracker, int plan_id);

// Clean up progress tracker
void cleanup_progress_tracker(ProgressTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif // PROGRESS_TRACKER_H
