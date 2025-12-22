/**
 * @file progress_tracker.c
 * @brief Progress tracking implementation
 *
 * Implements multi-step plan tracking, issue detection,
 * stall monitoring, and progress analytics integration.
 */

#include "quantum_geometric/distributed/progress_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Progress tracker - internal structure
struct ProgressTrackerImpl {
    // Plan tracking
    TrackedPlanProgress** plans;
    size_t num_plans;
    size_t plan_capacity;

    // Issue tracking
    ImplementationIssue** issues;
    size_t num_issues;
    size_t issue_capacity;

    // Analytics
    ProgressAnalytics* analytics;

    // Configuration
    TrackerConfig config;
    time_t last_update;
};

// Forward declarations
static TrackerActionPlan* create_action_plan(const char* name, const char** step_names, size_t num_steps);
static void cleanup_action_plan(TrackerActionPlan* plan);
static TrackedPlanProgress* create_plan_progress(TrackerActionPlan* plan);
static void cleanup_plan_progress(TrackedPlanProgress* progress);
static void update_overall_progress(TrackedPlanProgress* progress);
static void check_for_stalls(ProgressTracker* tracker, TrackedPlanProgress* progress);
static ProgressStatus compute_status(const TrackedPlanProgress* progress, const TrackerConfig* config);

// Initialize progress tracker
ProgressTracker* init_progress_tracker(const TrackerConfig* config) {
    ProgressTracker* tracker = calloc(1, sizeof(ProgressTracker));
    if (!tracker) return NULL;

    // Store configuration
    if (config) {
        tracker->config = *config;
    } else {
        // Default configuration
        tracker->config.max_tracked_plans = TRACKER_MAX_PLANS;
        tracker->config.update_interval_sec = TRACKER_UPDATE_INTERVAL;
        tracker->config.alert_threshold = 0.2;
        tracker->config.stall_timeout_sec = TRACKER_STALL_TIMEOUT;
        tracker->config.enable_analytics = true;
        tracker->config.enable_alerts = true;
    }

    // Initialize plan tracking
    tracker->plan_capacity = tracker->config.max_tracked_plans;
    tracker->plans = calloc(tracker->plan_capacity, sizeof(TrackedPlanProgress*));
    if (!tracker->plans) {
        free(tracker);
        return NULL;
    }
    tracker->num_plans = 0;

    // Initialize issue tracking
    tracker->issue_capacity = TRACKER_MAX_ISSUES;
    tracker->issues = calloc(tracker->issue_capacity, sizeof(ImplementationIssue*));
    if (!tracker->issues) {
        free(tracker->plans);
        free(tracker);
        return NULL;
    }
    tracker->num_issues = 0;

    // Initialize analytics
    if (tracker->config.enable_analytics) {
        tracker->analytics = init_progress_analytics(NULL);
    }

    tracker->last_update = time(NULL);

    return tracker;
}

// Create action plan
static TrackerActionPlan* create_action_plan(const char* name, const char** step_names, size_t num_steps) {
    TrackerActionPlan* plan = calloc(1, sizeof(TrackerActionPlan));
    if (!plan) return NULL;

    plan->name = name ? strdup(name) : strdup("Unnamed Plan");
    plan->description = strdup("");
    plan->num_steps = num_steps;
    plan->priority = 1.0;

    // Copy step names
    plan->step_names = calloc(num_steps, sizeof(char*));
    plan->step_weights = calloc(num_steps, sizeof(double));

    if (!plan->step_names || !plan->step_weights) {
        free(plan->name);
        free(plan->description);
        free(plan->step_names);
        free(plan->step_weights);
        free(plan);
        return NULL;
    }

    double equal_weight = 1.0 / (double)num_steps;
    for (size_t i = 0; i < num_steps; i++) {
        plan->step_names[i] = (step_names && step_names[i]) ? strdup(step_names[i]) : strdup("Step");
        plan->step_weights[i] = equal_weight;
    }

    return plan;
}

// Cleanup action plan
static void cleanup_action_plan(TrackerActionPlan* plan) {
    if (!plan) return;

    free(plan->name);
    free(plan->description);

    for (size_t i = 0; i < plan->num_steps; i++) {
        free(plan->step_names[i]);
    }
    free(plan->step_names);
    free(plan->step_weights);
    free(plan);
}

// Create plan progress
static TrackedPlanProgress* create_plan_progress(TrackerActionPlan* plan) {
    if (!plan) return NULL;

    TrackedPlanProgress* progress = calloc(1, sizeof(TrackedPlanProgress));
    if (!progress) return NULL;

    progress->plan = plan;
    progress->num_steps = plan->num_steps;
    progress->current_step = 0;
    progress->overall_progress = 0.0;
    progress->overall_status = PROGRESS_STATUS_PENDING;
    progress->start_time = time(NULL);
    progress->last_update = progress->start_time;
    progress->expected_duration = 0.0;
    progress->estimated_remaining = 0.0;

    // Create step progress
    progress->steps = calloc(plan->num_steps, sizeof(StepProgress));
    if (!progress->steps) {
        free(progress);
        return NULL;
    }

    for (size_t i = 0; i < plan->num_steps; i++) {
        progress->steps[i].step_index = i;
        progress->steps[i].step_name = plan->step_names[i] ? strdup(plan->step_names[i]) : NULL;
        progress->steps[i].expected_progress = 0.0;
        progress->steps[i].actual_progress = 0.0;
        progress->steps[i].start_time = 0;
        progress->steps[i].last_update = 0;
        progress->steps[i].status = PROGRESS_STATUS_PENDING;
        progress->steps[i].active_issue = NULL;
    }

    return progress;
}

// Cleanup plan progress
static void cleanup_plan_progress(TrackedPlanProgress* progress) {
    if (!progress) return;

    // Cleanup steps
    for (size_t i = 0; i < progress->num_steps; i++) {
        free(progress->steps[i].step_name);
        if (progress->steps[i].active_issue) {
            free(progress->steps[i].active_issue->description);
            free(progress->steps[i].active_issue->resolution);
            free(progress->steps[i].active_issue);
        }
    }
    free(progress->steps);

    // Cleanup plan
    cleanup_action_plan(progress->plan);

    free(progress);
}

// Add plan to tracker
int tracker_add_plan(ProgressTracker* tracker, const char* name, const char** step_names, size_t num_steps) {
    if (!tracker || num_steps == 0) return -1;
    if (tracker->num_plans >= tracker->plan_capacity) return -1;

    // Create plan
    TrackerActionPlan* plan = create_action_plan(name, step_names, num_steps);
    if (!plan) return -1;

    // Create progress
    TrackedPlanProgress* progress = create_plan_progress(plan);
    if (!progress) {
        cleanup_action_plan(plan);
        return -1;
    }

    // Add to tracker
    int plan_id = (int)tracker->num_plans;
    tracker->plans[tracker->num_plans++] = progress;

    return plan_id;
}

// Update step progress
void tracker_update_step(ProgressTracker* tracker, int plan_id, size_t step_index, double progress) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    TrackedPlanProgress* plan_progress = tracker->plans[plan_id];
    if (!plan_progress || step_index >= plan_progress->num_steps) return;

    StepProgress* step = &plan_progress->steps[step_index];

    // Update step
    step->actual_progress = progress;
    step->last_update = time(NULL);

    if (step->start_time == 0) {
        step->start_time = step->last_update;
    }

    // Update status
    if (progress >= 1.0) {
        step->status = PROGRESS_STATUS_COMPLETED;
    } else if (step->status == PROGRESS_STATUS_PENDING) {
        step->status = PROGRESS_STATUS_ON_TRACK;
    }

    // Update overall progress
    update_overall_progress(plan_progress);
    plan_progress->last_update = time(NULL);

    // Update analytics
    if (tracker->analytics && tracker->config.enable_analytics) {
        PlanProgress analytics_progress;
        analytics_progress.overall_progress = plan_progress->overall_progress;
        analytics_progress.completed_steps = plan_progress->current_step;
        analytics_progress.total_steps = plan_progress->num_steps;
        analytics_progress.start_time = plan_progress->start_time;
        analytics_progress.last_update_time = plan_progress->last_update;
        analytics_progress.estimated_remaining = plan_progress->estimated_remaining;
        analytics_progress.average_step_time = 0.0;

        analytics_update(tracker->analytics, &analytics_progress);
    }
}

// Complete step
void tracker_complete_step(ProgressTracker* tracker, int plan_id, size_t step_index) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    TrackedPlanProgress* plan_progress = tracker->plans[plan_id];
    if (!plan_progress || step_index >= plan_progress->num_steps) return;

    // Complete the step
    plan_progress->steps[step_index].actual_progress = 1.0;
    plan_progress->steps[step_index].status = PROGRESS_STATUS_COMPLETED;
    plan_progress->steps[step_index].last_update = time(NULL);

    // Move to next step
    if (step_index >= plan_progress->current_step) {
        plan_progress->current_step = step_index + 1;
    }

    // Update overall progress
    update_overall_progress(plan_progress);
}

// Update overall progress
static void update_overall_progress(TrackedPlanProgress* progress) {
    if (!progress || !progress->plan) return;

    double total = 0.0;
    for (size_t i = 0; i < progress->num_steps; i++) {
        double weight = progress->plan->step_weights[i];
        total += progress->steps[i].actual_progress * weight;
    }

    progress->overall_progress = total;

    // Update status
    progress->overall_status = compute_status(progress, NULL);

    // Estimate remaining time
    if (progress->overall_progress > 0.0) {
        time_t now = time(NULL);
        double elapsed = difftime(now, progress->start_time);
        double rate = progress->overall_progress / elapsed;
        if (rate > 0) {
            progress->estimated_remaining = (1.0 - progress->overall_progress) / rate;
        }
    }
}

// Compute status
static ProgressStatus compute_status(const TrackedPlanProgress* progress, const TrackerConfig* config) {
    if (!progress) return PROGRESS_STATUS_PENDING;
    (void)config;

    // Check for completion
    if (progress->overall_progress >= 1.0) {
        return PROGRESS_STATUS_COMPLETED;
    }

    // Check for blocking issues
    for (size_t i = 0; i < progress->num_steps; i++) {
        if (progress->steps[i].active_issue &&
            progress->steps[i].active_issue->is_blocking &&
            !progress->steps[i].active_issue->is_resolved) {
            return PROGRESS_STATUS_BLOCKED;
        }
    }

    // Check for stalls
    time_t now = time(NULL);
    double since_update = difftime(now, progress->last_update);
    if (since_update > TRACKER_STALL_TIMEOUT) {
        return PROGRESS_STATUS_STALLED;
    }

    return PROGRESS_STATUS_ON_TRACK;
}

// Report issue
void tracker_report_issue(ProgressTracker* tracker, int plan_id, size_t step_index,
                          const char* description, double severity, bool is_blocking) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    TrackedPlanProgress* plan_progress = tracker->plans[plan_id];
    if (!plan_progress || step_index >= plan_progress->num_steps) return;

    // Create issue
    ImplementationIssue* issue = calloc(1, sizeof(ImplementationIssue));
    if (!issue) return;

    issue->description = description ? strdup(description) : strdup("Unknown issue");
    issue->detection_time = time(NULL);
    issue->severity = severity;
    issue->is_blocking = is_blocking;
    issue->is_resolved = false;
    issue->resolution = NULL;
    issue->step_index = (int)step_index;

    // Attach to step
    if (plan_progress->steps[step_index].active_issue) {
        // Free existing issue
        free(plan_progress->steps[step_index].active_issue->description);
        free(plan_progress->steps[step_index].active_issue->resolution);
        free(plan_progress->steps[step_index].active_issue);
    }
    plan_progress->steps[step_index].active_issue = issue;

    // Update status
    if (is_blocking) {
        plan_progress->steps[step_index].status = PROGRESS_STATUS_BLOCKED;
    }

    // Track globally
    if (tracker->num_issues < tracker->issue_capacity) {
        tracker->issues[tracker->num_issues++] = issue;
    }

    update_overall_progress(plan_progress);
}

// Resolve issue
void tracker_resolve_issue(ProgressTracker* tracker, int plan_id, size_t step_index, const char* resolution) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    TrackedPlanProgress* plan_progress = tracker->plans[plan_id];
    if (!plan_progress || step_index >= plan_progress->num_steps) return;

    ImplementationIssue* issue = plan_progress->steps[step_index].active_issue;
    if (!issue) return;

    issue->is_resolved = true;
    issue->resolution = resolution ? strdup(resolution) : strdup("Resolved");

    // Update status
    if (plan_progress->steps[step_index].status == PROGRESS_STATUS_BLOCKED) {
        plan_progress->steps[step_index].status = PROGRESS_STATUS_ON_TRACK;
    }

    update_overall_progress(plan_progress);
}

// Get plan progress
const TrackedPlanProgress* tracker_get_plan_progress(const ProgressTracker* tracker, int plan_id) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return NULL;
    return tracker->plans[plan_id];
}

// Get status
ProgressStatus tracker_get_status(const ProgressTracker* tracker, int plan_id) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) {
        return PROGRESS_STATUS_PENDING;
    }
    return tracker->plans[plan_id]->overall_status;
}

// Get active count
size_t tracker_get_active_count(const ProgressTracker* tracker) {
    if (!tracker) return 0;

    size_t count = 0;
    for (size_t i = 0; i < tracker->num_plans; i++) {
        ProgressStatus status = tracker->plans[i]->overall_status;
        if (status != PROGRESS_STATUS_COMPLETED && status != PROGRESS_STATUS_FAILED) {
            count++;
        }
    }
    return count;
}

// Get issue count
size_t tracker_get_issue_count(const ProgressTracker* tracker) {
    if (!tracker) return 0;

    size_t count = 0;
    for (size_t i = 0; i < tracker->num_issues; i++) {
        if (tracker->issues[i] && !tracker->issues[i]->is_resolved) {
            count++;
        }
    }
    return count;
}

// Check for stalls
static void check_for_stalls(ProgressTracker* tracker, TrackedPlanProgress* progress) {
    if (!tracker || !progress) return;

    time_t now = time(NULL);
    double since_update = difftime(now, progress->last_update);

    if (since_update > tracker->config.stall_timeout_sec &&
        progress->overall_status == PROGRESS_STATUS_ON_TRACK) {
        progress->overall_status = PROGRESS_STATUS_STALLED;
    }
}

// Periodic update
void tracker_periodic_update(ProgressTracker* tracker) {
    if (!tracker) return;

    time_t now = time(NULL);
    double since_last = difftime(now, tracker->last_update);

    if (since_last < (double)tracker->config.update_interval_sec) {
        return;  // Not time yet
    }

    tracker->last_update = now;

    // Check all plans for stalls
    for (size_t i = 0; i < tracker->num_plans; i++) {
        check_for_stalls(tracker, tracker->plans[i]);
    }
}

// Complete plan
void tracker_complete_plan(ProgressTracker* tracker, int plan_id) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    TrackedPlanProgress* progress = tracker->plans[plan_id];
    progress->overall_progress = 1.0;
    progress->overall_status = PROGRESS_STATUS_COMPLETED;

    for (size_t i = 0; i < progress->num_steps; i++) {
        progress->steps[i].actual_progress = 1.0;
        progress->steps[i].status = PROGRESS_STATUS_COMPLETED;
    }
}

// Cancel plan
void tracker_cancel_plan(ProgressTracker* tracker, int plan_id) {
    if (!tracker || plan_id < 0 || (size_t)plan_id >= tracker->num_plans) return;

    tracker->plans[plan_id]->overall_status = PROGRESS_STATUS_FAILED;
}

// Cleanup
void cleanup_progress_tracker(ProgressTracker* tracker) {
    if (!tracker) return;

    // Cleanup plans
    for (size_t i = 0; i < tracker->num_plans; i++) {
        cleanup_plan_progress(tracker->plans[i]);
    }
    free(tracker->plans);

    // Cleanup issues (those not attached to steps)
    // Note: Issues attached to steps are cleaned up with the plans
    free(tracker->issues);

    // Cleanup analytics
    if (tracker->analytics) {
        cleanup_progress_analytics(tracker->analytics);
    }

    free(tracker);
}
