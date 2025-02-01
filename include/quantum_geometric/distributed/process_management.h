#ifndef PROCESS_MANAGEMENT_H
#define PROCESS_MANAGEMENT_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/config/mpi_config.h"

// Core process management functions
int qg_process_init(const process_config_t* config);
void qg_process_cleanup(void);
int qg_process_finalize(void);

// Process creation and control
int qg_create_process(int* process_id, const char* program_name, char* const argv[]);
int qg_terminate_process(int process_id);
int qg_suspend_process(int process_id);
int qg_resume_process(int process_id);

// Process group management
int qg_create_process_group(const char* group_name, const int* process_ids, 
                          size_t num_processes, process_group_t* group);
int qg_delete_process_group(process_group_t* group);
int qg_add_to_group(process_group_t* group, int process_id);
int qg_remove_from_group(process_group_t* group, int process_id);

// Process communication
int qg_send_signal(int process_id, int signal);
int qg_broadcast_signal(const process_group_t* group, int signal);

// Process monitoring
int qg_wait_for_process(int process_id, process_status_t* status);
int qg_wait_for_group(const process_group_t* group, process_status_t* statuses);
int qg_get_process_status(int process_id, process_status_t* status);
int qg_get_group_status(const process_group_t* group, process_status_t* statuses);
int qg_monitor_process(int process_id, void (*callback)(const process_status_t*));

// Load balancing
int qg_check_load_balance(void);
int qg_rebalance_processes(void);
int qg_migrate_process(int process_id, int target_node);

// Fault tolerance
int qg_enable_fault_tolerance(int process_id);
int qg_create_process_checkpoint(int process_id, const char* checkpoint_file);
int qg_restore_from_checkpoint(int process_id, const char* checkpoint_file);

// Resource management
int qg_set_process_priority(int process_id, int priority);
int qg_set_process_affinity(int process_id, const int* cpu_ids, size_t num_cpus);
int qg_set_resource_limits(int process_id, const size_t* limits, size_t num_limits);

// Utility functions
int qg_get_process_count(void);
int qg_get_current_process_id(void);
int qg_get_parent_process_id(int process_id);
const char* qg_get_process_name(int process_id);
const char* qg_process_get_error_string(process_error_t error);

#endif // PROCESS_MANAGEMENT_H
