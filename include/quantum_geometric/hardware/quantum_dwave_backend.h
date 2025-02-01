/**
 * @file quantum_dwave_backend.h
 * @brief D-Wave Quantum backend with semi-classical emulation fallback
 */

#ifndef QUANTUM_DWAVE_BACKEND_H
#define QUANTUM_DWAVE_BACKEND_H

#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_simulator.h"
#include "quantum_geometric/core/system_dependencies.h"
#include "quantum_geometric/core/numeric_utils.h"

// D-Wave backend types
typedef enum {
    DWAVE_BACKEND_REAL,     // Real quantum hardware
    DWAVE_BACKEND_SIMULATOR // Semi-classical emulation
} DWaveBackendType;

// D-Wave backend status
typedef enum {
    DWAVE_STATUS_IDLE,
    DWAVE_STATUS_QUEUED,
    DWAVE_STATUS_RUNNING,
    DWAVE_STATUS_COMPLETED,
    DWAVE_STATUS_ERROR,
    DWAVE_STATUS_CANCELLED
} DWaveJobStatus;

// D-Wave solver types
typedef enum {
    DWAVE_SOLVER_ADVANTAGE,  // Advantage quantum processor
    DWAVE_SOLVER_DW2000Q,   // D-Wave 2000Q quantum processor
    DWAVE_SOLVER_HYBRID,    // Hybrid quantum-classical solver
    DWAVE_SOLVER_NEAL,      // Classical simulated annealing
    DWAVE_SOLVER_TABU,      // Classical tabu search
    DWAVE_SOLVER_CUSTOM     // Custom solver
} DWaveSolverType;

// D-Wave problem types
typedef enum {
    DWAVE_PROBLEM_ISING,     // Ising model
    DWAVE_PROBLEM_QUBO,      // Quadratic Unconstrained Binary Optimization
    DWAVE_PROBLEM_BQM,       // Binary Quadratic Model
    DWAVE_PROBLEM_CQM,       // Constrained Quadratic Model
    DWAVE_PROBLEM_DQM        // Discrete Quadratic Model
} DWaveProblemType;

// D-Wave sampling parameters
typedef struct {
    uint32_t num_reads;          // Number of samples
    uint32_t annealing_time;     // Annealing time in microseconds
    double chain_strength;       // Chain strength for embedding
    double programming_thermalization; // Post-programming thermalization
    bool auto_scale;            // Auto-scale problem parameters
    bool reduce_intersample_correlation; // Reduce correlation between samples
    char* readout_thermalization;  // Readout thermalization
    void* custom_params;        // Additional solver-specific parameters
} DWaveSamplingParams;

// D-Wave backend capabilities
typedef struct {
    uint32_t max_qubits;
    uint32_t num_qubits;
    uint32_t num_couplers;
    double min_annealing_time;
    double max_annealing_time;
    double* qubit_connectivity;
    size_t connectivity_size;
    bool per_qubit_coupling_range;
    bool annealing_time_range;
    bool h_gain_schedule;
    bool j_gain_schedule;
    bool flux_biases;
    bool anneal_offsets;
    bool qubit_offset_ranges;
    bool extended_j_range;
} DWaveCapabilities;

// D-Wave backend configuration
typedef struct {
    DWaveBackendType type;
    char* api_token;
    char* solver_name;
    DWaveSolverType solver_type;
    DWaveProblemType problem_type;
    DWaveSamplingParams sampling_params;
    NoiseModel noise_model;
    MitigationParams error_mitigation;
    SimulatorConfig simulator_config; // For emulation mode
    void* custom_config;
} DWaveBackendConfig;

// D-Wave problem specification
typedef struct {
    double* linear_terms;      // Linear coefficients (h)
    double* quadratic_terms;   // Quadratic coefficients (J)
    size_t num_variables;      // Number of variables
    size_t num_interactions;   // Number of interactions
    double offset;            // Energy offset
    bool* fixed_variables;    // Fixed variable values
    void* constraints;        // Problem constraints
    void* custom_data;       // Additional problem data
} DWaveProblem;

// D-Wave job configuration
typedef struct {
    DWaveProblem* problem;
    DWaveSamplingParams params;
    bool use_embedding;
    bool use_error_mitigation;
    char* job_tags[8];
    void* custom_options;
} DWaveJobConfig;

// D-Wave sample
typedef struct {
    int32_t* variables;     // Variable assignments
    double energy;         // Sample energy
    double* chain_breaks;  // Chain break fractions
    double occurrence;     // Number of occurrences
    void* custom_data;    // Additional sample data
} DWaveSample;

// D-Wave job result
typedef struct {
    DWaveSample* samples;
    size_t num_samples;
    double* energies;
    double* probabilities;
    double min_energy;
    double max_energy;
    DWaveJobStatus status;
    char* error_message;
    void* raw_data;
    double timing_info[8];
} DWaveJobResult;

// Initialize D-Wave backend
DWaveConfig* init_dwave_backend(const DWaveBackendConfig* config);

// Create quantum annealing problem
DWaveProblem* create_dwave_problem(size_t num_variables, size_t num_interactions);

// Add problem terms
bool add_linear_term(DWaveProblem* problem, size_t variable, double coefficient);
bool add_quadratic_term(DWaveProblem* problem, size_t var1, size_t var2, double coefficient);
bool add_constraint(DWaveProblem* problem, const void* constraint);

// Submit job to D-Wave backend
char* submit_dwave_job(DWaveConfig* config, const DWaveJobConfig* job_config);

// Get job status
DWaveJobStatus get_dwave_job_status(DWaveConfig* config, const char* job_id);

// Get job result
DWaveJobResult* get_dwave_job_result(DWaveConfig* config, const char* job_id);

// Cancel job
bool cancel_dwave_job(DWaveConfig* config, const char* job_id);

// Get backend capabilities
DWaveCapabilities* get_dwave_capabilities(DWaveConfig* config);

// Get available solvers
char** get_dwave_solvers(DWaveConfig* config, size_t* num_solvers);

// Get solver properties
char* get_dwave_solver_properties(DWaveConfig* config, const char* solver_name);

// Get queue information
size_t get_dwave_queue_position(DWaveConfig* config, const char* job_id);

// Get estimated runtime
double get_dwave_estimated_runtime(DWaveConfig* config, const DWaveProblem* problem);

// Optimize problem for backend
bool optimize_dwave_problem(DWaveProblem* problem, const DWaveCapabilities* capabilities);

// Apply error mitigation
bool apply_dwave_error_mitigation(DWaveJobResult* result, const MitigationParams* params);

// Convert between problem formats
DWaveProblem* qubo_to_ising(const DWaveProblem* qubo);
DWaveProblem* ising_to_qubo(const DWaveProblem* ising);

// Validate problem for backend
bool validate_dwave_problem(const DWaveProblem* problem, const DWaveCapabilities* capabilities);

// Get error information
char* get_dwave_error_info(DWaveConfig* config, const char* job_id);

// Clean up resources
void cleanup_dwave_config(DWaveConfig* config);
void cleanup_dwave_problem(DWaveProblem* problem);
void cleanup_dwave_result(DWaveJobResult* result);
void cleanup_dwave_capabilities(DWaveCapabilities* capabilities);

// Utility functions
bool save_dwave_credentials(const char* token, const char* filename);
char* load_dwave_credentials(const char* filename);
bool test_dwave_connection(const char* token);
void set_dwave_log_level(int level);
char* get_dwave_version(void);

// Advanced annealing control
bool set_annealing_schedule(DWaveConfig* config, const double* schedule, size_t points);
bool set_flux_bias_offsets(DWaveConfig* config, const double* offsets, size_t num_qubits);
bool set_anneal_offsets(DWaveConfig* config, const double* offsets, size_t num_qubits);
bool set_programming_thermalization(DWaveConfig* config, double time_us);

// Problem manipulation
bool reverse_annealing(DWaveConfig* config, const int32_t* initial_state, double reinitialize_state);
bool set_chain_strength(DWaveConfig* config, double strength);
bool set_chain_break_method(DWaveConfig* config, const char* method);
bool set_postprocess_method(DWaveConfig* config, const char* method);

#endif // QUANTUM_DWAVE_BACKEND_H
