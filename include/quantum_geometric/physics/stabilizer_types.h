/**
 * @file stabilizer_types.h
 * @brief Type definitions for quantum stabilizer measurement system
 */

#ifndef QUANTUM_GEOMETRIC_STABILIZER_TYPES_H
#define QUANTUM_GEOMETRIC_STABILIZER_TYPES_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stdbool.h>
#include <stddef.h>

// Forward declarations for opaque types
// MatchingGraph is fully defined in error_syndrome.h
#ifndef MATCHING_GRAPH_DEFINED
struct MatchingGraph;
typedef struct MatchingGraph MatchingGraph;
#endif

typedef struct ProtectionSystem ProtectionSystem;
typedef struct AnyonSet AnyonSet;
typedef struct CorrectionPattern CorrectionPattern;

// Stabilizer array structure
typedef struct {
    size_t size;              // Number of stabilizers in array
    double* measurements;     // Array of measurement results
} StabilizerArray;

// Stabilizer type enumeration
#ifndef STABILIZER_TYPE_DEFINED
#define STABILIZER_TYPE_DEFINED
typedef enum StabilizerType {
    // Names used in stabilizer_types.h
    STABILIZER_PLAQUETTE = 0,    // Plaquette stabilizer (Z-type)
    STABILIZER_VERTEX = 1,       // Vertex stabilizer (X-type)
    // Aliases matching basic_topological naming convention
    PLAQUETTE_STABILIZER = 0,
    VERTEX_STABILIZER = 1,
    // Aliases for Floquet code naming convention
    STABILIZER_Z = 0,            // Z stabilizer (same as plaquette)
    STABILIZER_X = 1             // X stabilizer (same as vertex)
} StabilizerType;
#endif

// Stabilizer configuration (physics version - more comprehensive than Metal version)
#ifndef PHYSICS_STABILIZER_CONFIG_DEFINED
#define PHYSICS_STABILIZER_CONFIG_DEFINED
typedef struct PhysicsStabilizerConfig {
    size_t lattice_width;            // Width of stabilizer lattice
    size_t lattice_height;           // Height of stabilizer lattice
    double error_threshold;          // Error detection threshold
    bool auto_correction;            // Enable automatic error correction
    bool enable_parallel;            // Enable parallel measurements
    size_t max_parallel_ops;        // Maximum parallel operations
    double correlation_threshold;    // Correlation threshold for parallel ops
    size_t repetition_count;        // Number of measurement repetitions
    size_t min_valid_measurements;  // Minimum valid measurements required
    double min_confidence;          // Minimum confidence threshold
    double measurement_error_rate;  // Expected measurement error rate
    double confidence_threshold;    // Confidence threshold for measurements
    bool periodic_boundaries;       // Use periodic boundary conditions
    bool handle_boundaries;        // Special handling for boundaries
} PhysicsStabilizerConfig;
#endif // PHYSICS_STABILIZER_CONFIG_DEFINED

// =============================================================================
// Hardware Integration Types for Production Stabilizer Measurements
// =============================================================================

// Noise model for hardware error characterization
typedef struct StabilizerNoiseModel {
    double readout_error;           // Readout/measurement error rate
    double gate_error;              // Single-qubit gate error rate
    double two_qubit_error;         // Two-qubit gate error rate
    double decoherence_rate;        // Decoherence rate (1/T2)
    double crosstalk_threshold;     // Threshold for crosstalk mitigation
    double thermal_population;      // Thermal excited state population
} StabilizerNoiseModel;

// Error mitigation configuration for stabilizer measurements
typedef struct StabilizerMitigationConfig {
    bool readout_error_correction;      // Enable readout error correction
    bool dynamic_decoupling;            // Enable dynamic decoupling sequences
    bool zero_noise_extrapolation;      // Enable zero-noise extrapolation (ZNE)
    bool symmetrization;                // Enable measurement symmetrization
    bool richardson_extrapolation;      // Enable Richardson extrapolation
    bool quasi_probability;             // Enable quasi-probability decomposition
    bool spin_reversal_transform;       // Enable spin reversal transform (D-Wave)
    bool gauge_averaging;               // Enable gauge averaging (D-Wave)
    bool thermal_sampling;              // Enable thermal sampling (D-Wave)
    size_t extrapolation_order;         // Order for extrapolation methods
    double mitigation_strength;         // Overall mitigation strength factor
} StabilizerMitigationConfig;

// Performance configuration
typedef struct StabilizerPerformanceConfig {
    int optimization_level;         // Optimization level (0-3)
    size_t max_parallel_ops;        // Maximum parallel operations
    size_t pipeline_depth;          // Depth of measurement pipeline
    size_t batch_size;              // Batch size for measurements
} StabilizerPerformanceConfig;
// Alias for backward compatibility
typedef StabilizerPerformanceConfig PerformanceConfig;

// Parallel measurement configuration
typedef struct StabilizerParallelConfig {
    size_t group_size;              // Size of parallel measurement groups
    size_t min_distance;            // Minimum distance between parallel qubits
    double max_crosstalk;           // Maximum allowed crosstalk
} StabilizerParallelConfig;

// Hardware configuration for stabilizer measurements
typedef struct StabilizerHardwareConfig {
    HardwareType type;                          // Hardware backend type
    int optimization_level;                     // Optimization level (0-3)
    bool error_mitigation;                      // Enable error mitigation
    bool parallel_enabled;                      // Enable parallel measurements
    bool gpu_enabled;                           // Enable GPU acceleration
    size_t max_parallel_ops;                    // Maximum parallel operations
    StabilizerNoiseModel noise_model;           // Noise model parameters
    StabilizerMitigationConfig mitigation_config;  // Error mitigation config
    StabilizerPerformanceConfig perf_config;    // Performance configuration
} StabilizerHardwareConfig;

// =============================================================================
// Hardware Metrics Structures for Production Monitoring
// =============================================================================

// Hardware performance metrics
typedef struct StabilizerHardwareMetrics {
    // Basic performance metrics
    double readout_fidelity;                // Overall readout fidelity
    double gate_fidelity;                   // Overall gate fidelity
    double parallel_efficiency;             // Parallel execution efficiency
    double hardware_efficiency;             // Overall hardware utilization

    // Error mitigation metrics
    double error_mitigation_factor;         // Error reduction factor
    double readout_error;                   // Measured readout error rate
    double gate_error;                      // Measured gate error rate

    // Parallel measurement metrics
    size_t parallel_group_count;            // Number of parallel groups
    double avg_group_size;                  // Average parallel group size
    double parallel_speedup;                // Parallel speedup factor
    double crosstalk_level;                 // Measured crosstalk level

    // IBM-specific metrics
    double custom_instruction_usage;        // Custom instruction utilization
    double dynamic_decoupling_effectiveness; // DD effectiveness

    // Rigetti-specific metrics
    double native_gate_utilization;         // Native gate usage ratio
    double qubit_mapping_efficiency;        // Qubit mapping efficiency
    double pipeline_efficiency;             // Pipeline efficiency
    double native_gate_latency_us;          // Native gate latency (μs)
    double qubit_routing_overhead;          // Routing overhead factor

    // D-Wave-specific metrics
    double chain_strength_optimization;     // Chain strength optimization
    double embedding_efficiency;            // Embedding efficiency
    double spin_reversal_effectiveness;     // Spin reversal effectiveness
    double gauge_sampling_quality;          // Gauge sampling quality
    double thermal_calibration_accuracy;    // Thermal calibration accuracy
    double annealing_time_us;               // Annealing time (μs)
    double readout_efficiency;              // Readout efficiency
    double chain_break_fraction;            // Chain break fraction

    // ZNE/extrapolation metrics
    double readout_correction_effectiveness; // Readout correction effectiveness
    double decoupling_fidelity;             // Dynamic decoupling fidelity
    double zne_extrapolation_quality;       // ZNE extrapolation quality
    double extrapolation_confidence;        // Extrapolation confidence
    double symmetrization_quality;          // Symmetrization quality
    double quasi_probability_accuracy;      // Quasi-probability accuracy

    // Timing metrics
    double measurement_time_us;             // Total measurement time (μs)
    double correction_time_us;              // Correction time (μs)
    double measurement_latency_us;          // Measurement latency (μs)
    double gate_execution_time_us;          // Gate execution time (μs)
    double readout_time_us;                 // Readout time (μs)
    double instruction_throughput;          // Instructions per second
    double operation_throughput;            // Operations per second

    // Resource metrics
    size_t memory_overhead_kb;              // Memory overhead (KB)
    size_t memory_usage_kb;                 // Total memory usage (KB)
    double power_consumption_mw;            // Power consumption (mW)
    double qubit_utilization;               // Qubit utilization ratio
} StabilizerHardwareMetrics;

// Resource usage metrics
typedef struct StabilizerResourceMetrics {
    // Memory metrics
    size_t memory_overhead_kb;              // Memory overhead (KB)
    size_t memory_peak_kb;                  // Peak memory usage (KB)
    double memory_fragmentation;            // Memory fragmentation ratio

    // CPU metrics
    double cpu_usage_percent;               // CPU usage percentage
    double cpu_peak_percent;                // Peak CPU usage
    size_t thread_count;                    // Number of threads used

    // GPU metrics
    double gpu_usage_percent;               // GPU usage percentage
    size_t gpu_memory_usage_kb;             // GPU memory usage (KB)
    double gpu_temperature_c;               // GPU temperature (°C)

    // System impact metrics
    double system_memory_impact;            // System memory impact ratio
    size_t io_operations_per_sec;           // I/O operations per second
    size_t network_usage_bytes;             // Network usage (bytes)
} StabilizerResourceMetrics;

// Reliability metrics
typedef struct StabilizerReliabilityMetrics {
    bool operation_successful;              // Last operation succeeded
    bool false_positive_detected;           // False positive detected
    bool recovery_attempted;                // Recovery was attempted
    bool recovery_successful;               // Recovery succeeded

    double measurement_fidelity;            // Measurement fidelity
    double error_detection_confidence;      // Error detection confidence
    double correction_confidence;           // Correction confidence

    double system_uptime_seconds;           // System uptime (seconds)
    size_t consecutive_failures;            // Consecutive failure count
    double error_correction_latency_us;     // Error correction latency (μs)
} StabilizerReliabilityMetrics;

// Parallel measurement statistics
typedef struct StabilizerParallelStats {
    size_t total_groups;                    // Total parallel groups used
    double avg_group_size;                  // Average group size
    size_t max_group_size;                  // Maximum group size
    double execution_time_us;               // Total execution time (μs)
    double speedup_factor;                  // Speedup vs sequential
} StabilizerParallelStats;

// Parallel measurement group
typedef struct StabilizerParallelGroup {
    size_t* stabilizer_indices;             // Indices of stabilizers in group
    size_t size;                            // Number of stabilizers in group
    double group_confidence;                // Group measurement confidence
    double crosstalk_estimate;              // Estimated crosstalk in group
} StabilizerParallelGroup;
// Alias for backward compatibility
typedef StabilizerParallelGroup ParallelGroup;

// =============================================================================
// Extended Stabilizer Configuration with Hardware Support
// =============================================================================

// Extended stabilizer configuration (with hardware integration)
typedef struct StabilizerConfigExtended {
    size_t lattice_width;                   // Width of stabilizer lattice
    size_t lattice_height;                  // Height of stabilizer lattice
    double error_threshold;                 // Error detection threshold
    bool auto_correction;                   // Enable automatic error correction
    bool enable_parallel;                   // Enable parallel measurements
    size_t max_parallel_ops;                // Maximum parallel operations
    double correlation_threshold;           // Correlation threshold
    size_t repetition_count;                // Measurement repetitions
    size_t min_valid_measurements;          // Minimum valid measurements
    double min_confidence;                  // Minimum confidence threshold
    double measurement_error_rate;          // Expected measurement error rate
    double confidence_threshold;            // Confidence threshold
    bool periodic_boundaries;               // Use periodic boundaries
    bool handle_boundaries;                 // Special boundary handling
    bool reliability_tracking;              // Enable reliability tracking
    StabilizerHardwareConfig hardware_config;  // Hardware configuration
    StabilizerParallelConfig parallel_config;  // Parallel config
} StabilizerConfigExtended;

// StabilizerConfig - aliases to extended version for full feature support
// This allows using the simple field subset while supporting extended features
#ifndef METAL_STABILIZER_CONFIG_DEFINED
typedef StabilizerConfigExtended StabilizerConfig;
#endif

// =============================================================================
// Stabilizer State Structure (Full Feature Set)
// =============================================================================

// Forward declaration for HardwareProfile (defined in quantum_hardware_types.h)
#ifndef HARDWARE_PROFILE_FORWARD_DECL
#define HARDWARE_PROFILE_FORWARD_DECL
struct HardwareProfile;
#endif

// Stabilizer state structure (uses full StabilizerConfig for complete feature set)
typedef struct {
    StabilizerArray* plaquette_stabilizers;  // Plaquette stabilizer array
    StabilizerArray* vertex_stabilizers;     // Vertex stabilizer array
    StabilizerConfigExtended config;        // Configuration parameters (extended)
    size_t measurement_count;               // Number of measurements performed
    double error_rate;                      // Current error rate
    double* last_syndrome;                  // Last syndrome measurement
    double* measurement_confidence;         // Confidence values for measurements
    size_t* repetition_results;            // Results from repeated measurements
    double* error_correlations;            // Error correlation tracking
    bool* measured_in_parallel;            // Parallel measurement tracking
    size_t current_parallel_group;         // Current parallel measurement group
    size_t history_capacity;               // Capacity of measurement history
    size_t history_size;                   // Current size of measurement history
    double** measurement_history;          // History of measurements

    // Hardware integration (production monitoring)
    StabilizerHardwareConfig hardware_config;    // Hardware configuration
    StabilizerHardwareMetrics hardware_metrics;  // Hardware metrics
    StabilizerResourceMetrics resource_metrics;  // Resource usage metrics
    StabilizerReliabilityMetrics reliability_metrics;  // Reliability metrics
    StabilizerParallelStats parallel_stats;      // Parallel measurement stats
    StabilizerParallelGroup* parallel_groups;    // Parallel measurement groups
    size_t num_parallel_groups;                  // Number of parallel groups

    // Cross-platform hardware profile (calibration data from any backend)
    const struct HardwareProfile* hw_profile;   // Hardware profile for calibration
    bool owns_hw_profile;                       // Whether we own the profile memory
} StabilizerState;

// SyndromeConfig is defined in error_syndrome.h
// Forward declaration here for use in types that can't include error_syndrome.h
#ifndef SYNDROME_CONFIG_DEFINED
typedef struct SyndromeConfig SyndromeConfig;
#endif

#endif // QUANTUM_GEOMETRIC_STABILIZER_TYPES_H
