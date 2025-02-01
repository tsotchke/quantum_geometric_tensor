#ifndef CAPABILITY_ANALYZER_H
#define CAPABILITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Capability types
typedef enum {
    CAP_QUANTUM_GATES,        // Quantum gate operations
    CAP_QUANTUM_MEMORY,       // Quantum memory
    CAP_ERROR_CORRECTION,     // Error correction
    CAP_ENTANGLEMENT,        // Entanglement operations
    CAP_MEASUREMENT         // Measurement capabilities
} capability_type_t;

// Hardware types
typedef enum {
    HARDWARE_IBM,            // IBM quantum hardware
    HARDWARE_RIGETTI,        // Rigetti quantum hardware
    HARDWARE_DWAVE,          // D-Wave quantum hardware
    HARDWARE_SIMULATOR      // Quantum simulator
} hardware_type_t;

// Feature support levels
typedef enum {
    SUPPORT_NONE,            // Feature not supported
    SUPPORT_BASIC,           // Basic support
    SUPPORT_ADVANCED,        // Advanced support
    SUPPORT_FULL            // Full support
} support_level_t;

// Performance levels
typedef enum {
    PERF_LOW,               // Low performance
    PERF_MEDIUM,            // Medium performance
    PERF_HIGH,              // High performance
    PERF_OPTIMAL           // Optimal performance
} performance_level_t;

// Analyzer configuration
typedef struct {
    hardware_type_t hardware;         // Hardware type
    size_t update_interval;           // Update interval
    bool track_history;               // Track capability history
    bool enable_prediction;           // Enable capability prediction
    bool validate_capabilities;       // Validate capabilities
    size_t history_size;             // History size
} analyzer_config_t;

// Capability metrics
typedef struct {
    capability_type_t type;           // Capability type
    support_level_t support;          // Support level
    performance_level_t performance;  // Performance level
    double reliability;               // Reliability score
    double efficiency;                // Efficiency score
    size_t max_qubits;               // Maximum qubits
} capability_metrics_t;

// Hardware specifications
typedef struct {
    hardware_type_t type;             // Hardware type
    size_t num_qubits;                // Number of qubits
    double coherence_time;            // Coherence time
    double gate_fidelity;             // Gate fidelity
    double measurement_fidelity;      // Measurement fidelity
    size_t max_circuit_depth;         // Maximum circuit depth
} hardware_specs_t;

// Feature requirements
typedef struct {
    capability_type_t capability;      // Required capability
    support_level_t min_support;       // Minimum support level
    performance_level_t min_perf;      // Minimum performance
    size_t min_qubits;                // Minimum qubits
    double min_fidelity;              // Minimum fidelity
    bool error_correction;            // Error correction required
} feature_requirements_t;

// Capability prediction
typedef struct {
    capability_type_t capability;      // Capability type
    support_level_t future_support;    // Predicted support
    performance_level_t future_perf;   // Predicted performance
    struct timespec prediction_time;   // Prediction timestamp
    double confidence;                // Prediction confidence
} capability_prediction_t;

// Opaque analyzer handle
typedef struct capability_analyzer_t capability_analyzer_t;

// Core functions
capability_analyzer_t* create_capability_analyzer(const analyzer_config_t* config);
void destroy_capability_analyzer(capability_analyzer_t* analyzer);

// Analysis functions
bool analyze_capabilities(capability_analyzer_t* analyzer,
                         capability_metrics_t* metrics,
                         size_t* num_capabilities);
bool analyze_hardware_specs(capability_analyzer_t* analyzer,
                          hardware_specs_t* specs);
bool validate_capabilities(capability_analyzer_t* analyzer,
                         const capability_metrics_t* metrics);

// Feature support functions
bool check_feature_support(capability_analyzer_t* analyzer,
                          const feature_requirements_t* requirements,
                          bool* supported);
bool get_supported_features(capability_analyzer_t* analyzer,
                          capability_type_t* features,
                          size_t* num_features);
bool validate_feature_requirements(capability_analyzer_t* analyzer,
                                 const feature_requirements_t* requirements);

// Performance functions
bool measure_capability_performance(capability_analyzer_t* analyzer,
                                  capability_type_t capability,
                                  performance_level_t* performance);
bool evaluate_hardware_performance(capability_analyzer_t* analyzer,
                                 hardware_specs_t* specs,
                                 performance_level_t* performance);
bool track_performance_trends(capability_analyzer_t* analyzer,
                            capability_type_t capability,
                            performance_level_t* history,
                            size_t* num_entries);

// Prediction functions
bool predict_capabilities(capability_analyzer_t* analyzer,
                         capability_prediction_t* predictions,
                         size_t* num_predictions);
bool validate_predictions(capability_analyzer_t* analyzer,
                         const capability_prediction_t* predictions,
                         const capability_metrics_t* actual);
bool update_prediction_model(capability_analyzer_t* analyzer,
                           const capability_metrics_t* metrics);

// Monitoring functions
bool monitor_capability_changes(capability_analyzer_t* analyzer,
                              capability_type_t capability,
                              capability_metrics_t* metrics);
bool get_capability_history(const capability_analyzer_t* analyzer,
                          capability_metrics_t* history,
                          size_t* num_entries);
bool reset_capability_history(capability_analyzer_t* analyzer);

// Utility functions
bool export_capability_data(const capability_analyzer_t* analyzer,
                          const char* filename);
bool import_capability_data(capability_analyzer_t* analyzer,
                          const char* filename);
void free_capability_metrics(capability_metrics_t* metrics);

#endif // CAPABILITY_ANALYZER_H
