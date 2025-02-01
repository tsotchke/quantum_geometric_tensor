#ifndef QUANTUM_ERROR_HANDLING_H
#define QUANTUM_ERROR_HANDLING_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/core/error_codes.h"

// Error types
typedef enum {
    ERROR_QUANTUM,          // Quantum errors
    ERROR_CLASSICAL,        // Classical errors
    ERROR_HARDWARE,         // Hardware errors
    ERROR_SOFTWARE,         // Software errors
    ERROR_COMMUNICATION    // Communication errors
} error_type_t;

// Error severity levels
typedef enum {
    SEVERITY_CRITICAL,      // Critical errors
    SEVERITY_HIGH,          // High severity
    SEVERITY_MEDIUM,        // Medium severity
    SEVERITY_LOW,          // Low severity
    SEVERITY_INFO         // Informational
} error_severity_t;

// Error handling modes
typedef enum {
    HANDLE_ABORT,          // Abort on error
    HANDLE_RETRY,          // Retry on error
    HANDLE_RECOVER,        // Attempt recovery
    HANDLE_IGNORE,         // Ignore error
    HANDLE_DELEGATE       // Delegate to handler
} error_handling_mode_t;

// Error correction types
typedef enum {
    CORRECTION_SURFACE,     // Surface code
    CORRECTION_STABILIZER,  // Stabilizer code
    CORRECTION_TOPOLOGICAL, // Topological code
    CORRECTION_QUANTUM,     // Quantum code
    CORRECTION_HYBRID      // Hybrid correction
} error_correction_type_t;

// Error handler configuration
typedef struct {
    error_handling_mode_t mode;     // Handling mode
    size_t max_retries;             // Maximum retries
    double timeout;                 // Operation timeout
    bool log_errors;                // Enable error logging
    char* log_file;                // Error log file path
    void* handler_data;            // Additional data
} error_handler_config_t;

// Error context
typedef struct {
    error_type_t type;              // Error type
    error_severity_t severity;       // Error severity
    qgt_error_t code;               // Error code
    char* message;                  // Error message
    char* file;                     // Source file
    int line;                       // Line number
    void* context_data;            // Additional context
} error_context_t;

// Protection system
typedef struct {
    size_t num_qubits;              // Number of qubits
    size_t code_distance;           // Code distance
    error_correction_type_t type;    // Correction type
    void* error_correction;         // Error correction
    void* syndrome_extraction;      // Syndrome extraction
    void* decoder;                 // Error decoder
    void* protection_data;         // Additional data
} quantum_protection_t;

// Error statistics
typedef struct {
    size_t total_errors;            // Total errors
    size_t corrected_errors;        // Corrected errors
    size_t uncorrected_errors;      // Uncorrected errors
    double error_rate;              // Error rate
    double correction_rate;         // Correction rate
    void* statistics_data;         // Additional data
} error_statistics_t;

// Opaque error handler handle
typedef struct error_handler_t error_handler_t;

// Core functions
error_handler_t* create_error_handler(const error_handler_config_t* config);
void destroy_error_handler(error_handler_t* handler);

// Error handling functions
qgt_error_t handle_error(error_handler_t* handler,
                        const error_context_t* context);
qgt_error_t register_error_callback(error_handler_t* handler,
                                   void (*callback)(const error_context_t*));
qgt_error_t set_error_mode(error_handler_t* handler,
                          error_handling_mode_t mode);

// Error reporting functions
qgt_error_t report_error(error_handler_t* handler,
                        error_type_t type,
                        error_severity_t severity,
                        const char* message);
qgt_error_t log_error(error_handler_t* handler,
                      const error_context_t* context);
const char* get_error_message(qgt_error_t code);

// Protection system functions
quantum_protection_t* create_protection_system(size_t num_qubits,
                                            error_correction_type_t type);
void destroy_protection_system(quantum_protection_t* protection);
qgt_error_t configure_protection(quantum_protection_t* protection,
                               const void* config);

// Error correction functions
qgt_error_t correct_errors(quantum_protection_t* protection,
                          quantum_system_t* system);
qgt_error_t extract_syndrome(quantum_protection_t* protection,
                            quantum_system_t* system,
                            void* syndrome);
qgt_error_t decode_syndrome(quantum_protection_t* protection,
                           const void* syndrome,
                           void* correction);

// System protection functions
qgt_error_t protect_quantum_system(quantum_system_t* system,
                                  quantum_protection_t* protection);
qgt_error_t protect_quantum_state(void* state,
                                 quantum_protection_t* protection);
qgt_error_t validate_protection(const quantum_protection_t* protection);

// Statistics functions
qgt_error_t get_error_statistics(const error_handler_t* handler,
                                error_statistics_t* statistics);
qgt_error_t reset_statistics(error_handler_t* handler);
qgt_error_t export_statistics(const error_handler_t* handler,
                             const char* filename);

// Utility functions
bool is_error_correctable(error_type_t type,
                         const quantum_protection_t* protection);
double estimate_error_rate(const error_statistics_t* statistics);
void clear_error_context(error_context_t* context);

#endif // QUANTUM_ERROR_HANDLING_H
