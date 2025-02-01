#ifndef CONSTRAINT_ANALYZER_H
#define CONSTRAINT_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Constraint types
typedef enum {
    CONSTRAINT_PHYSICAL,      // Physical constraints
    CONSTRAINT_GEOMETRIC,     // Geometric constraints
    CONSTRAINT_QUANTUM,       // Quantum constraints
    CONSTRAINT_RESOURCE,      // Resource constraints
    CONSTRAINT_TEMPORAL      // Temporal constraints
} constraint_type_t;

// Validation modes
typedef enum {
    VALIDATE_STRICT,         // Strict validation
    VALIDATE_RELAXED,        // Relaxed validation
    VALIDATE_ADAPTIVE,       // Adaptive validation
    VALIDATE_QUANTUM        // Quantum-aware validation
} validation_mode_t;

// Constraint priorities
typedef enum {
    PRIORITY_CRITICAL,       // Critical constraints
    PRIORITY_HIGH,           // High priority
    PRIORITY_MEDIUM,         // Medium priority
    PRIORITY_LOW            // Low priority
} constraint_priority_t;

// Violation types
typedef enum {
    VIOLATION_HARD,          // Hard constraint violation
    VIOLATION_SOFT,          // Soft constraint violation
    VIOLATION_WARNING,       // Warning level violation
    VIOLATION_POTENTIAL     // Potential violation
} violation_type_t;

// Analyzer configuration
typedef struct {
    validation_mode_t mode;          // Validation mode
    double tolerance;                // Validation tolerance
    bool enable_optimization;        // Enable optimization
    bool track_violations;           // Track violations
    bool auto_correction;            // Auto correction
    size_t max_iterations;           // Maximum iterations
} analyzer_config_t;

// Constraint definition
typedef struct {
    constraint_type_t type;          // Constraint type
    constraint_priority_t priority;   // Constraint priority
    char* expression;                // Constraint expression
    double threshold;                // Constraint threshold
    bool is_mandatory;               // Mandatory flag
    void* constraint_data;           // Additional data
} constraint_def_t;

// Violation record
typedef struct {
    violation_type_t type;           // Violation type
    constraint_type_t constraint;     // Violated constraint
    double severity;                 // Violation severity
    struct timespec timestamp;       // Violation timestamp
    char* description;              // Violation description
    void* violation_data;           // Additional data
} violation_record_t;

// Validation result
typedef struct {
    bool is_valid;                   // Overall validity
    size_t total_constraints;        // Total constraints
    size_t violated_constraints;     // Violated constraints
    double compliance_score;         // Compliance score
    violation_record_t* violations;  // Violation records
    size_t num_violations;          // Number of violations
} validation_result_t;

// Correction strategy
typedef struct {
    constraint_type_t target;        // Target constraint
    double correction_factor;        // Correction factor
    bool preserve_consistency;       // Preserve consistency
    char* strategy_description;      // Strategy description
    void* strategy_data;            // Strategy-specific data
} correction_strategy_t;

// Opaque analyzer handle
typedef struct constraint_analyzer_t constraint_analyzer_t;

// Core functions
constraint_analyzer_t* create_constraint_analyzer(const analyzer_config_t* config);
void destroy_constraint_analyzer(constraint_analyzer_t* analyzer);

// Constraint management
bool add_constraint(constraint_analyzer_t* analyzer,
                   const constraint_def_t* constraint);
bool remove_constraint(constraint_analyzer_t* analyzer,
                      const char* constraint_id);
bool update_constraint(constraint_analyzer_t* analyzer,
                      const constraint_def_t* constraint);

// Validation functions
bool validate_constraints(constraint_analyzer_t* analyzer,
                        validation_result_t* result);
bool validate_specific_constraint(constraint_analyzer_t* analyzer,
                                const constraint_def_t* constraint,
                                validation_result_t* result);
bool validate_constraint_set(constraint_analyzer_t* analyzer,
                           const constraint_def_t* constraints,
                           size_t num_constraints,
                           validation_result_t* result);

// Violation handling
bool handle_violation(constraint_analyzer_t* analyzer,
                     const violation_record_t* violation,
                     correction_strategy_t* strategy);
bool track_violations(constraint_analyzer_t* analyzer,
                     violation_record_t* violations,
                     size_t* num_violations);
bool clear_violations(constraint_analyzer_t* analyzer);

// Correction functions
bool suggest_corrections(constraint_analyzer_t* analyzer,
                        const validation_result_t* result,
                        correction_strategy_t* strategies,
                        size_t* num_strategies);
bool apply_correction(constraint_analyzer_t* analyzer,
                     const correction_strategy_t* strategy);
bool validate_correction(constraint_analyzer_t* analyzer,
                        const correction_strategy_t* strategy,
                        validation_result_t* result);

// Quantum-specific functions
bool validate_quantum_constraints(constraint_analyzer_t* analyzer,
                                validation_result_t* result);
bool optimize_quantum_constraints(constraint_analyzer_t* analyzer,
                                correction_strategy_t* strategy);
bool verify_quantum_consistency(constraint_analyzer_t* analyzer,
                              validation_result_t* result);

// Geometric functions
bool validate_geometric_constraints(constraint_analyzer_t* analyzer,
                                  validation_result_t* result);
bool optimize_geometric_constraints(constraint_analyzer_t* analyzer,
                                  correction_strategy_t* strategy);
bool verify_geometric_consistency(constraint_analyzer_t* analyzer,
                                validation_result_t* result);

// Utility functions
bool export_constraint_data(const constraint_analyzer_t* analyzer,
                          const char* filename);
bool import_constraint_data(constraint_analyzer_t* analyzer,
                          const char* filename);
void free_validation_result(validation_result_t* result);

#endif // CONSTRAINT_ANALYZER_H
