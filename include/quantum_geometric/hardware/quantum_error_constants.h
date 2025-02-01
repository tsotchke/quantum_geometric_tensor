#ifndef QUANTUM_ERROR_CONSTANTS_H
#define QUANTUM_ERROR_CONSTANTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error correction code types
typedef enum {
    QEC_CODE_NONE = 0,
    QEC_CODE_SURFACE = 1,    // Surface code
    QEC_CODE_STEANE = 2,     // Steane [[7,1,3]] code
    QEC_CODE_SHOR = 3,       // Shor [[9,1,3]] code
    QEC_CODE_BACON_SHOR = 4, // Bacon-Shor code
    QEC_CODE_ROTATED = 5,    // Rotated surface code
    QEC_CODE_HEAVY_HEX = 6,  // Heavy hexagon code
    QEC_CODE_FLOQUET = 7,    // Floquet code
    QEC_CODE_CUSTOM = 8      // Custom error correction code
} quantum_error_code_t;

// Error correction thresholds
#define QEC_DEFAULT_ERROR_THRESHOLD (1.0e-3)    // Default error threshold
#define QEC_MIN_ERROR_THRESHOLD (1.0e-6)        // Minimum error threshold
#define QEC_MAX_ERROR_THRESHOLD (1.0e-1)        // Maximum error threshold

// Error correction parameters
#define QEC_DEFAULT_CODE_DISTANCE 3U         // Default code distance
#define QEC_MIN_CODE_DISTANCE 3U             // Minimum code distance
#define QEC_MAX_CODE_DISTANCE 31U            // Maximum code distance

// Syndrome extraction parameters
#define QEC_MAX_SYNDROME_ROUNDS 100U         // Maximum syndrome measurement rounds
#define QEC_MIN_SYNDROME_ROUNDS 1U           // Minimum syndrome measurement rounds
#define QEC_DEFAULT_SYNDROME_ROUNDS 10U      // Default syndrome measurement rounds

// Error correction flags
#define QEC_FLAG_ENABLE_FAST_MATCHING 0x01U  // Enable fast syndrome matching
#define QEC_FLAG_ENABLE_PARALLEL 0x02U       // Enable parallel syndrome extraction
#define QEC_FLAG_ENABLE_BOUNDARY 0x04U       // Enable boundary error correction
#define QEC_FLAG_ENABLE_WEIGHT_UPDATE 0x08U  // Enable error weight updates
#define QEC_FLAG_ENABLE_CORRELATION 0x10U    // Enable error correlations
#define QEC_FLAG_ENABLE_PREDICTION 0x20U     // Enable error prediction
#define QEC_FLAG_ENABLE_MITIGATION 0x40U     // Enable error mitigation
#define QEC_FLAG_ENABLE_VALIDATION 0x80U     // Enable result validation

// Error correction timeouts (in microseconds)
#define QEC_DEFAULT_TIMEOUT UINT32_C(1000000)         // Default timeout (1 second)
#define QEC_MIN_TIMEOUT UINT32_C(1000)                // Minimum timeout (1 millisecond)
#define QEC_MAX_TIMEOUT UINT32_C(3600000000)          // Maximum timeout (1 hour)

// Error correction buffer sizes
#define QEC_MAX_ERROR_BUFFER 1024U           // Maximum error buffer size
#define QEC_MAX_SYNDROME_BUFFER 2048U        // Maximum syndrome buffer size
#define QEC_MAX_CORRECTION_BUFFER 4096U      // Maximum correction buffer size

// Error correction weights
#define QEC_DEFAULT_X_WEIGHT (1.0)            // Default X error weight
#define QEC_DEFAULT_Y_WEIGHT (1.0)            // Default Y error weight
#define QEC_DEFAULT_Z_WEIGHT (1.0)            // Default Z error weight

// Error correction confidence thresholds
#define QEC_HIGH_CONFIDENCE (0.99)            // High confidence threshold
#define QEC_MEDIUM_CONFIDENCE (0.95)          // Medium confidence threshold
#define QEC_LOW_CONFIDENCE (0.90)             // Low confidence threshold

// Error correction optimization levels
typedef enum {
    QEC_OPTIMIZE_NONE = 0,      // No optimization
    QEC_OPTIMIZE_SPEED = 1,     // Optimize for speed
    QEC_OPTIMIZE_ACCURACY = 2,   // Optimize for accuracy
    QEC_OPTIMIZE_MEMORY = 3,     // Optimize for memory usage
    QEC_OPTIMIZE_BALANCED = 4    // Balance speed/accuracy/memory
} qec_optimization_level_t;

// Error correction validation modes
typedef enum {
    QEC_VALIDATE_NONE = 0,      // No validation
    QEC_VALIDATE_BASIC = 1,     // Basic validation
    QEC_VALIDATE_FULL = 2,      // Full validation
    QEC_VALIDATE_STRICT = 3     // Strict validation with all checks
} qec_validation_mode_t;

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ERROR_CONSTANTS_H
