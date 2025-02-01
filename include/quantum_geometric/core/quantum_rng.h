#ifndef QUANTUM_GEOMETRIC_QUANTUM_RNG_H
#define QUANTUM_GEOMETRIC_QUANTUM_RNG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>

// Version info
#define QRNG_VERSION_MAJOR 1
#define QRNG_VERSION_MINOR 0
#define QRNG_VERSION_PATCH 0

// Constants
#define QRNG_NUM_QUBITS 16
#define QRNG_MIXING_ROUNDS 4
#define QRNG_BUFFER_SIZE 1024

// Error codes
typedef enum {
    QRNG_SUCCESS = 0,
    QRNG_ERROR_NULL_CONTEXT,
    QRNG_ERROR_NULL_BUFFER,
    QRNG_ERROR_INVALID_LENGTH,
    QRNG_ERROR_INSUFFICIENT_ENTROPY,
    QRNG_ERROR_INVALID_RANGE
} qrng_error;

// Context structure
typedef struct {
    uint64_t counter;
    uint64_t system_entropy;
    uint64_t runtime_entropy;
    uint64_t unique_id;
    uint64_t pool_mixer;
    double entropy_pool[16];
    size_t pool_index;
    double quantum_state[QRNG_NUM_QUBITS];
    uint64_t phase[QRNG_NUM_QUBITS];
    uint64_t entangle[QRNG_NUM_QUBITS];
    uint64_t last_measurement[QRNG_NUM_QUBITS];
    struct timeval init_time;
    pid_t pid;
    union {
        uint8_t bytes[QRNG_BUFFER_SIZE];
        uint64_t words[QRNG_BUFFER_SIZE / sizeof(uint64_t)];
    } buffer;
    size_t buffer_pos;
} qrng_ctx;

// Core API
qrng_error qrng_init(qrng_ctx **ctx, const uint8_t *seed, size_t seed_len);
void qrng_free(qrng_ctx *ctx);
qrng_error qrng_reseed(qrng_ctx *ctx, const uint8_t *seed, size_t seed_len);
qrng_error qrng_bytes(qrng_ctx *ctx, uint8_t *out, size_t len);
uint64_t qrng_uint64(qrng_ctx *ctx);
double qrng_double(qrng_ctx *ctx);
int32_t qrng_range32(qrng_ctx *ctx, int32_t min, int32_t max);
uint64_t qrng_range64(qrng_ctx *ctx, uint64_t min, uint64_t max);

// Advanced features
double qrng_get_entropy_estimate(qrng_ctx *ctx);
qrng_error qrng_entangle_states(qrng_ctx *ctx, uint8_t *state1, uint8_t *state2, size_t len);
qrng_error qrng_measure_state(qrng_ctx *ctx, uint8_t *state, size_t len);

// Utility functions
const char* qrng_version(void);
const char* qrng_error_string(qrng_error err);

#endif // QUANTUM_GEOMETRIC_QUANTUM_RNG_H
