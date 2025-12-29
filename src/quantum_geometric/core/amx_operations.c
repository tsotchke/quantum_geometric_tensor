/**
 * @file amx_operations.c
 * @brief AMX (Apple Matrix Extension) operations implementation
 *
 * Provides matrix multiplication acceleration using Apple's AMX
 * hardware on Apple Silicon, with CPU fallback for other platforms.
 */

#include "quantum_geometric/core/amx_operations.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// AMX state
static bool g_amx_initialized = false;
static bool g_amx_available = false;

bool amx_available(void) {
#ifdef __APPLE__
#ifdef __arm64__
    return true;  // Apple Silicon has AMX via Accelerate
#else
    return false;
#endif
#else
    return false;
#endif
}

int amx_init(void) {
    if (g_amx_initialized) {
        return g_amx_available ? 0 : -1;
    }

#ifdef __APPLE__
#ifdef __arm64__
    // On Apple Silicon, AMX is available through Accelerate framework
    // The actual AMX instructions are used internally by Accelerate's BLAS
    g_amx_available = true;
#else
    // x86 Mac - no AMX, but Accelerate still works
    g_amx_available = false;
#endif
#else
    // Non-Apple platform - no AMX
    g_amx_available = false;
#endif

    g_amx_initialized = true;
    return g_amx_available ? 0 : -1;
}

void amx_cleanup(void) {
    // Reset AMX state
    g_amx_initialized = false;
    g_amx_available = false;
}

// Provide amx_shutdown as alias for amx_cleanup (backward compatibility)
void amx_shutdown(void) {
    amx_cleanup();
}

void amx_matrix_multiply(float* C, const float* A, const float* B, int size) {
    if (!A || !B || !C || size <= 0) {
        return;
    }

#ifdef __APPLE__
    // Use Accelerate's cblas_sgemm which utilizes AMX on Apple Silicon
    // C = alpha * A * B + beta * C
    // With alpha = 1.0, beta = 0.0: C = A * B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size,
                1.0f,  // alpha
                A, size,   // A and its leading dimension
                B, size,   // B and its leading dimension
                0.0f,  // beta
                C, size);  // C and its leading dimension
#else
    // CPU fallback: naive matrix multiplication
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
#endif
}
