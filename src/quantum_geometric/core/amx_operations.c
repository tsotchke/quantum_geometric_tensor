#include <stdlib.h>
#include <string.h>
#include "quantum_geometric/core/amx_operations.h"

#ifdef __aarch64__

static amx_state_t amx_state;

// AMX instructions (implemented in assembly)
extern void _amx_init_asm(void);
extern void amx_stop(void);
extern void amx_ldx(const void* ptr, uint64_t offset);
extern void amx_ldy(const void* ptr, uint64_t offset);
extern void amx_stx(void* ptr, uint64_t offset);
extern void amx_sty(void* ptr, uint64_t offset);
extern void amx_ldz(const void* ptr, uint64_t offset);
extern void amx_stz(void* ptr, uint64_t offset);
extern void amx_fma64(uint64_t x_offset, uint64_t y_offset, uint64_t z_offset);

// Initialize AMX unit
static void initialize_amx() {
    static int initialized = 0;
    if (!initialized) {
        _amx_init_asm();
        initialized = 1;
    }
}

// Helper function to load matrix block into AMX registers
static void load_matrix_block(const float* matrix, int row, int col, int stride, int block_size) {
    float block[AMX_TILE_M][AMX_TILE_N] __attribute__((aligned(AMX_ALIGNMENT)));
    
    // Copy block from matrix to aligned buffer
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            block[i][j] = matrix[(row + i) * stride + (col + j)];
        }
    }
    
    // Load block into AMX registers with proper alignment
    amx_ldx(block, 0);
}

// Helper function to store matrix block from AMX registers
static void store_matrix_block(float* matrix, int row, int col, int stride, int block_size) {
    float block[AMX_TILE_M][AMX_TILE_N] __attribute__((aligned(AMX_ALIGNMENT)));
    
    // Store block from AMX registers
    amx_stz(block, 0);
    
    // Copy block to matrix
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            matrix[(row + i) * stride + (col + j)] = block[i][j];
        }
    }
}

void amx_matrix_multiply(float* C, const float* A, const float* B, int size) {
    initialize_amx();
    
    // Process matrix in blocks of AMX tile size
    for (int i = 0; i < size; i += AMX_TILE_M) {
        for (int j = 0; j < size; j += AMX_TILE_N) {
            // Clear accumulator
            memset(amx_state.z, 0, sizeof(amx_state.z));
            
            for (int k = 0; k < size; k += AMX_TILE_K) {
                // Load blocks from A and B with proper alignment
                load_matrix_block(A, i, k, size, AMX_TILE_M);
                load_matrix_block(B, k, j, size, AMX_TILE_N);
                
                // Perform block matrix multiplication using AMX
                amx_fma64(0, 0, 0);
            }
            
            // Store result block to C
            store_matrix_block(C, i, j, size, AMX_TILE_M);
        }
    }
}

int amx_init(void) {
    static int initialized = 0;
    if (!initialized) {
        _amx_init_asm();
        initialized = 1;
    }
    return 0;
}

void amx_cleanup(void) {
    amx_stop();
    memset(&amx_state, 0, sizeof(amx_state));
}

#else

// Fallback implementation for non-Apple Silicon platforms
void amx_matrix_multiply(float* C, const float* A, const float* B, int size) {
    // Use basic OpenMP parallel implementation
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int amx_init(void) {
    return 0;
}

void amx_cleanup(void) {
    // Nothing to clean up in fallback implementation
}

#endif // __aarch64__

// Helper function to check if AMX is available
bool amx_available() {
#ifdef __aarch64__
    // Check if running on Apple Silicon
    #ifdef __APPLE__
        return true;  // AMX is available on all Apple Silicon chips
    #else
        return false;  // Not on Apple platform
    #endif
#else
    return false;  // Not ARM64 architecture
#endif
}
