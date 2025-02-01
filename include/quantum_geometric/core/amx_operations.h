#ifndef AMX_OPERATIONS_H
#define AMX_OPERATIONS_H

#include <stdbool.h>
#include <stdint.h>

// AMX tile dimensions and alignment
#define AMX_TILE_M 32  // Tile height
#define AMX_TILE_N 32  // Tile width
#define AMX_TILE_K 32  // Tile depth for matrix multiply
#define AMX_ALIGNMENT 64  // Cache line alignment requirement

// AMX state structure
typedef struct {
    uint8_t x[1024];  // x registers
    uint8_t y[1024];  // y registers
    uint8_t z[1024];  // z registers
} amx_state_t;

// Core AMX operations
bool amx_available(void);
int amx_init(void);
void amx_cleanup(void);
void amx_matrix_multiply(float* C, const float* A, const float* B, int size);

// Low-level AMX instructions (implemented in assembly)
extern void _amx_init_asm(void);
extern void amx_stop(void);
extern void amx_ldx(const void* ptr, uint64_t offset);
extern void amx_ldy(const void* ptr, uint64_t offset);
extern void amx_stx(void* ptr, uint64_t offset);
extern void amx_sty(void* ptr, uint64_t offset);
extern void amx_ldz(const void* ptr, uint64_t offset);
extern void amx_stz(void* ptr, uint64_t offset);
extern void amx_fma64(uint64_t x_offset, uint64_t y_offset, uint64_t z_offset);

#endif // AMX_OPERATIONS_H
