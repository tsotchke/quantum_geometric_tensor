#ifndef AMX_OPERATIONS_H
#define AMX_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// AMX configuration modes
#define AMX_MODE_FP64 0x2ULL // Enable FP64 mode
#define AMX_MODE_FP32 0x1ULL // Enable FP32 mode
#define AMX_MODE_OFF 0x0ULL  // Disable AMX

// AMX system registers
#define AMX_SYSREG_CONFIG "S3_3_C11_C0_2" // AMX configuration register
#define AMX_SYSREG_STATE "S3_3_C11_C4_2"  // AMX state register
#define AMX_SYSREG_ADDR "S3_3_C11_C4_1"   // AMX address register
#define AMX_SYSREG_STRIDE "S3_3_C11_C4_3" // AMX stride register
#define AMX_SYSREG_ROWS "S3_3_C11_C2_0"   // AMX rows register
#define AMX_SYSREG_COLS "S3_3_C11_C2_1"   // AMX columns register
#define AMX_SYSREG_DEPTH "S3_3_C11_C2_2"  // AMX depth register

// Initialize AMX unit
bool amx_init(void);

// Shutdown AMX unit
void amx_shutdown(void);

// Load matrix into AMX registers
bool amx_load_matrix(const double *data, size_t rows, size_t cols);

// Store matrix from AMX registers
bool amx_store_matrix(double *result, size_t rows, size_t cols);

// Perform matrix multiplication using AMX
bool amx_matrix_multiply(const double *a, const double *b, double *c, size_t m,
                         size_t n, size_t k);

// Check if AMX is available on this system
bool amx_is_available(void);

// Get AMX capabilities
uint64_t amx_get_capabilities(void);

#endif // AMX_OPERATIONS_H
