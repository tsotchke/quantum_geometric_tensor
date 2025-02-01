#ifndef QUANTUM_GEOMETRIC_HARDWARE_CAPABILITIES_H
#define QUANTUM_GEOMETRIC_HARDWARE_CAPABILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Hardware feature flags
#define HARDWARE_FEATURE_AVX2      (1 << 0)
#define HARDWARE_FEATURE_AVX512    (1 << 1)
#define HARDWARE_FEATURE_FMA       (1 << 2)
#define HARDWARE_FEATURE_SSE4_2    (1 << 3)
#define HARDWARE_FEATURE_NEON      (1 << 4)
#define HARDWARE_FEATURE_SVE       (1 << 5)
#define HARDWARE_FEATURE_AMX       (1 << 6)

// Cache levels
typedef enum {
    CACHE_LEVEL_1 = 1,
    CACHE_LEVEL_2 = 2,
    CACHE_LEVEL_3 = 3
} CacheLevel;

// Cache configuration
typedef struct {
    size_t size;           // Cache size in bytes
    size_t line_size;      // Cache line size in bytes
    size_t associativity;  // Cache associativity
    bool inclusive;        // Whether this cache level is inclusive
} CacheConfig;

// Hardware capabilities structure
// System-level hardware capabilities
typedef struct {
    uint32_t feature_flags;            // Bitfield of supported features
    size_t num_physical_cores;         // Number of physical CPU cores
    size_t num_logical_cores;          // Number of logical CPU cores (with SMT/hyperthreading)
    size_t vector_register_width;      // Width of vector registers in bits
    size_t max_vector_elements;        // Maximum number of elements per vector
    CacheConfig cache_config[3];       // L1, L2, L3 cache configurations
    bool has_fma;                      // Fused multiply-add support
    bool has_avx;                      // AVX support
    bool has_avx2;                     // AVX2 support
    bool has_avx512;                   // AVX-512 support
    bool has_neon;                     // ARM NEON support
    bool has_sve;                      // ARM SVE support
    bool has_amx;                      // Intel AMX support
    size_t page_size;                  // System page size in bytes
    size_t huge_page_size;            // Huge page size in bytes (if supported)
    bool supports_huge_pages;          // Whether huge pages are supported
} SystemCapabilities;

// Function declarations
SystemCapabilities* get_system_capabilities(void);
bool has_feature(uint32_t feature);
size_t get_optimal_vector_width(void);
size_t get_cache_line_size(CacheLevel level);
size_t get_cache_size(CacheLevel level);
bool supports_hardware_feature(const char* feature_name);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_HARDWARE_CAPABILITIES_H
