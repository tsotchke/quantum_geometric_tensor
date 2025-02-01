#ifndef QUANTUM_GEOMETRIC_CPU_FEATURES_H
#define QUANTUM_GEOMETRIC_CPU_FEATURES_H

// CPUID feature bits
#define bit_SSE3       (1 << 0)
#define bit_PCLMUL     (1 << 1)
#define bit_DTES64     (1 << 2)
#define bit_MONITOR    (1 << 3)
#define bit_DS_CPL     (1 << 4)
#define bit_VMX        (1 << 5)
#define bit_SMX        (1 << 6)
#define bit_EST        (1 << 7)
#define bit_TM2        (1 << 8)
#define bit_SSSE3      (1 << 9)
#define bit_CID        (1 << 10)
#define bit_FMA        (1 << 12)
#define bit_CX16       (1 << 13)
#define bit_ETPRD      (1 << 14)
#define bit_PDCM       (1 << 15)
#define bit_PCIDE      (1 << 17)
#define bit_DCA        (1 << 18)
#define bit_SSE4_1     (1 << 19)
#define bit_SSE4_2     (1 << 20)
#define bit_x2APIC     (1 << 21)
#define bit_MOVBE      (1 << 22)
#define bit_POPCNT     (1 << 23)
#define bit_AES        (1 << 25)
#define bit_XSAVE      (1 << 26)
#define bit_OSXSAVE    (1 << 27)
#define bit_AVX        (1 << 28)

// Extended features
#define bit_AVX2          (1 << 5)
#define bit_AVX512F       (1 << 16)
#define bit_AVX512DQ      (1 << 17)
#define bit_AVX512IFMA    (1 << 21)
#define bit_AVX512PF      (1 << 26)
#define bit_AVX512ER      (1 << 27)
#define bit_AVX512CD      (1 << 28)
#define bit_SHA           (1 << 29)
#define bit_AVX512BW      (1 << 30)
#define bit_AVX512VL      (1 << 31)

// AMX features
#define bit_AMX_BF16      (1 << 22)  // AMX BFloat16 Support
#define bit_AMX_TILE      (1 << 24)  // AMX Tile Architecture
#define bit_AMX_INT8      (1 << 25)  // AMX Int8 Support

// Function declarations for CPU feature detection
#ifdef __cplusplus
extern "C" {
#endif

// Get CPU vendor string
void get_cpu_vendor(char* vendor);

// Get CPU brand string
void get_cpu_brand(char* brand);

// Check if CPU supports a specific feature
bool cpu_has_feature(unsigned int feature);

// Get CPU cache information
void get_cpu_cache_info(unsigned int level, unsigned int* size, unsigned int* line_size, unsigned int* associativity);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_CPU_FEATURES_H
