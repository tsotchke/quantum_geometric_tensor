#include <string.h>
#include <stdbool.h>
#include "quantum_geometric/hardware/cpu_features.h"

// Platform-specific includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define QGT_ARCH_X86
#include <cpuid.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define QGT_ARCH_ARM64
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

// Buffer sizes for CPU strings
#define VENDOR_STRING_LENGTH 13
#define BRAND_STRING_LENGTH 49

// ============================================================================
// x86 Implementation
// ============================================================================

#ifdef QGT_ARCH_X86

void get_cpu_vendor(char* vendor) {
    unsigned int eax, ebx, ecx, edx;

    // Get vendor string using CPUID leaf 0
    __cpuid(0, eax, ebx, ecx, edx);

    // Combine registers into vendor string
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
}

void get_cpu_brand(char* brand) {
    unsigned int eax, ebx, ecx, edx;

    // Check if brand string is supported
    __cpuid(0x80000000, eax, ebx, ecx, edx);
    if (eax < 0x80000004) {
        strcpy(brand, "Unknown");
        return;
    }

    // Get brand string using CPUID leaves 0x80000002-0x80000004
    for (unsigned int i = 0; i < 3; i++) {
        __cpuid(0x80000002 + i, eax, ebx, ecx, edx);
        memcpy(brand + i * 16, &eax, 4);
        memcpy(brand + i * 16 + 4, &ebx, 4);
        memcpy(brand + i * 16 + 8, &ecx, 4);
        memcpy(brand + i * 16 + 12, &edx, 4);
    }
    brand[48] = '\0';
}

bool cpu_has_feature(unsigned int feature) {
    unsigned int eax, ebx, ecx, edx;

    // Get standard features
    __cpuid(1, eax, ebx, ecx, edx);

    // Check standard feature bits
    switch (feature) {
        case bit_FMA:
            return (ecx & bit_FMA) != 0;
        case bit_AVX:
            return (ecx & bit_AVX) != 0;
        case bit_SSE4_1:
            return (ecx & bit_SSE4_1) != 0;
        case bit_SSE4_2:
            return (ecx & bit_SSE4_2) != 0;
    }

    // Get extended features
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    // Check extended feature bits
    switch (feature) {
        case bit_AVX2:
            return (ebx & bit_AVX2) != 0;
        case bit_AVX512F:
            return (ebx & bit_AVX512F) != 0;
        case bit_AVX512DQ:
            return (ebx & bit_AVX512DQ) != 0;
        case bit_AVX512BW:
            return (ebx & bit_AVX512BW) != 0;
        case bit_AVX512VL:
            return (ebx & bit_AVX512VL) != 0;
        case bit_AMX_BF16:
            return (edx & bit_AMX_BF16) != 0;
        case bit_AMX_TILE:
            return (edx & bit_AMX_TILE) != 0;
        case bit_AMX_INT8:
            return (edx & bit_AMX_INT8) != 0;
    }

    // Feature not recognized
    return false;
}

void get_cpu_cache_info(unsigned int level, unsigned int* size,
                        unsigned int* line_size, unsigned int* associativity) {
    unsigned int eax, ebx, ecx, edx;

    // Initialize outputs
    *size = 0;
    *line_size = 0;
    *associativity = 0;

    // Check if cache info is supported
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 4) return;

    // Query cache info
    __cpuid_count(4, level - 1, eax, ebx, ecx, edx);

    // Check if this cache level exists
    if ((eax & 0x1f) == 0) return;

    // Calculate cache properties
    *line_size = (ebx & 0xfff) + 1;
    *associativity = ((ebx >> 22) & 0x3ff) + 1;
    unsigned int partitions = ((ebx >> 12) & 0x3ff) + 1;
    unsigned int sets = ecx + 1;

    // Calculate total size
    *size = (*associativity) * partitions * (*line_size) * sets;
}

// Helper function to check OS support for features
static bool os_supports_feature(unsigned int feature) {
    unsigned int eax, ebx, ecx, edx;

    // Check if XSAVE is supported and enabled by OS
    __cpuid(1, eax, ebx, ecx, edx);
    if ((ecx & bit_XSAVE) == 0 || (ecx & bit_OSXSAVE) == 0) {
        return false;
    }

    // Get XCR0 register
    unsigned int xcr0_low, xcr0_high;
    __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));

    // Check feature-specific OS support
    switch (feature) {
        case bit_AVX:
            // Check if OS saves YMM registers (bits 2:1 = 11b)
            return (xcr0_low & 6) == 6;

        case bit_AVX512F:
            // Check if OS saves ZMM registers (bits 7:5 = 111b)
            return (xcr0_low & 0xe0) == 0xe0;

        case bit_AMX_BF16:
        case bit_AMX_TILE:
        case bit_AMX_INT8:
            // Check if OS saves AMX state (bit 17 = 1b)
            return (xcr0_low & (1 << 17)) != 0;

        default:
            return true;
    }
}

#endif // QGT_ARCH_X86

// ============================================================================
// ARM64 Implementation
// ============================================================================

#ifdef QGT_ARCH_ARM64

void get_cpu_vendor(char* vendor) {
#ifdef __APPLE__
    strcpy(vendor, "Apple");
#else
    strcpy(vendor, "ARM");
#endif
}

void get_cpu_brand(char* brand) {
#ifdef __APPLE__
    // Query macOS for CPU brand
    size_t len = BRAND_STRING_LENGTH - 1;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0) != 0) {
        strcpy(brand, "Apple Silicon");
    }
#else
    strcpy(brand, "ARM64 Processor");
#endif
}

bool cpu_has_feature(unsigned int feature) {
    // ARM feature detection
    // On ARM, we use compile-time feature detection and runtime checks

#ifdef __ARM_NEON
    // NEON is always available on ARM64
    if (feature == CPU_FEATURE_NEON) return true;
#endif

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
    if (feature == CPU_FEATURE_FP16) return true;
#endif

#ifdef __ARM_FEATURE_DOTPROD
    if (feature == CPU_FEATURE_DOTPROD) return true;
#endif

#ifdef __APPLE__
    // Apple Silicon always has these features
    switch (feature) {
        case CPU_FEATURE_NEON:
        case CPU_FEATURE_FP16:
        case CPU_FEATURE_DOTPROD:
            return true;
        default:
            break;
    }
#endif

    return false;
}

void get_cpu_cache_info(unsigned int level, unsigned int* size,
                        unsigned int* line_size, unsigned int* associativity) {
    // Initialize outputs
    *size = 0;
    *line_size = 0;
    *associativity = 0;

#ifdef __APPLE__
    // Query macOS for cache info
    const char* size_key = NULL;
    const char* line_key = NULL;

    switch (level) {
        case 1:
            size_key = "hw.l1dcachesize";
            line_key = "hw.cachelinesize";
            break;
        case 2:
            size_key = "hw.l2cachesize";
            line_key = "hw.cachelinesize";
            break;
        case 3:
            size_key = "hw.l3cachesize";
            line_key = "hw.cachelinesize";
            break;
        default:
            return;
    }

    if (size_key) {
        size_t val = 0;
        size_t len = sizeof(val);
        if (sysctlbyname(size_key, &val, &len, NULL, 0) == 0) {
            *size = (unsigned int)val;
        }
    }

    if (line_key) {
        size_t val = 0;
        size_t len = sizeof(val);
        if (sysctlbyname(line_key, &val, &len, NULL, 0) == 0) {
            *line_size = (unsigned int)val;
        }
    }

    // Estimate associativity (Apple doesn't expose this directly)
    // Typical values for Apple Silicon
    switch (level) {
        case 1: *associativity = 8; break;
        case 2: *associativity = 12; break;
        case 3: *associativity = 16; break;
        default: *associativity = 8; break;
    }
#else
    // Linux ARM: could read from /sys/devices/system/cpu/cpu0/cache/
    // For now, use reasonable defaults
    switch (level) {
        case 1:
            *size = 64 * 1024;       // 64KB L1
            *line_size = 64;
            *associativity = 4;
            break;
        case 2:
            *size = 512 * 1024;      // 512KB L2
            *line_size = 64;
            *associativity = 8;
            break;
        case 3:
            *size = 8 * 1024 * 1024; // 8MB L3
            *line_size = 64;
            *associativity = 16;
            break;
    }
#endif
}

// Unused on ARM but kept for API compatibility
static bool os_supports_feature(unsigned int feature) {
    (void)feature;
    return true;
}

#endif // QGT_ARCH_ARM64

// ============================================================================
// Fallback Implementation (for unknown architectures)
// ============================================================================

#if !defined(QGT_ARCH_X86) && !defined(QGT_ARCH_ARM64)

void get_cpu_vendor(char* vendor) {
    strcpy(vendor, "Unknown");
}

void get_cpu_brand(char* brand) {
    strcpy(brand, "Unknown Processor");
}

bool cpu_has_feature(unsigned int feature) {
    (void)feature;
    return false;
}

void get_cpu_cache_info(unsigned int level, unsigned int* size,
                        unsigned int* line_size, unsigned int* associativity) {
    (void)level;
    *size = 0;
    *line_size = 64;  // Common default
    *associativity = 8;
}

static bool os_supports_feature(unsigned int feature) {
    (void)feature;
    return false;
}

#endif

// ============================================================================
// Common Implementation (for all architectures)
// ============================================================================

void get_cpu_features(unsigned int* feature_flags, bool* has_fma, bool* has_avx,
                     bool* has_avx2, bool* has_avx512, bool* has_neon,
                     bool* has_sve, bool* has_amx) {
    // Initialize all to safe defaults
    if (feature_flags) *feature_flags = 0;
    if (has_fma) *has_fma = false;
    if (has_avx) *has_avx = false;
    if (has_avx2) *has_avx2 = false;
    if (has_avx512) *has_avx512 = false;
    if (has_neon) *has_neon = false;
    if (has_sve) *has_sve = false;
    if (has_amx) *has_amx = false;

#ifdef QGT_ARCH_X86
    // Detect x86 features
    if (has_fma && cpu_has_feature(bit_FMA) && os_supports_feature(bit_FMA)) {
        *has_fma = true;
        if (feature_flags) *feature_flags |= (1 << 9);  // CAP_FMA
    }
    if (has_avx && cpu_has_feature(bit_AVX) && os_supports_feature(bit_AVX)) {
        *has_avx = true;
        if (feature_flags) *feature_flags |= (1 << 10);  // CAP_AVX
    }
    if (has_avx2 && cpu_has_feature(bit_AVX2) && os_supports_feature(bit_AVX)) {
        *has_avx2 = true;
        if (feature_flags) *feature_flags |= (1 << 11);  // CAP_AVX2
    }
    if (has_avx512 && cpu_has_feature(bit_AVX512F) && os_supports_feature(bit_AVX512F)) {
        *has_avx512 = true;
        if (feature_flags) *feature_flags |= (1 << 12);  // CAP_AVX512
    }
    if (has_amx && cpu_has_feature(bit_AMX_TILE) && os_supports_feature(bit_AMX_TILE)) {
        *has_amx = true;
        if (feature_flags) *feature_flags |= (1 << 15);  // CAP_AMX
    }
#endif

#ifdef QGT_ARCH_ARM64
    // Detect ARM features
    if (has_neon && cpu_has_feature(CPU_FEATURE_NEON)) {
        *has_neon = true;
        if (feature_flags) *feature_flags |= (1 << 13);  // CAP_NEON
    }
    if (has_sve && cpu_has_feature(CPU_FEATURE_SVE)) {
        *has_sve = true;
        if (feature_flags) *feature_flags |= (1 << 14);  // CAP_SVE
    }
#endif
}
