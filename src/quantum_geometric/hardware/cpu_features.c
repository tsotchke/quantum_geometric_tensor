#include <string.h>
#include <cpuid.h>
#include "quantum_geometric/hardware/cpu_features.h"

// Buffer sizes for CPU strings
#define VENDOR_STRING_LENGTH 13
#define BRAND_STRING_LENGTH 49

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

void get_cpu_cache_info(unsigned int level, unsigned int* size, unsigned int* line_size, unsigned int* associativity) {
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
