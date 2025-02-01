#include <unistd.h>
#include <sys/sysinfo.h>
#include <string.h>
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/hardware_capabilities.h"
#include "quantum_geometric/hardware/cpu_features.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Platform-specific memory size detection
static size_t get_system_memory_size(void) {
#ifdef __APPLE__
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t memory_size = 0;
    size_t len = sizeof(memory_size);
    if (sysctl(mib, 2, &memory_size, &len, NULL, 0) == 0) {
        return (size_t)memory_size;
    }
    return 0;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (size_t)si.totalram * si.mem_unit;
    }
    return 0;
#endif
}

// Helper to detect CPU features and set flags
static void detect_cpu_features(SystemCapabilities* caps) {
    caps->feature_flags = 0;
    
    // Check each feature and update flags
    caps->has_fma = cpu_has_feature(bit_FMA);
    caps->has_avx = cpu_has_feature(bit_AVX);
    caps->has_avx2 = cpu_has_feature(bit_AVX2);
    caps->has_avx512 = cpu_has_feature(bit_AVX512F);
    
    // Set feature flags
    if (caps->has_fma) caps->feature_flags |= CAP_FMA;
    if (caps->has_avx) caps->feature_flags |= CAP_AVX;
    if (caps->has_avx2) caps->feature_flags |= CAP_AVX2;
    if (caps->has_avx512) caps->feature_flags |= CAP_AVX512;
    
    // ARM features
    #if defined(__ARM_NEON)
        caps->has_neon = true;
        caps->feature_flags |= CAP_NEON;
    #else
        caps->has_neon = false;
    #endif
    
    #if defined(__ARM_FEATURE_SVE)
        caps->has_sve = true;
        caps->feature_flags |= CAP_SVE;
    #else
        caps->has_sve = false;
    #endif
    
    // AMX features
    caps->has_amx = cpu_has_feature(bit_AMX_BF16) && 
                    cpu_has_feature(bit_AMX_TILE) && 
                    cpu_has_feature(bit_AMX_INT8);
    if (caps->has_amx) caps->feature_flags |= CAP_AMX;
}

// Helper to detect cache configuration
static void detect_cache_config(CacheConfig cache_config[3]) {
    for (int i = 0; i < 3; i++) {
        unsigned int size = 0, line_size = 0, associativity = 0;
        get_cpu_cache_info(i + 1, &size, &line_size, &associativity);
        
        cache_config[i].size = size;
        cache_config[i].line_size = line_size;
        cache_config[i].associativity = associativity;
        cache_config[i].inclusive = false; // Determined by architecture
    }
}

SystemCapabilities detect_system_capabilities(void) {
    SystemCapabilities caps = {0};
    
    // Detect CPU cores
    caps.num_physical_cores = sysconf(_SC_NPROCESSORS_CONF);
    caps.num_logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Detect CPU features
    get_cpu_features(&caps.feature_flags, &caps.has_fma, &caps.has_avx,
                    &caps.has_avx2, &caps.has_avx512, &caps.has_neon,
                    &caps.has_sve, &caps.has_amx);
    
    // Detect cache configuration
    detect_cache_config(caps.cache_config);
    
    // Get memory page sizes
    caps.page_size = sysconf(_SC_PAGESIZE);
    #ifdef _SC_HUGETLB_PAGES
        caps.huge_page_size = caps.page_size * 512; // Typical huge page size
        caps.supports_huge_pages = sysconf(_SC_HUGETLB_PAGES) > 0;
    #else
        caps.huge_page_size = 0;
        caps.supports_huge_pages = false;
    #endif
    
    // Set vector capabilities based on detected features
    if (caps.has_avx512) {
        caps.vector_register_width = 512;
        caps.max_vector_elements = 16;
    } else if (caps.has_avx2 || caps.has_avx) {
        caps.vector_register_width = 256;
        caps.max_vector_elements = 8;
    } else if (caps.has_neon) {
        caps.vector_register_width = 128;
        caps.max_vector_elements = 4;
    } else {
        caps.vector_register_width = 128;
        caps.max_vector_elements = 4;
    }
    
    return caps;
}

QuantumHardwareCapabilities detect_quantum_capabilities(HardwareType type) {
    QuantumHardwareCapabilities caps = {0};
    
    switch (type) {
        case HARDWARE_TYPE_CPU:
            caps.supports_openmp = true;
            caps.supports_reset = true;
            caps.max_qubits = 32;
            caps.max_gates = 1000000;
            caps.max_depth = 1000;
            caps.max_shots = 10000;
            caps.max_parallel_jobs = sysconf(_SC_NPROCESSORS_ONLN);
            break;
            
        case HARDWARE_TYPE_GPU:
            caps.supports_gpu = true;
            #ifdef __APPLE__
                caps.supports_metal = true;
            #else
                caps.supports_cuda = true;
            #endif
            caps.supports_reset = true;
            caps.max_qubits = 32;
            caps.max_gates = 1000000;
            caps.max_depth = 1000;
            caps.max_shots = 10000;
            caps.max_parallel_jobs = 1;
            break;
            
        case HARDWARE_TYPE_QPU:
            // QPU capabilities would be set based on specific quantum hardware
            caps.supports_reset = true;
            caps.max_qubits = 16;  // Conservative estimate
            caps.max_gates = 100;
            caps.max_depth = 50;
            caps.max_shots = 1000;
            caps.max_parallel_jobs = 1;
            caps.coherence_time = 100e-6;  // 100 microseconds
            caps.gate_time = 100e-9;      // 100 nanoseconds
            caps.readout_time = 1e-6;     // 1 microsecond
            break;
            
        case HARDWARE_TYPE_SIMULATOR:
            caps.supports_openmp = true;
            caps.supports_reset = true;
            caps.max_qubits = 32;
            caps.max_gates = 1000000;
            caps.max_depth = 1000;
            caps.max_shots = 10000;
            caps.max_parallel_jobs = sysconf(_SC_NPROCESSORS_ONLN);
            break;
            
        case HARDWARE_TYPE_METAL:
            caps.supports_gpu = true;
            caps.supports_metal = true;
            caps.supports_reset = true;
            caps.max_qubits = 32;
            caps.max_gates = 1000000;
            caps.max_depth = 1000;
            caps.max_shots = 10000;
            caps.max_parallel_jobs = 1;
            break;
            
        default:
            // Set minimal capabilities for unknown hardware
            caps.supports_reset = true;
            caps.max_qubits = 16;
            caps.max_gates = 1000;
            caps.max_depth = 100;
            caps.max_shots = 1000;
            caps.max_parallel_jobs = 1;
            break;
    }
    
    return caps;
}
