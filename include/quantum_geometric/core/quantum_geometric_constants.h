#ifndef QUANTUM_GEOMETRIC_CONSTANTS_H
#define QUANTUM_GEOMETRIC_CONSTANTS_H

// Mathematical constants
#define QGT_PI                  3.14159265358979323846
#define QGT_E                   2.71828182845904523536
#define QGT_SQRT2              1.41421356237309504880
#define QGT_SQRT3              1.73205080756887729352
#define QGT_GOLDEN_RATIO       1.61803398874989484820

// Physical constants (SI units)
#define QGT_PLANCK             6.62607015e-34    // Planck constant (J⋅s)
#define QGT_GATE_DELAY         1e-7              // Gate delay time (s)
#define QGT_HBAR               1.054571817e-34   // Reduced Planck constant (J⋅s)
#define QGT_C                  299792458.0       // Speed of light (m/s)
#define QGT_G                  6.67430e-11       // Gravitational constant (m³/kg⋅s²)
#define QGT_ELECTRON_MASS      9.1093837015e-31  // Electron mass (kg)
#define QGT_PROTON_MASS       1.67262192369e-27  // Proton mass (kg)
#define QGT_ELECTRON_CHARGE    1.602176634e-19   // Elementary charge (C)
#define QGT_FINE_STRUCTURE     7.297352569e-3    // Fine structure constant
#define QGT_BOLTZMANN         1.380649e-23       // Boltzmann constant (J/K)
#define QGT_VACUUM_PERMITTIVITY 8.8541878128e-12 // Vacuum permittivity (F/m)

// Quantum computing constants
#define QGT_MAX_QUBITS         64                // Maximum number of qubits
#define QGT_MAX_GATES          1024              // Maximum number of gates
#define QGT_MAX_DEPTH          256               // Maximum circuit depth
#define QGT_MAX_MEASUREMENTS   128               // Maximum measurements
#define QGT_MAX_CLASSICAL_BITS 256               // Maximum classical bits

// Geometric constants
#define QGT_MAX_DIMENSIONS     16                // Maximum geometric dimensions
#define QGT_MAX_RANK          8                 // Maximum tensor rank
#define QGT_MAX_INDICES       32                // Maximum contraction indices
#define QGT_MAX_SYMMETRIES    16                // Maximum symmetry operations
#define QGT_MAX_BOUNDARIES    32                // Maximum boundary conditions

// Computational constants
#define QGT_EPSILON           1e-10             // Numerical epsilon
#define QGT_MAX_ITERATIONS    10000             // Maximum iterations
#define QGT_CONVERGENCE_TOL   1e-8              // Convergence tolerance
#define QGT_MIN_EIGENVALUE    1e-12             // Minimum eigenvalue
#define QGT_MAX_BATCH_SIZE    1024              // Maximum batch size
#define QGT_MAX_THREADS       64                // Maximum threads
#define QGT_CACHE_LINE_SIZE   64                // Cache line size in bytes
#define QGT_PAGE_SIZE         4096              // Memory page size in bytes

// Error correction constants
#define QGT_MAX_ERROR_RATE    0.01              // Maximum error rate
#define QGT_MIN_FIDELITY      0.99              // Minimum fidelity
#define QGT_MAX_SYNDROME_SIZE 128               // Maximum syndrome size
#define QGT_MAX_CODE_DISTANCE 32                // Maximum code distance
#define QGT_MAX_STABILIZERS   256               // Maximum stabilizers

// Resource management constants
#define QGT_MAX_MEMORY        (1ULL << 32)      // Maximum memory (4GB)
#define QGT_MAX_REGISTERS     256               // Maximum quantum registers
#define QGT_MAX_DEVICES       16                // Maximum hardware devices
#define QGT_MAX_STREAMS       32                // Maximum concurrent streams
#define QGT_MAX_EVENTS        64                // Maximum events

// Performance constants
#define QGT_MIN_BLOCK_SIZE    32                // Minimum block size
#define QGT_MAX_BLOCK_SIZE    1024              // Maximum block size
#define QGT_MIN_GRID_SIZE     1                 // Minimum grid size
#define QGT_MAX_GRID_SIZE     65535             // Maximum grid size
#define QGT_MAX_SHARED_MEMORY (48 * 1024)       // Maximum shared memory (48KB)
#define QGT_MAX_REGISTERS_PER_BLOCK 255         // Maximum registers per block

// Memory pool constants
#define QGT_POOL_ALIGNMENT    64                // Memory pool alignment (aligned with cache line size)
#define QGT_NUM_SIZE_CLASSES  8                 // Number of size classes
#define QGT_MIN_POOL_BLOCK    64                // Minimum block size for small allocations
#define QGT_POOL_PREFETCH     8                 // Reduced prefetch distance for better cache utilization
#define QGT_MAX_POOL_BLOCK    (1024 * 1024)     // Maximum block size (1MB)
#define QGT_MAX_POOL_THREADS  32                // Maximum threads supported by memory pool

// Optimization constants
#define QGT_MAX_OPTIMIZATION_STEPS 1000         // Maximum optimization steps
#define QGT_LEARNING_RATE     0.001             // Default learning rate
#define QGT_MOMENTUM          0.9               // Default momentum
#define QGT_WEIGHT_DECAY      0.0001            // Default weight decay
#define QGT_DROPOUT_RATE      0.5               // Default dropout rate
#define QGT_BATCH_NORM_EPSILON 1e-5             // Batch normalization epsilon

// Distributed computing constants
#define QGT_MAX_NODES         1024              // Maximum compute nodes
#define QGT_MAX_PROCESSES     4096              // Maximum processes
#define QGT_MAX_CONNECTIONS   8192              // Maximum connections
#define QGT_MAX_MESSAGE_SIZE  (1 << 20)         // Maximum message size (1MB)
#define QGT_MAX_BUFFER_SIZE   (1 << 24)         // Maximum buffer size (16MB)

// Version information
#define QGT_VERSION_MAJOR     1                 // Major version number
#define QGT_VERSION_MINOR     0                 // Minor version number
#define QGT_VERSION_PATCH     0                 // Patch version number
#define QGT_VERSION_STRING    "1.0.0"           // Version string

// Optimization levels
typedef enum {
    QUANTUM_OPT_NONE = 0,      // No optimizations
    QUANTUM_OPT_BASIC = 1,     // Basic optimizations
    QUANTUM_OPT_AGGRESSIVE = 2  // Aggressive optimizations
} quantum_optimization_level_t;

#endif // QUANTUM_GEOMETRIC_CONSTANTS_H
