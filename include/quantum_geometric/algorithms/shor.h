/**
 * @file shor.h
 * @brief Shor's Algorithm for Integer Factorization
 *
 * Production implementation of Shor's quantum factoring algorithm.
 * This algorithm finds prime factors of composite integers in polynomial time
 * on a quantum computer, compared to exponential time classically.
 *
 * Algorithm Overview:
 * 1. Classical preprocessing: check if N is even, prime, or prime power
 * 2. Choose random a coprime to N
 * 3. Quantum order finding: find r such that a^r ≡ 1 (mod N)
 * 4. Classical post-processing: extract factors using gcd(a^(r/2) ± 1, N)
 *
 * The quantum part uses:
 * - Quantum Fourier Transform for phase estimation
 * - Modular exponentiation circuits
 * - Continued fraction expansion for period extraction
 */

#ifndef SHOR_H
#define SHOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * @brief Shor algorithm configuration
 */
typedef struct {
    size_t num_trials;              // Number of quantum trials before giving up
    size_t precision_bits;          // Bits of precision for phase estimation
    bool use_semiclassical_qft;     // Use semiclassical QFT (sequential measurement)
    bool use_approximate_qft;       // Use approximate QFT (reduced rotations)
    size_t approximate_qft_cutoff;  // Cutoff for approximate QFT rotations
    bool use_gpu;                   // Enable GPU acceleration
    void* backend;                  // Hardware backend (NULL for simulator)
} shor_config_t;

/**
 * @brief State for order-finding subroutine
 */
typedef struct shor_state {
    uint64_t N;                     // Number to factor
    uint64_t a;                     // Random base coprime to N
    size_t num_qubits;              // Total qubits needed
    size_t precision_qubits;        // Qubits for precision register
    size_t work_qubits;             // Qubits for work register
    QuantumState* qstate;           // Quantum state
    shor_config_t config;           // Configuration
} shor_state_t;

/**
 * @brief Result of Shor's algorithm
 */
typedef struct {
    bool success;                   // Whether factorization succeeded
    uint64_t factor1;               // First factor (or 0 if failed)
    uint64_t factor2;               // Second factor (or 0 if failed)
    uint64_t order;                 // Order found (r where a^r ≡ 1 mod N)
    size_t num_trials;              // Number of trials used
    double execution_time;          // Total execution time in seconds
    char* error_message;            // Error message if failed (NULL if success)
} shor_result_t;

/**
 * @brief Continued fraction convergent
 */
typedef struct {
    uint64_t numerator;
    uint64_t denominator;
} cf_convergent_t;

// ============================================================================
// Main Algorithm Functions
// ============================================================================

/**
 * @brief Factor an integer using Shor's algorithm
 *
 * @param N The integer to factor (must be odd composite > 1)
 * @param config Algorithm configuration (NULL for defaults)
 * @return Result containing factors or error information
 */
shor_result_t* shor_factor(uint64_t N, const shor_config_t* config);

/**
 * @brief Find the multiplicative order of a modulo N
 *
 * Finds the smallest positive r such that a^r ≡ 1 (mod N)
 * This is the core quantum subroutine of Shor's algorithm.
 *
 * @param a Base (must be coprime to N)
 * @param N Modulus
 * @param config Algorithm configuration
 * @return Order r, or 0 if not found
 */
uint64_t shor_find_order(uint64_t a, uint64_t N, const shor_config_t* config);

/**
 * @brief Create default Shor configuration
 */
shor_config_t shor_default_config(void);

/**
 * @brief Clean up Shor result
 */
void shor_destroy_result(shor_result_t* result);

// ============================================================================
// Classical Number Theory Functions
// ============================================================================

/**
 * @brief Compute GCD using Euclidean algorithm
 */
uint64_t shor_gcd(uint64_t a, uint64_t b);

/**
 * @brief Compute a^b mod n using repeated squaring
 */
uint64_t shor_mod_exp(uint64_t base, uint64_t exp, uint64_t mod);

/**
 * @brief Check if n is prime using Miller-Rabin
 */
bool shor_is_prime(uint64_t n);

/**
 * @brief Check if n is a perfect power (n = a^k for some a, k > 1)
 */
bool shor_is_perfect_power(uint64_t n, uint64_t* base, uint64_t* power);

/**
 * @brief Find a random integer coprime to N
 */
uint64_t shor_random_coprime(uint64_t N);

/**
 * @brief Compute continued fraction expansion of p/q
 *
 * @param numerator Measured phase numerator
 * @param denominator Phase denominator (2^precision_bits)
 * @param max_denom Maximum denominator to consider
 * @param convergents Output array for convergents
 * @param max_convergents Size of convergents array
 * @return Number of convergents computed
 */
size_t shor_continued_fraction(uint64_t numerator, uint64_t denominator,
                                uint64_t max_denom,
                                cf_convergent_t* convergents, size_t max_convergents);

// ============================================================================
// Quantum Circuit Functions
// ============================================================================

/**
 * @brief Initialize quantum state for order finding
 *
 * Creates the initial state |0⟩^n |1⟩ where n is precision qubits
 * and the work register is initialized to |1⟩.
 */
shor_state_t* shor_init_state(uint64_t N, uint64_t a, const shor_config_t* config);

/**
 * @brief Apply modular exponentiation controlled by precision register
 *
 * Implements the transformation:
 * |x⟩|y⟩ → |x⟩|y * a^x mod N⟩
 *
 * This is the core quantum operation that encodes the period.
 */
qgt_error_t shor_apply_modexp(shor_state_t* state);

/**
 * @brief Apply inverse QFT to precision register
 */
qgt_error_t shor_apply_inverse_qft(shor_state_t* state);

/**
 * @brief Measure precision register and extract phase
 *
 * @param state Quantum state after QFT
 * @param measured_phase Output: measured phase as fraction of 2π
 * @return Error code
 */
qgt_error_t shor_measure_phase(shor_state_t* state, double* measured_phase);

/**
 * @brief Clean up Shor state
 */
void shor_destroy_state(shor_state_t* state);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate number of qubits needed for factoring N
 */
size_t shor_required_qubits(uint64_t N);

/**
 * @brief Validate input for Shor's algorithm
 *
 * Checks that N is odd, composite, and not a prime power.
 */
bool shor_validate_input(uint64_t N, char** error_message);

/**
 * @brief Print Shor result summary
 */
void shor_print_result(const shor_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // SHOR_H
