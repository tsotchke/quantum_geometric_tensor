/**
 * @file shor.c
 * @brief Production Implementation of Shor's Factoring Algorithm
 *
 * Complete implementation of Shor's quantum integer factorization algorithm.
 * Uses quantum phase estimation to find the period of modular exponentiation,
 * then extracts factors using classical continued fraction expansion.
 */

#include "quantum_geometric/algorithms/shor.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Classical Number Theory Implementation
// ============================================================================

uint64_t shor_gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

uint64_t shor_mod_exp(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;

    uint64_t result = 1;
    base = base % mod;

    while (exp > 0) {
        if (exp & 1) {
            // Prevent overflow using 128-bit multiplication if available
#if defined(__SIZEOF_INT128__)
            __uint128_t tmp = (__uint128_t)result * base;
            result = (uint64_t)(tmp % mod);
#else
            // Fallback: use double for modular multiplication (less precise for large N)
            result = (uint64_t)fmod((double)result * (double)base, (double)mod);
#endif
        }
        exp >>= 1;
#if defined(__SIZEOF_INT128__)
        __uint128_t tmp = (__uint128_t)base * base;
        base = (uint64_t)(tmp % mod);
#else
        base = (uint64_t)fmod((double)base * (double)base, (double)mod);
#endif
    }
    return result;
}

bool shor_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Miller-Rabin primality test with deterministic witnesses for n < 2^64
    // These witnesses guarantee correctness for all 64-bit integers
    uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    size_t num_witnesses = sizeof(witnesses) / sizeof(witnesses[0]);

    // Write n-1 as 2^r * d
    uint64_t d = n - 1;
    uint64_t r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }

    for (size_t i = 0; i < num_witnesses && witnesses[i] < n; i++) {
        uint64_t a = witnesses[i];
        uint64_t x = shor_mod_exp(a, d, n);

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (uint64_t j = 0; j < r - 1; j++) {
            x = shor_mod_exp(x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) return false;
    }

    return true;
}

bool shor_is_perfect_power(uint64_t n, uint64_t* base, uint64_t* power) {
    if (n <= 1) return false;

    // Check all possible exponents from 2 to log2(n)
    size_t max_exp = (size_t)(log2((double)n) + 1);

    for (size_t k = 2; k <= max_exp; k++) {
        // Binary search for base b such that b^k = n
        uint64_t lo = 2;
        uint64_t hi = (uint64_t)pow((double)n, 1.0 / (double)k) + 2;

        while (lo <= hi) {
            uint64_t mid = lo + (hi - lo) / 2;

            // Compute mid^k carefully to avoid overflow
            uint64_t result = 1;
            bool overflow = false;
            for (size_t i = 0; i < k && !overflow; i++) {
                if (result > n / mid) {
                    overflow = true;
                } else {
                    result *= mid;
                }
            }

            if (overflow || result > n) {
                hi = mid - 1;
            } else if (result < n) {
                lo = mid + 1;
            } else {
                // Found: mid^k = n
                if (base) *base = mid;
                if (power) *power = k;
                return true;
            }
        }
    }

    return false;
}

uint64_t shor_random_coprime(uint64_t N) {
    // Use a simple PRNG seeded with time
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }

    // Try random values until we find one coprime to N
    for (int attempts = 0; attempts < 1000; attempts++) {
        uint64_t a = 2 + (rand() % (N - 3));  // Random in [2, N-2]
        if (shor_gcd(a, N) == 1) {
            return a;
        }
    }

    // Fallback: sequential search
    for (uint64_t a = 2; a < N; a++) {
        if (shor_gcd(a, N) == 1) {
            return a;
        }
    }

    return 0;  // Should never happen for composite N
}

size_t shor_continued_fraction(uint64_t numerator, uint64_t denominator,
                                uint64_t max_denom,
                                cf_convergent_t* convergents, size_t max_convergents) {
    if (!convergents || max_convergents == 0) return 0;

    // Extended Euclidean algorithm to compute continued fraction
    uint64_t p_prev = 0, p_curr = 1;  // Previous and current numerators
    uint64_t q_prev = 1, q_curr = 0;  // Previous and current denominators

    uint64_t a = numerator;
    uint64_t b = denominator;
    size_t count = 0;

    while (b != 0 && count < max_convergents) {
        uint64_t quotient = a / b;
        uint64_t remainder = a % b;

        // Update convergents: p_n = a_n * p_{n-1} + p_{n-2}
        uint64_t p_new = quotient * p_curr + p_prev;
        uint64_t q_new = quotient * q_curr + q_prev;

        if (q_new > max_denom) break;

        convergents[count].numerator = p_new;
        convergents[count].denominator = q_new;
        count++;

        p_prev = p_curr;
        p_curr = p_new;
        q_prev = q_curr;
        q_curr = q_new;

        a = b;
        b = remainder;
    }

    return count;
}

// ============================================================================
// Quantum State Management
// ============================================================================

size_t shor_required_qubits(uint64_t N) {
    // Need 2n qubits for precision register and n qubits for work register
    // where n = ceil(log2(N))
    size_t n = 0;
    uint64_t temp = N;
    while (temp > 0) {
        n++;
        temp >>= 1;
    }
    return 3 * n;  // 2n for precision, n for work
}

bool shor_validate_input(uint64_t N, char** error_message) {
    if (N <= 1) {
        if (error_message) *error_message = "N must be greater than 1";
        return false;
    }

    if (N % 2 == 0) {
        if (error_message) *error_message = "N must be odd (factor of 2 is trivial)";
        return false;
    }

    if (shor_is_prime(N)) {
        if (error_message) *error_message = "N is prime, cannot factor";
        return false;
    }

    uint64_t base, power;
    if (shor_is_perfect_power(N, &base, &power)) {
        if (error_message) *error_message = "N is a perfect power, use classical algorithm";
        return false;
    }

    return true;
}

shor_config_t shor_default_config(void) {
    return (shor_config_t){
        .num_trials = 100,
        .precision_bits = 0,  // Auto-calculate
        .use_semiclassical_qft = false,
        .use_approximate_qft = true,
        .approximate_qft_cutoff = 10,
        .use_gpu = false,
        .backend = NULL
    };
}

shor_state_t* shor_init_state(uint64_t N, uint64_t a, const shor_config_t* config) {
    shor_state_t* state = calloc(1, sizeof(shor_state_t));
    if (!state) return NULL;

    state->N = N;
    state->a = a;

    // Calculate number of qubits
    size_t n = 0;
    uint64_t temp = N;
    while (temp > 0) {
        n++;
        temp >>= 1;
    }

    state->work_qubits = n;
    state->precision_qubits = config && config->precision_bits > 0 ?
                              config->precision_bits : 2 * n + 1;
    state->num_qubits = state->precision_qubits + state->work_qubits;

    if (config) {
        state->config = *config;
    } else {
        state->config = shor_default_config();
    }

    // Initialize quantum state
    size_t dim = 1ULL << state->num_qubits;

    state->qstate = malloc(sizeof(QuantumState));
    if (!state->qstate) {
        free(state);
        return NULL;
    }

    state->qstate->amplitudes = calloc(dim, sizeof(ComplexFloat));
    if (!state->qstate->amplitudes) {
        free(state->qstate);
        free(state);
        return NULL;
    }

    state->qstate->num_qubits = state->num_qubits;
    state->qstate->dimension = dim;
    state->qstate->is_normalized = true;

    // Initialize to |0...0⟩|1⟩ (precision register = 0, work register = 1)
    // In our convention: precision qubits are MSBs, work qubits are LSBs
    // So |0...0⟩|1⟩ corresponds to index 1
    state->qstate->amplitudes[1] = COMPLEX_FLOAT_ONE;

    // Apply Hadamard to all precision qubits to create superposition
    // |ψ⟩ = (1/√2^n) Σ_x |x⟩|1⟩
    size_t precision_dim = 1ULL << state->precision_qubits;
    float norm = 1.0f / sqrtf((float)precision_dim);

    // Clear the initial state
    memset(state->qstate->amplitudes, 0, dim * sizeof(ComplexFloat));

    // Create uniform superposition over precision register with work = 1
    for (size_t x = 0; x < precision_dim; x++) {
        // Index: x << work_qubits | 1 (work register = 1)
        size_t idx = (x << state->work_qubits) | 1;
        state->qstate->amplitudes[idx].real = norm;
    }

    return state;
}

void shor_destroy_state(shor_state_t* state) {
    if (!state) return;

    if (state->qstate) {
        free(state->qstate->amplitudes);
        free(state->qstate);
    }

    free(state);
}

// ============================================================================
// Quantum Circuit Operations
// ============================================================================

/**
 * @brief Apply controlled modular multiplication: |x⟩|y⟩ → |x⟩|y * a^(2^k) mod N⟩
 *        when control bit x_k = 1
 */
static void apply_controlled_modmul(ComplexFloat* amplitudes, size_t dim,
                                     size_t precision_qubits, size_t work_qubits,
                                     size_t control_bit, uint64_t multiplier, uint64_t N) {
    if (!amplitudes || dim == 0) return;

    size_t work_dim = 1ULL << work_qubits;
    size_t control_mask = 1ULL << (work_qubits + control_bit);

    // Precompute modular multiplication table
    uint64_t* mul_table = malloc(work_dim * sizeof(uint64_t));
    if (!mul_table) return;

    for (size_t y = 0; y < work_dim; y++) {
        if (y < N) {
#if defined(__SIZEOF_INT128__)
            __uint128_t tmp = (__uint128_t)y * multiplier;
            mul_table[y] = (uint64_t)(tmp % N);
#else
            mul_table[y] = (uint64_t)fmod((double)y * (double)multiplier, (double)N);
#endif
        } else {
            mul_table[y] = y;  // Leave unchanged if y >= N
        }
    }

    // Create temporary storage for the transformation
    ComplexFloat* temp = calloc(dim, sizeof(ComplexFloat));
    if (!temp) {
        free(mul_table);
        return;
    }

    // Apply the controlled modular multiplication
    for (size_t idx = 0; idx < dim; idx++) {
        size_t x = idx >> work_qubits;  // Precision register value
        size_t y = idx & (work_dim - 1);  // Work register value

        // Check if this control bit is set
        if (idx & control_mask) {
            // Transform y → y * multiplier mod N
            size_t new_y = mul_table[y];
            size_t new_idx = (x << work_qubits) | new_y;
            temp[new_idx] = complex_float_add(temp[new_idx], amplitudes[idx]);
        } else {
            // No transformation
            temp[idx] = complex_float_add(temp[idx], amplitudes[idx]);
        }
    }

    // Copy back
    memcpy(amplitudes, temp, dim * sizeof(ComplexFloat));

    free(temp);
    free(mul_table);
}

qgt_error_t shor_apply_modexp(shor_state_t* state) {
    if (!state || !state->qstate || !state->qstate->amplitudes) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = state->qstate->dimension;
    ComplexFloat* amplitudes = state->qstate->amplitudes;

    // For each bit position k in the precision register,
    // apply controlled multiplication by a^(2^k) mod N
    uint64_t a_power = state->a;  // Start with a^(2^0) = a

    for (size_t k = 0; k < state->precision_qubits; k++) {
        apply_controlled_modmul(amplitudes, dim,
                                state->precision_qubits, state->work_qubits,
                                k, a_power, state->N);

        // Update a_power = a^(2^(k+1)) = (a^(2^k))^2 mod N
#if defined(__SIZEOF_INT128__)
        __uint128_t tmp = (__uint128_t)a_power * a_power;
        a_power = (uint64_t)(tmp % state->N);
#else
        a_power = (uint64_t)fmod((double)a_power * (double)a_power, (double)state->N);
#endif
    }

    return QGT_SUCCESS;
}

/**
 * @brief Apply controlled phase rotation for QFT
 */
static void apply_controlled_phase_qft(ComplexFloat* amplitudes, size_t dim,
                                        size_t work_qubits,
                                        size_t control_qubit, size_t target_qubit,
                                        double phase) {
    float cos_phase = (float)cos(phase);
    float sin_phase = (float)sin(phase);

    size_t control_mask = 1ULL << (work_qubits + control_qubit);
    size_t target_mask = 1ULL << (work_qubits + target_qubit);

    for (size_t idx = 0; idx < dim; idx++) {
        if ((idx & control_mask) && (idx & target_mask)) {
            ComplexFloat old = amplitudes[idx];
            amplitudes[idx].real = cos_phase * old.real - sin_phase * old.imag;
            amplitudes[idx].imag = sin_phase * old.real + cos_phase * old.imag;
        }
    }
}

/**
 * @brief Apply Hadamard gate to a precision qubit
 */
static void apply_hadamard_precision(ComplexFloat* amplitudes, size_t dim,
                                      size_t work_qubits, size_t qubit) {
    float inv_sqrt2 = 0.7071067811865475f;
    size_t mask = 1ULL << (work_qubits + qubit);

    for (size_t idx = 0; idx < dim; idx++) {
        if ((idx & mask) == 0) {
            size_t idx0 = idx;
            size_t idx1 = idx | mask;

            ComplexFloat a0 = amplitudes[idx0];
            ComplexFloat a1 = amplitudes[idx1];

            amplitudes[idx0].real = inv_sqrt2 * (a0.real + a1.real);
            amplitudes[idx0].imag = inv_sqrt2 * (a0.imag + a1.imag);
            amplitudes[idx1].real = inv_sqrt2 * (a0.real - a1.real);
            amplitudes[idx1].imag = inv_sqrt2 * (a0.imag - a1.imag);
        }
    }
}

/**
 * @brief Apply SWAP gate between two precision qubits
 */
static void apply_swap_precision(ComplexFloat* amplitudes, size_t dim,
                                  size_t work_qubits, size_t qubit1, size_t qubit2) {
    if (qubit1 == qubit2) return;

    size_t mask1 = 1ULL << (work_qubits + qubit1);
    size_t mask2 = 1ULL << (work_qubits + qubit2);

    for (size_t idx = 0; idx < dim; idx++) {
        size_t bit1 = (idx & mask1) ? 1 : 0;
        size_t bit2 = (idx & mask2) ? 1 : 0;

        if (bit1 != bit2) {
            size_t swapped = (idx ^ mask1) ^ mask2;
            if (idx < swapped) {
                ComplexFloat temp = amplitudes[idx];
                amplitudes[idx] = amplitudes[swapped];
                amplitudes[swapped] = temp;
            }
        }
    }
}

qgt_error_t shor_apply_inverse_qft(shor_state_t* state) {
    if (!state || !state->qstate || !state->qstate->amplitudes) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = state->qstate->dimension;
    ComplexFloat* amplitudes = state->qstate->amplitudes;
    size_t n = state->precision_qubits;
    size_t work_qubits = state->work_qubits;
    size_t cutoff = state->config.approximate_qft_cutoff;

    // Inverse QFT: apply inverse rotations then Hadamards, in reverse order
    for (size_t i = 0; i < n; i++) {
        size_t qubit = n - 1 - i;  // Work from MSB to LSB

        // Apply controlled rotations from previous qubits
        for (size_t j = 0; j < i; j++) {
            // Skip small rotations in approximate QFT
            if (state->config.use_approximate_qft && (i - j) > cutoff) continue;

            size_t control = n - 1 - j;
            double phase = -M_PI / (double)(1ULL << (i - j));
            apply_controlled_phase_qft(amplitudes, dim, work_qubits, control, qubit, phase);
        }

        // Apply Hadamard
        apply_hadamard_precision(amplitudes, dim, work_qubits, qubit);
    }

    // Reverse qubit order via swaps
    for (size_t i = 0; i < n / 2; i++) {
        apply_swap_precision(amplitudes, dim, work_qubits, i, n - 1 - i);
    }

    return QGT_SUCCESS;
}

qgt_error_t shor_measure_phase(shor_state_t* state, double* measured_phase) {
    if (!state || !state->qstate || !state->qstate->amplitudes || !measured_phase) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    ComplexFloat* amplitudes = state->qstate->amplitudes;
    size_t precision_dim = 1ULL << state->precision_qubits;
    size_t work_dim = 1ULL << state->work_qubits;

    // Compute probability distribution over precision register
    double* probabilities = calloc(precision_dim, sizeof(double));
    if (!probabilities) return QGT_ERROR_MEMORY_ALLOCATION;

    double total_prob = 0.0;
    for (size_t x = 0; x < precision_dim; x++) {
        for (size_t y = 0; y < work_dim; y++) {
            size_t idx = (x << state->work_qubits) | y;
            double prob = (double)(amplitudes[idx].real * amplitudes[idx].real +
                                   amplitudes[idx].imag * amplitudes[idx].imag);
            probabilities[x] += prob;
        }
        total_prob += probabilities[x];
    }

    // Normalize
    if (total_prob > 1e-10) {
        for (size_t x = 0; x < precision_dim; x++) {
            probabilities[x] /= total_prob;
        }
    }

    // Sample from distribution
    double r = (double)rand() / (double)RAND_MAX;
    double cumulative = 0.0;
    size_t measured_x = 0;

    for (size_t x = 0; x < precision_dim; x++) {
        cumulative += probabilities[x];
        if (r <= cumulative) {
            measured_x = x;
            break;
        }
    }

    free(probabilities);

    // Convert to phase: φ = x / 2^n
    *measured_phase = (double)measured_x / (double)precision_dim;

    return QGT_SUCCESS;
}

// ============================================================================
// Main Factorization Algorithm
// ============================================================================

uint64_t shor_find_order(uint64_t a, uint64_t N, const shor_config_t* config) {
    shor_config_t cfg = config ? *config : shor_default_config();

    // Initialize quantum state
    shor_state_t* state = shor_init_state(N, a, &cfg);
    if (!state) return 0;

    // Apply modular exponentiation circuit
    if (shor_apply_modexp(state) != QGT_SUCCESS) {
        shor_destroy_state(state);
        return 0;
    }

    // Apply inverse QFT to precision register
    if (shor_apply_inverse_qft(state) != QGT_SUCCESS) {
        shor_destroy_state(state);
        return 0;
    }

    // Measure and extract phase
    double measured_phase = 0.0;
    if (shor_measure_phase(state, &measured_phase) != QGT_SUCCESS) {
        shor_destroy_state(state);
        return 0;
    }

    shor_destroy_state(state);

    // Use continued fractions to find the order
    size_t precision_dim = 1ULL << (cfg.precision_bits > 0 ? cfg.precision_bits :
                                     shor_required_qubits(N) * 2 / 3 + 1);
    uint64_t numerator = (uint64_t)(measured_phase * (double)precision_dim + 0.5);

    cf_convergent_t convergents[64];
    size_t num_convergents = shor_continued_fraction(numerator, precision_dim, N,
                                                      convergents, 64);

    // Try each convergent denominator as a potential order
    for (size_t i = 0; i < num_convergents; i++) {
        uint64_t candidate = convergents[i].denominator;
        if (candidate == 0) continue;

        // Verify: a^candidate ≡ 1 (mod N)
        if (shor_mod_exp(a, candidate, N) == 1) {
            return candidate;
        }

        // Try small multiples
        for (uint64_t k = 2; k <= 4 && candidate * k <= N; k++) {
            if (shor_mod_exp(a, candidate * k, N) == 1) {
                return candidate * k;
            }
        }
    }

    return 0;  // Failed to find order
}

shor_result_t* shor_factor(uint64_t N, const shor_config_t* config) {
    shor_result_t* result = calloc(1, sizeof(shor_result_t));
    if (!result) return NULL;

    clock_t start_time = clock();

    // Validate input
    char* error_msg = NULL;
    if (!shor_validate_input(N, &error_msg)) {
        result->success = false;
        result->error_message = error_msg ? strdup(error_msg) : NULL;
        return result;
    }

    shor_config_t cfg = config ? *config : shor_default_config();

    // Try multiple random bases
    for (size_t trial = 0; trial < cfg.num_trials; trial++) {
        result->num_trials = trial + 1;

        // Choose random a coprime to N
        uint64_t a = shor_random_coprime(N);
        if (a == 0) continue;

        // Quick check: gcd(a, N) might give a factor directly
        uint64_t g = shor_gcd(a, N);
        if (g > 1 && g < N) {
            result->success = true;
            result->factor1 = g;
            result->factor2 = N / g;
            result->order = 0;  // Not used
            break;
        }

        // Find the order of a modulo N
        uint64_t r = shor_find_order(a, N, &cfg);
        if (r == 0) continue;

        result->order = r;

        // Order must be even for the algorithm to work
        if (r % 2 != 0) continue;

        // Compute a^(r/2) mod N
        uint64_t half_power = shor_mod_exp(a, r / 2, N);

        // Check that a^(r/2) ≢ -1 (mod N)
        if (half_power == N - 1) continue;

        // Extract factors: gcd(a^(r/2) ± 1, N)
        uint64_t f1 = shor_gcd(half_power + 1, N);
        uint64_t f2 = shor_gcd(half_power - 1 + N, N);  // +N to handle unsigned

        // Check for non-trivial factors
        if (f1 > 1 && f1 < N) {
            result->success = true;
            result->factor1 = f1;
            result->factor2 = N / f1;
            break;
        }
        if (f2 > 1 && f2 < N) {
            result->success = true;
            result->factor1 = f2;
            result->factor2 = N / f2;
            break;
        }
    }

    clock_t end_time = clock();
    result->execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (!result->success && !result->error_message) {
        result->error_message = strdup("Failed to find factors within trial limit");
    }

    return result;
}

void shor_destroy_result(shor_result_t* result) {
    if (!result) return;
    free(result->error_message);
    free(result);
}

void shor_print_result(const shor_result_t* result) {
    if (!result) return;

    printf("Shor's Algorithm Result:\n");
    printf("  Success: %s\n", result->success ? "YES" : "NO");

    if (result->success) {
        printf("  Factor 1: %llu\n", (unsigned long long)result->factor1);
        printf("  Factor 2: %llu\n", (unsigned long long)result->factor2);
        printf("  Order found: %llu\n", (unsigned long long)result->order);
    } else if (result->error_message) {
        printf("  Error: %s\n", result->error_message);
    }

    printf("  Trials used: %zu\n", result->num_trials);
    printf("  Execution time: %.4f seconds\n", result->execution_time);
}
