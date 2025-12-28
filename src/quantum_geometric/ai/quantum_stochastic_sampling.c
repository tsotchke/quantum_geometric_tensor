/**
 * @file quantum_stochastic_sampling.c
 * @brief Quantum stochastic sampling operations for AI/ML applications
 *
 * Cross-platform implementation with SIMD acceleration (AVX2/NEON)
 * and GPU support for large-scale machine learning applications.
 */

#include "quantum_geometric/ai/quantum_stochastic_sampling.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/core/performance_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Platform-specific SIMD includes
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#endif

// Constants for optimal performance
#define BLOCK_SIZE 64
#define CACHE_LINE_SIZE 64
#define MIN_SIZE_FOR_GPU 1024
#define MAX_THREADS 32

// Performance tracking
static struct {
    size_t total_samples;
    size_t total_measurements;
    double total_sampling_time;
    double avg_sampling_time;
    size_t peak_memory;
    size_t current_memory;
} perf_stats = {0};

// Memory pool for sampling operations
static MemoryPool* sampling_pool = NULL;

// Initialize sampling pool
static int init_sampling_pool(void) {
    if (!sampling_pool) {
        PoolConfig config = {
            .min_block_size = QG_MIN_BLOCK_SIZE,
            .alignment = CACHE_LINE_SIZE,
            .num_size_classes = QG_NUM_SIZE_CLASSES,
            .growth_factor = 2.0f,
            .prefetch_distance = 4,
            .use_huge_pages = false,
            .cache_local_free_lists = true,
            .max_blocks_per_class = 1024,
            .thread_cache_size = 64,
            .enable_stats = true
        };
        sampling_pool = create_memory_pool(&config);
    }
    return sampling_pool != NULL ? QGT_SUCCESS : QGT_ERROR_MEMORY_ALLOCATION;
}

// Helper function to track memory usage
static void update_memory_stats(size_t size, bool allocate) {
    if (allocate) {
        perf_stats.current_memory += size;
        if (perf_stats.current_memory > perf_stats.peak_memory) {
            perf_stats.peak_memory = perf_stats.current_memory;
        }
    } else {
        if (perf_stats.current_memory >= size) {
            perf_stats.current_memory -= size;
        }
    }
}

int qg_sampling_init(sampling_state_t* state, const sampling_config_t* config) {
    if (!state || !config || config->num_samples == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Initialize memory pool
    int status = init_sampling_pool();
    if (status != QGT_SUCCESS) {
        return status;
    }

    // Allocate state vectors from pool
    size_t alloc_size = config->num_samples * sizeof(float);
    state->state_vector = (float*)pool_allocate(sampling_pool, alloc_size);
    state->probabilities = (float*)pool_allocate(sampling_pool, alloc_size);
    state->cumulative_probs = (float*)pool_allocate(sampling_pool, alloc_size);

    if (!state->state_vector || !state->probabilities || !state->cumulative_probs) {
        qg_sampling_cleanup(state);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Initialize to zero
    memset(state->state_vector, 0, alloc_size);
    memset(state->probabilities, 0, alloc_size);
    memset(state->cumulative_probs, 0, alloc_size);

    state->state_dim = config->num_samples;
    state->normalized = false;
    update_memory_stats(3 * alloc_size, true);

    // Set random seed if provided
    if (config->seed != 0) {
        srand(config->seed);
    }

    return QGT_SUCCESS;
}

void qg_sampling_cleanup(sampling_state_t* state) {
    if (!state) return;

    size_t alloc_size = state->state_dim * sizeof(float);

    if (state->state_vector) {
        update_memory_stats(alloc_size, false);
        pool_free(sampling_pool, state->state_vector);
    }
    if (state->probabilities) {
        update_memory_stats(alloc_size, false);
        pool_free(sampling_pool, state->probabilities);
    }
    if (state->cumulative_probs) {
        update_memory_stats(alloc_size, false);
        pool_free(sampling_pool, state->cumulative_probs);
    }

    state->state_vector = NULL;
    state->probabilities = NULL;
    state->cumulative_probs = NULL;
    state->state_dim = 0;
    state->normalized = false;
}

int qg_sampling_prepare_state(sampling_state_t* state,
                              const float* initial_state,
                              size_t state_size) {
    if (!state || !initial_state || state_size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    if (state_size > state->state_dim) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Use GPU for large states
    if (state_size >= MIN_SIZE_FOR_GPU && is_gpu_available()) {
        return qg_sampling_prepare_state_gpu(state, initial_state, state_size);
    }

    // Copy state using SIMD when available
#if defined(USE_AVX2)
    size_t simd_size = state_size / 8;
    for (size_t i = 0; i < simd_size; i++) {
        __m256 src = _mm256_loadu_ps(&initial_state[i * 8]);
        _mm256_storeu_ps(&state->state_vector[i * 8], src);
    }
    // Handle remaining elements
    for (size_t i = simd_size * 8; i < state_size; i++) {
        state->state_vector[i] = initial_state[i];
    }
#elif defined(USE_NEON)
    size_t simd_size = state_size / 4;
    for (size_t i = 0; i < simd_size; i++) {
        float32x4_t src = vld1q_f32(&initial_state[i * 4]);
        vst1q_f32(&state->state_vector[i * 4], src);
    }
    // Handle remaining elements
    for (size_t i = simd_size * 4; i < state_size; i++) {
        state->state_vector[i] = initial_state[i];
    }
#else
    // Scalar fallback
    memcpy(state->state_vector, initial_state, state_size * sizeof(float));
#endif

    return QGT_SUCCESS;
}

int qg_sampling_prepare_complex(sampling_state_t* state,
                                const float* amplitudes_real,
                                const float* amplitudes_imag,
                                size_t state_size) {
    if (!state || !amplitudes_real || !amplitudes_imag || state_size == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    if (state_size > state->state_dim) {
        return QGT_ERROR_DIMENSION_MISMATCH;
    }

    // Compute probabilities from complex amplitudes: P(i) = |a_i|^2 = real^2 + imag^2
#if defined(USE_AVX2)
    size_t simd_size = state_size / 8;
    for (size_t i = 0; i < simd_size; i++) {
        __m256 real_vec = _mm256_loadu_ps(&amplitudes_real[i * 8]);
        __m256 imag_vec = _mm256_loadu_ps(&amplitudes_imag[i * 8]);

        // Compute |amplitude|^2 = real^2 + imag^2
        __m256 real_sq = _mm256_mul_ps(real_vec, real_vec);
        __m256 imag_sq = _mm256_mul_ps(imag_vec, imag_vec);
        __m256 prob = _mm256_add_ps(real_sq, imag_sq);

        _mm256_storeu_ps(&state->probabilities[i * 8], prob);
    }
    // Handle remaining elements
    for (size_t i = simd_size * 8; i < state_size; i++) {
        float real = amplitudes_real[i];
        float imag = amplitudes_imag[i];
        state->probabilities[i] = real * real + imag * imag;
    }
#elif defined(USE_NEON)
    size_t simd_size = state_size / 4;
    for (size_t i = 0; i < simd_size; i++) {
        float32x4_t real_vec = vld1q_f32(&amplitudes_real[i * 4]);
        float32x4_t imag_vec = vld1q_f32(&amplitudes_imag[i * 4]);

        // Compute |amplitude|^2 = real^2 + imag^2
        float32x4_t real_sq = vmulq_f32(real_vec, real_vec);
        float32x4_t imag_sq = vmulq_f32(imag_vec, imag_vec);
        float32x4_t prob = vaddq_f32(real_sq, imag_sq);

        vst1q_f32(&state->probabilities[i * 4], prob);
    }
    // Handle remaining elements
    for (size_t i = simd_size * 4; i < state_size; i++) {
        float real = amplitudes_real[i];
        float imag = amplitudes_imag[i];
        state->probabilities[i] = real * real + imag * imag;
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < state_size; i++) {
        float real = amplitudes_real[i];
        float imag = amplitudes_imag[i];
        state->probabilities[i] = real * real + imag * imag;
    }
#endif

    // Also store the state vector (use probabilities as state for now)
    memcpy(state->state_vector, state->probabilities, state_size * sizeof(float));

    // Mark as not normalized - caller should call qg_sampling_normalize
    state->normalized = false;

    return QGT_SUCCESS;
}

int qg_sampling_update_probabilities(sampling_state_t* state,
                                     const float* new_probs,
                                     size_t prob_size) {
    if (!state || !new_probs || prob_size != state->state_dim) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Use GPU for large states
    if (prob_size >= MIN_SIZE_FOR_GPU && is_gpu_available()) {
        return qg_sampling_update_probabilities_gpu(state, new_probs, prob_size);
    }

    // Copy probabilities
#if defined(USE_AVX2)
    size_t simd_size = prob_size / 8;
    for (size_t i = 0; i < simd_size; i++) {
        __m256 src = _mm256_loadu_ps(&new_probs[i * 8]);
        _mm256_storeu_ps(&state->probabilities[i * 8], src);
    }
    for (size_t i = simd_size * 8; i < prob_size; i++) {
        state->probabilities[i] = new_probs[i];
    }
#elif defined(USE_NEON)
    size_t simd_size = prob_size / 4;
    for (size_t i = 0; i < simd_size; i++) {
        float32x4_t src = vld1q_f32(&new_probs[i * 4]);
        vst1q_f32(&state->probabilities[i * 4], src);
    }
    for (size_t i = simd_size * 4; i < prob_size; i++) {
        state->probabilities[i] = new_probs[i];
    }
#else
    memcpy(state->probabilities, new_probs, prob_size * sizeof(float));
#endif

    // Update cumulative probabilities
    float cumsum = 0.0f;
    for (size_t i = 0; i < prob_size; i++) {
        cumsum += state->probabilities[i];
        state->cumulative_probs[i] = cumsum;
    }

    // Normalize if needed
    if (fabsf(cumsum - 1.0f) > 1e-6f && cumsum > 0.0f) {
        float inv_sum = 1.0f / cumsum;
#if defined(USE_AVX2)
        __m256 norm = _mm256_set1_ps(inv_sum);
        size_t simd_size = prob_size / 8;
        for (size_t i = 0; i < simd_size; i++) {
            __m256 probs = _mm256_loadu_ps(&state->cumulative_probs[i * 8]);
            _mm256_storeu_ps(&state->cumulative_probs[i * 8], _mm256_mul_ps(probs, norm));
        }
        for (size_t i = simd_size * 8; i < prob_size; i++) {
            state->cumulative_probs[i] *= inv_sum;
        }
#elif defined(USE_NEON)
        float32x4_t norm = vdupq_n_f32(inv_sum);
        size_t simd_size = prob_size / 4;
        for (size_t i = 0; i < simd_size; i++) {
            float32x4_t probs = vld1q_f32(&state->cumulative_probs[i * 4]);
            vst1q_f32(&state->cumulative_probs[i * 4], vmulq_f32(probs, norm));
        }
        for (size_t i = simd_size * 4; i < prob_size; i++) {
            state->cumulative_probs[i] *= inv_sum;
        }
#else
        for (size_t i = 0; i < prob_size; i++) {
            state->cumulative_probs[i] *= inv_sum;
        }
#endif
        state->normalized = true;
    }

    return QGT_SUCCESS;
}

int qg_sampling_normalize(sampling_state_t* state) {
    if (!state || !state->probabilities) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Compute sum
    float sum = 0.0f;
    for (size_t i = 0; i < state->state_dim; i++) {
        sum += state->probabilities[i];
    }

    if (sum <= 0.0f) {
        return QGT_ERROR_NUMERICAL_INSTABILITY;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < state->state_dim; i++) {
        state->probabilities[i] *= inv_sum;
    }

    // Update cumulative
    float cumsum = 0.0f;
    for (size_t i = 0; i < state->state_dim; i++) {
        cumsum += state->probabilities[i];
        state->cumulative_probs[i] = cumsum;
    }

    state->normalized = true;
    return QGT_SUCCESS;
}

// Binary search for sampling
static size_t binary_search_cumulative(const float* cumulative, size_t n, float target) {
    size_t left = 0;
    size_t right = n;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (cumulative[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left < n ? left : n - 1;
}

int qg_sampling_sample(sampling_state_t* state,
                       size_t num_samples,
                       sampling_result_t* result) {
    if (!state || !result || num_samples == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Allocate result arrays
    result->samples = (size_t*)malloc(num_samples * sizeof(size_t));
    result->weights = (float*)malloc(num_samples * sizeof(float));
    if (!result->samples || !result->weights) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Use GPU for large sample counts
    if (num_samples >= MIN_SIZE_FOR_GPU && is_gpu_available()) {
        int status = qg_sampling_sample_gpu(state, num_samples, result);
        if (status == QGT_SUCCESS) {
            return QGT_SUCCESS;
        }
        // Fall through to CPU if GPU fails
    }

    // Draw samples using binary search
    for (size_t i = 0; i < num_samples; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        size_t idx = binary_search_cumulative(state->cumulative_probs, state->state_dim, r);
        result->samples[i] = idx;
        result->weights[i] = 1.0f;
    }

    result->num_samples = num_samples;
    perf_stats.total_samples += num_samples;

    return QGT_SUCCESS;
}

int qg_sampling_importance(sampling_state_t* state,
                           const float* proposal_probs,
                           size_t num_samples,
                           sampling_result_t* result) {
    if (!state || !proposal_probs || !result || num_samples == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Allocate result arrays
    result->samples = (size_t*)malloc(num_samples * sizeof(size_t));
    result->weights = (float*)malloc(num_samples * sizeof(float));
    if (!result->samples || !result->weights) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Build cumulative proposal distribution
    float* proposal_cumulative = (float*)malloc(state->state_dim * sizeof(float));
    if (!proposal_cumulative) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    float cumsum = 0.0f;
    for (size_t i = 0; i < state->state_dim; i++) {
        cumsum += proposal_probs[i];
        proposal_cumulative[i] = cumsum;
    }

    // Normalize proposal
    if (cumsum > 0.0f) {
        float inv_sum = 1.0f / cumsum;
        for (size_t i = 0; i < state->state_dim; i++) {
            proposal_cumulative[i] *= inv_sum;
        }
    }

    // Draw samples with importance weights
    for (size_t i = 0; i < num_samples; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        size_t idx = binary_search_cumulative(proposal_cumulative, state->state_dim, r);
        result->samples[i] = idx;

        // Importance weight = target / proposal
        float prop = proposal_probs[idx];
        float target = state->probabilities[idx];
        result->weights[i] = (prop > 1e-10f) ? (target / prop) : 0.0f;
    }

    free(proposal_cumulative);
    result->num_samples = num_samples;
    perf_stats.total_samples += num_samples;

    return QGT_SUCCESS;
}

int qg_sampling_metropolis(sampling_state_t* state,
                           const float* proposal_probs,
                           size_t num_samples,
                           double temperature,
                           sampling_result_t* result) {
    if (!state || !proposal_probs || !result || num_samples == 0 || temperature <= 0.0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    // Allocate result arrays
    result->samples = (size_t*)malloc(num_samples * sizeof(size_t));
    result->weights = (float*)malloc(num_samples * sizeof(float));
    if (!result->samples || !result->weights) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Build proposal cumulative
    float* proposal_cumulative = (float*)malloc(state->state_dim * sizeof(float));
    if (!proposal_cumulative) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    float cumsum = 0.0f;
    for (size_t i = 0; i < state->state_dim; i++) {
        cumsum += proposal_probs[i];
        proposal_cumulative[i] = cumsum;
    }
    if (cumsum > 0.0f) {
        for (size_t i = 0; i < state->state_dim; i++) {
            proposal_cumulative[i] /= cumsum;
        }
    }

    // Metropolis-Hastings sampling
    double inv_temp = 1.0 / temperature;
    size_t current_state = 0;

    // Initialize with a random state
    float r = (float)rand() / (float)RAND_MAX;
    current_state = binary_search_cumulative(proposal_cumulative, state->state_dim, r);

    for (size_t i = 0; i < num_samples; i++) {
        // Propose new state
        r = (float)rand() / (float)RAND_MAX;
        size_t proposed_state = binary_search_cumulative(proposal_cumulative, state->state_dim, r);

        // Compute acceptance probability
        double current_prob = state->probabilities[current_state];
        double proposed_prob = state->probabilities[proposed_state];
        double acceptance = exp(inv_temp * (log(proposed_prob + 1e-10) - log(current_prob + 1e-10)));

        // Accept or reject
        float u = (float)rand() / (float)RAND_MAX;
        if (u < acceptance) {
            current_state = proposed_state;
        }

        result->samples[i] = current_state;
        result->weights[i] = 1.0f;
    }

    free(proposal_cumulative);
    result->num_samples = num_samples;
    perf_stats.total_samples += num_samples;

    return QGT_SUCCESS;
}

int qg_sampling_stratified(sampling_state_t* state,
                           size_t num_strata,
                           size_t samples_per_stratum,
                           sampling_result_t* result) {
    if (!state || !result || num_strata == 0 || samples_per_stratum == 0) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    size_t total_samples = num_strata * samples_per_stratum;

    // Allocate result arrays
    result->samples = (size_t*)malloc(total_samples * sizeof(size_t));
    result->weights = (float*)malloc(total_samples * sizeof(float));
    if (!result->samples || !result->weights) {
        free(result->samples);
        free(result->weights);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    float stratum_width = 1.0f / (float)num_strata;
    size_t sample_idx = 0;

    for (size_t s = 0; s < num_strata; s++) {
        float stratum_start = s * stratum_width;

        for (size_t i = 0; i < samples_per_stratum; i++) {
            // Sample within stratum
            float r = stratum_start + ((float)rand() / (float)RAND_MAX) * stratum_width;
            size_t idx = binary_search_cumulative(state->cumulative_probs, state->state_dim, r);
            result->samples[sample_idx] = idx;
            result->weights[sample_idx] = 1.0f;
            sample_idx++;
        }
    }

    result->num_samples = total_samples;
    perf_stats.total_samples += total_samples;

    return QGT_SUCCESS;
}

// GPU stub functions - to be implemented with Metal/CUDA
int qg_sampling_prepare_state_gpu(sampling_state_t* state,
                                  const float* initial_state,
                                  size_t state_size) {
    (void)state;
    (void)initial_state;
    (void)state_size;
    return QGT_ERROR_NOT_IMPLEMENTED;
}

int qg_sampling_update_probabilities_gpu(sampling_state_t* state,
                                         const float* new_probs,
                                         size_t prob_size) {
    (void)state;
    (void)new_probs;
    (void)prob_size;
    return QGT_ERROR_NOT_IMPLEMENTED;
}

int qg_sampling_sample_gpu(sampling_state_t* state,
                           size_t num_samples,
                           sampling_result_t* result) {
    (void)state;
    (void)num_samples;
    (void)result;
    return QGT_ERROR_NOT_IMPLEMENTED;
}

// is_gpu_available() - Use canonical implementation from quantum_field_gpu_monitor.c

int qg_sampling_get_stats(sampling_stats_t* stats) {
    if (!stats) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    stats->total_samples = perf_stats.total_samples;
    stats->total_measurements = perf_stats.total_measurements;
    stats->total_sampling_time = perf_stats.total_sampling_time;
    stats->avg_sampling_time = perf_stats.total_samples > 0
        ? perf_stats.total_sampling_time / (double)perf_stats.total_samples
        : 0.0;
    stats->peak_memory = perf_stats.peak_memory;

    return QGT_SUCCESS;
}

void qg_sampling_reset_stats(void) {
    memset(&perf_stats, 0, sizeof(perf_stats));
}

void qg_sampling_free_result(sampling_result_t* result) {
    if (!result) return;

    free(result->samples);
    free(result->weights);
    result->samples = NULL;
    result->weights = NULL;
    result->num_samples = 0;
}
