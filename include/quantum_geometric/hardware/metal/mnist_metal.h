/**
 * @file mnist_metal.h
 * @brief Metal GPU backend for MNIST quantum geometric classification
 *
 * Provides Metal-accelerated quantum geometric operations for MNIST
 * image classification on Apple Silicon (M1/M2).
 */

#ifndef MNIST_METAL_H
#define MNIST_METAL_H

#include <stddef.h>
#include <stdbool.h>

// C/C++ complex type compatibility
#ifdef __cplusplus
    #define complex _Complex
    extern "C" {
#else
    #include <complex.h>
#endif

// ============================================================================
// Constants
// ============================================================================

/** Maximum number of qubits for quantum state representation */
#ifndef MAX_QUBITS
#define MAX_QUBITS 10
#endif

/** MNIST image size (28x28) */
#ifndef MNIST_IMAGE_SIZE
#define MNIST_IMAGE_SIZE 784
#endif

/** Number of MNIST classes (0-9) */
#ifndef MNIST_NUM_CLASSES
#define MNIST_NUM_CLASSES 10
#endif

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Quantum geometric state for MNIST processing
 *
 * Represents a quantum state with geometric structure for
 * encoding and classifying MNIST images.
 */
typedef struct QuantumGeometricState {
    complex double* amplitudes;     /**< Quantum state amplitudes */
    size_t state_size;              /**< Number of amplitudes (2^n_qubits) */
    size_t n_qubits;                /**< Number of qubits */
    double* metric_tensor;          /**< Geometric metric tensor */
    size_t metric_dim;              /**< Dimension of metric tensor */
    double* connection;             /**< Geometric connection */
    double curvature;               /**< Scalar curvature */
    bool is_normalized;             /**< Whether state is normalized */
} QuantumGeometricState;

// ============================================================================
// Metal Device Management
// ============================================================================

/**
 * @brief Initialize Metal device and resources
 *
 * Sets up Metal device, command queue, and compiles shader library.
 * Must be called before any other Metal MNIST functions.
 *
 * @return 0 on success, error code on failure
 */
int init_metal_device(void);

/**
 * @brief Compile Metal shader library
 *
 * Compiles the Metal shaders from source file. Called internally
 * by init_metal_device().
 *
 * @return 0 on success, error code on failure
 */
int compile_metal_shaders(void);

/**
 * @brief Create Metal buffers for processing
 *
 * Allocates GPU memory buffers for quantum state processing.
 *
 * @param max_batch_size Maximum batch size to support
 * @return 0 on success, error code on failure
 */
int create_metal_buffers(size_t max_batch_size);

/**
 * @brief Release Metal buffers
 *
 * Frees GPU memory buffers.
 */
void release_metal_buffers(void);

/**
 * @brief Cleanup Metal resources
 *
 * Releases all Metal resources including device, command queue,
 * and shader pipelines. Call when done with Metal operations.
 */
void cleanup_metal_device(void);

// ============================================================================
// Quantum Geometric Operations
// ============================================================================

/**
 * @brief Encode classical data into quantum state using Metal GPU
 *
 * Transforms MNIST image data into quantum state amplitudes using
 * amplitude encoding with geometric phase factors.
 *
 * @param state Quantum state to store encoded data
 * @param data Input image data (normalized pixel values)
 * @param batch_size Number of images in batch
 * @param input_dim Dimension of input (should be MNIST_IMAGE_SIZE)
 * @return 0 on success, error code on failure
 */
int encode_quantum_state_metal(
    QuantumGeometricState* state,
    const float* data,
    size_t batch_size,
    float input_dim);

/**
 * @brief Apply geometric transformations using Metal GPU
 *
 * Applies quantum geometric transformations including metric tensor
 * computation and parallel transport.
 *
 * @param state Quantum state to transform
 * @param parameters Transformation parameters
 * @param batch_size Number of states in batch
 * @param latent_dim Dimension of latent space
 * @return 0 on success, error code on failure
 */
int apply_geometric_transform_metal(
    QuantumGeometricState* state,
    const float* parameters,
    size_t batch_size,
    float latent_dim);

/**
 * @brief Measure quantum state to get class probabilities using Metal GPU
 *
 * Computes measurement probabilities for each class by projecting
 * quantum state onto computational basis.
 *
 * @param state Quantum state to measure
 * @param probabilities Output array for class probabilities
 * @param batch_size Number of states in batch
 * @param num_classes Number of output classes
 * @return 0 on success, error code on failure
 */
int measure_quantum_state_metal(
    QuantumGeometricState* state,
    float* probabilities,
    size_t batch_size,
    float num_classes);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if Metal is available on this system
 *
 * @return true if Metal is available, false otherwise
 */
bool is_metal_available(void);

/**
 * @brief Get Metal device name
 *
 * @return Device name string or NULL if not initialized
 */
const char* get_metal_device_name_mnist(void);

#ifdef __cplusplus
}
#endif

#endif // MNIST_METAL_H
