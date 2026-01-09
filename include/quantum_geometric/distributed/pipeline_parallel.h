/**
 * @file pipeline_parallel.h
 * @brief Pipeline parallelism for distributed quantum computing
 *
 * Provides infrastructure for pipelined execution of quantum operations
 * across multiple stages, enabling efficient processing of batched data.
 */

#ifndef PIPELINE_PARALLEL_H
#define PIPELINE_PARALLEL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Constants
// ============================================================================

#define MAX_PIPELINE_STAGES 8
#define MAX_BATCH_SIZE 1024
#define QUEUE_SIZE 4
#define QUANTUM_CHUNK_SIZE 4096

// ============================================================================
// Quantum Operation Flags
// ============================================================================

#define QUANTUM_OPTIMIZE_AGGRESSIVE   0x01
#define QUANTUM_USE_PIPELINE          0x02
#define QUANTUM_USE_FORWARD           0x04
#define QUANTUM_USE_BACKWARD          0x08
#define QUANTUM_USE_GRADIENT          0x10

// ============================================================================
// Quantum Circuit Types
// ============================================================================

typedef enum {
    QUANTUM_CIRCUIT_OPTIMAL = 0,
    QUANTUM_CIRCUIT_FAST,
    QUANTUM_CIRCUIT_ACCURATE,
    QUANTUM_CIRCUIT_BALANCED
} QuantumCircuitType;

typedef enum {
    QUANTUM_ERROR_NONE = 0,
    QUANTUM_ERROR_ADAPTIVE,
    QUANTUM_ERROR_STABILIZER,
    QUANTUM_ERROR_SURFACE_CODE
} QuantumErrorCorrection;

typedef enum {
    QUANTUM_OPT_NONE = 0,
    QUANTUM_OPT_BASIC,
    QUANTUM_OPT_MODERATE,
    QUANTUM_OPT_AGGRESSIVE
} QuantumOptLevel;

typedef enum {
    QUANTUM_PIPELINE_BASIC = 0,
    QUANTUM_PIPELINE_OPTIMAL,
    QUANTUM_PIPELINE_DISTRIBUTED
} QuantumPipelineType;

typedef enum {
    QUANTUM_FORWARD_BASIC = 0,
    QUANTUM_FORWARD_OPTIMAL
} QuantumForwardType;

typedef enum {
    QUANTUM_BACKWARD_BASIC = 0,
    QUANTUM_BACKWARD_OPTIMAL
} QuantumBackwardType;

typedef enum {
    QUANTUM_GRADIENT_BASIC = 0,
    QUANTUM_GRADIENT_OPTIMAL
} QuantumGradientType;

// ============================================================================
// Pipeline Function Type
// ============================================================================

/**
 * @brief Function pointer type for pipeline stage operations
 *
 * @param input Input buffer
 * @param output Output buffer
 * @param size Buffer size in bytes
 * @param args User-provided arguments
 */
typedef void (*PipelineFunction)(const void* input, void* output, size_t size, void* args);

// ============================================================================
// Pipeline Stage Structure
// ============================================================================

/**
 * @brief Pipeline stage with input/output buffers and threading
 */
typedef struct {
    void* input_buffer;           /**< Input buffer */
    void* output_buffer;          /**< Output buffer */
    size_t buffer_size;           /**< Buffer size in bytes */
    PipelineFunction function;    /**< Stage processing function */
    void* function_args;          /**< Function arguments */
    sem_t input_ready;            /**< Semaphore for input readiness */
    sem_t output_ready;           /**< Semaphore for output readiness */
    pthread_t thread;             /**< Stage thread handle */
    bool active;                  /**< Whether stage is active */
} PipelineStage;

// ============================================================================
// Pipeline Context Structure
// ============================================================================

/**
 * @brief Pipeline context containing stages and configuration
 */
typedef struct PipelineContext {
    PipelineStage stages[MAX_PIPELINE_STAGES];  /**< Array of pipeline stages */
    size_t num_stages;            /**< Number of active stages */
    size_t batch_size;            /**< Batch size for processing */
    pthread_mutex_t mutex;        /**< Mutex for thread safety */
    bool initialized;             /**< Whether pipeline is initialized */
} PipelineContext;

// ============================================================================
// Layer Configuration
// ============================================================================

/**
 * @brief Configuration for a neural network layer
 */
typedef struct LayerConfig {
    size_t input_size;          /**< Input dimension */
    size_t output_size;         /**< Output dimension */
    size_t hidden_size;         /**< Hidden dimension (if applicable) */
    double* weights;            /**< Layer weights */
    double* biases;             /**< Layer biases */
    double* gradients;          /**< Gradient storage */
    int activation;             /**< Activation function type */
    bool use_bias;              /**< Whether to use bias terms */
    void* custom_data;          /**< Custom layer data */
} LayerConfig;

// ============================================================================
// Quantum System Types
// ============================================================================

/**
 * @brief Quantum system context for pipeline operations
 */
typedef struct quantum_system_t {
    size_t num_qubits;          /**< Number of qubits */
    size_t state_size;          /**< Size of state vector */
    void* state_vector;         /**< Quantum state vector */
    void* backend;              /**< Backend handle (simulator/hardware) */
    int flags;                  /**< Operation flags */
} quantum_system_t;

/**
 * @brief Quantum circuit for computations
 */
typedef struct quantum_circuit_t {
    size_t num_qubits;          /**< Number of qubits */
    size_t num_gates;           /**< Number of gates */
    void* gates;                /**< Gate sequence */
    QuantumCircuitType type;    /**< Circuit type */
} quantum_circuit_t;

/**
 * @brief Quantum register for data storage
 */
typedef struct quantum_register_t {
    void* data;                 /**< Register data */
    size_t size;                /**< Data size in bytes */
    size_t num_qubits;          /**< Number of qubits */
    bool allocated;             /**< Whether memory is allocated */
} quantum_register_t;

/**
 * @brief Quantum pipeline for staged execution
 */
typedef struct quantum_pipeline_t {
    size_t num_stages;          /**< Number of pipeline stages */
    void* stages;               /**< Stage array */
    void* buffers;              /**< Inter-stage buffers */
    QuantumPipelineType type;   /**< Pipeline type */
} quantum_pipeline_t;

/**
 * @brief Quantum stage within a pipeline
 */
typedef struct quantum_stage_t {
    PipelineFunction function;  /**< Stage function */
    void* args;                 /**< Function arguments */
    void* input_buffer;         /**< Input buffer */
    void* output_buffer;        /**< Output buffer */
    size_t buffer_size;         /**< Buffer size */
} quantum_stage_t;

/**
 * @brief Quantum batch for parallel processing
 */
typedef struct quantum_batch_t {
    void* data;                 /**< Batch data */
    size_t batch_offset;        /**< Offset in global data */
    size_t batch_size;          /**< Current batch size */
    void* processed;            /**< Processed results */
} quantum_batch_t;

/**
 * @brief Quantum workspace for temporary computations
 */
typedef struct QuantumWorkspace {
    void* scratch;              /**< Scratch memory */
    size_t scratch_size;        /**< Scratch size */
    void* auxiliary;            /**< Auxiliary buffer */
    size_t auxiliary_size;      /**< Auxiliary size */
} QuantumWorkspace;

// ============================================================================
// Pipeline Configuration Structures
// ============================================================================

/**
 * @brief Configuration for quantum pipeline operations
 */
typedef struct quantum_pipeline_config_t {
    double precision;                      /**< Target precision */
    double success_probability;            /**< Target success probability */
    bool use_quantum_memory;               /**< Use quantum memory */
    QuantumErrorCorrection error_correction; /**< Error correction method */
    QuantumOptLevel optimization_level;    /**< Optimization level */
    QuantumPipelineType pipeline_type;     /**< Pipeline type */
} quantum_pipeline_config_t;

/**
 * @brief Configuration for quantum forward pass
 */
typedef struct quantum_forward_config_t {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    QuantumErrorCorrection error_correction;
    QuantumOptLevel optimization_level;
    QuantumForwardType forward_type;
} quantum_forward_config_t;

/**
 * @brief Configuration for quantum backward pass
 */
typedef struct quantum_backward_config_t {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    QuantumErrorCorrection error_correction;
    QuantumOptLevel optimization_level;
    QuantumBackwardType backward_type;
} quantum_backward_config_t;

/**
 * @brief Configuration for quantum gradient computation
 */
typedef struct quantum_gradient_config_t {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    QuantumErrorCorrection error_correction;
    QuantumOptLevel optimization_level;
    QuantumGradientType gradient_type;
} quantum_gradient_config_t;

// ============================================================================
// Utility Macros
// ============================================================================

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// ============================================================================
// Pipeline Management Functions
// ============================================================================

/**
 * @brief Initialize a pipeline context
 *
 * @param batch_size Maximum batch size
 * @return Initialized pipeline context or NULL on failure
 */
PipelineContext* init_pipeline(size_t batch_size);

/**
 * @brief Add a stage to the pipeline
 *
 * @param ctx Pipeline context
 * @param function Stage processing function
 * @param args Arguments for the function
 * @param buffer_size Size of stage buffers
 * @return 0 on success, negative on error
 */
int add_pipeline_stage(PipelineContext* ctx,
                       PipelineFunction function,
                       void* args,
                       size_t buffer_size);

/**
 * @brief Execute the pipeline on input data
 *
 * @param ctx Pipeline context
 * @param input_data Input data buffer
 * @param output_data Output data buffer
 * @param data_size Size of data in bytes
 * @return 0 on success, negative on error
 */
int execute_pipeline(PipelineContext* ctx,
                     const void* input_data,
                     void* output_data,
                     size_t data_size);

/**
 * @brief Clean up pipeline resources
 *
 * @param ctx Pipeline context to clean up
 */
void cleanup_pipeline(PipelineContext* ctx);

// ============================================================================
// Stage Factory Functions
// ============================================================================

/**
 * @brief Create a forward pass pipeline stage
 *
 * @param config Layer configuration
 * @return Pipeline function for forward pass
 */
PipelineFunction create_forward_stage(const LayerConfig* config);

/**
 * @brief Create a backward pass pipeline stage
 *
 * @param config Layer configuration
 * @return Pipeline function for backward pass
 */
PipelineFunction create_backward_stage(const LayerConfig* config);

/**
 * @brief Create a gradient computation pipeline stage
 *
 * @param config Layer configuration
 * @return Pipeline function for gradient computation
 */
PipelineFunction create_gradient_stage(const LayerConfig* config);

// ============================================================================
// Quantum System Functions
// ============================================================================

/**
 * @brief Create a quantum system
 *
 * @param num_qubits Number of qubits
 * @param flags Operation flags
 * @return Quantum system or NULL on failure
 */
static inline quantum_system_t* quantum_system_create(size_t num_qubits, int flags) {
    quantum_system_t* system = (quantum_system_t*)calloc(1, sizeof(quantum_system_t));
    if (!system) return NULL;
    system->num_qubits = num_qubits;
    system->state_size = (size_t)1 << num_qubits;
    system->flags = flags;
    return system;
}

/**
 * @brief Destroy a quantum system
 *
 * @param system System to destroy
 */
static inline void quantum_system_destroy(quantum_system_t* system) {
    if (system) {
        free(system->state_vector);
        free(system);
    }
}

// ============================================================================
// Quantum Circuit Functions
// ============================================================================

static inline quantum_circuit_t* quantum_create_pipeline_circuit(size_t num_qubits, size_t num_stages, QuantumCircuitType type) {
    quantum_circuit_t* circuit = (quantum_circuit_t*)calloc(1, sizeof(quantum_circuit_t));
    if (!circuit) return NULL;
    circuit->num_qubits = num_qubits;
    circuit->num_gates = num_stages;
    circuit->type = type;
    return circuit;
}

static inline quantum_circuit_t* quantum_create_forward_circuit(size_t num_qubits, QuantumCircuitType type) {
    return quantum_create_pipeline_circuit(num_qubits, 1, type);
}

static inline quantum_circuit_t* quantum_create_backward_circuit(size_t num_qubits, QuantumCircuitType type) {
    return quantum_create_pipeline_circuit(num_qubits, 1, type);
}

static inline quantum_circuit_t* quantum_create_gradient_circuit(size_t num_qubits, QuantumCircuitType type) {
    return quantum_create_pipeline_circuit(num_qubits, 1, type);
}

static inline void quantum_circuit_destroy(quantum_circuit_t* circuit) {
    if (circuit) {
        free(circuit->gates);
        free(circuit);
    }
}

// ============================================================================
// Quantum Register Functions
// ============================================================================

static inline quantum_register_t* quantum_register_create_state(const void* data, size_t size, quantum_system_t* system) {
    (void)system;
    quantum_register_t* reg = (quantum_register_t*)calloc(1, sizeof(quantum_register_t));
    if (!reg) return NULL;
    reg->data = malloc(size);
    if (!reg->data) {
        free(reg);
        return NULL;
    }
    memcpy(reg->data, data, size);
    reg->size = size;
    reg->allocated = true;
    return reg;
}

static inline quantum_register_t* quantum_register_create_empty(size_t size) {
    quantum_register_t* reg = (quantum_register_t*)calloc(1, sizeof(quantum_register_t));
    if (!reg) return NULL;
    reg->data = calloc(1, size);
    if (!reg->data) {
        free(reg);
        return NULL;
    }
    reg->size = size;
    reg->allocated = true;
    return reg;
}

static inline void quantum_register_destroy(quantum_register_t* reg) {
    if (reg) {
        if (reg->allocated) free(reg->data);
        free(reg);
    }
}

// ============================================================================
// Quantum Pipeline Functions
// ============================================================================

static inline quantum_pipeline_t* quantum_pipeline_create(quantum_system_t* system, size_t num_stages, QuantumPipelineType type) {
    (void)system;
    quantum_pipeline_t* pipeline = (quantum_pipeline_t*)calloc(1, sizeof(quantum_pipeline_t));
    if (!pipeline) return NULL;
    pipeline->num_stages = num_stages;
    pipeline->type = type;
    return pipeline;
}

static inline void quantum_pipeline_destroy(quantum_pipeline_t* pipeline) {
    if (pipeline) {
        free(pipeline->stages);
        free(pipeline->buffers);
        free(pipeline);
    }
}

// ============================================================================
// Quantum Stage Functions
// ============================================================================

static inline quantum_stage_t* quantum_stage_create(PipelineFunction func, void* args,
                                                     quantum_system_t* system,
                                                     quantum_circuit_t* circuit,
                                                     const void* config) {
    (void)system; (void)circuit; (void)config;
    quantum_stage_t* stage = (quantum_stage_t*)calloc(1, sizeof(quantum_stage_t));
    if (!stage) return NULL;
    stage->function = func;
    stage->args = args;
    return stage;
}

static inline void quantum_stage_destroy(quantum_stage_t* stage) {
    if (stage) {
        free(stage->input_buffer);
        free(stage->output_buffer);
        free(stage);
    }
}

// ============================================================================
// Quantum Batch Functions
// ============================================================================

static inline quantum_batch_t* quantum_batch_create(quantum_register_t* reg,
                                                     size_t offset, size_t batch_size,
                                                     quantum_system_t* system,
                                                     quantum_circuit_t* circuit,
                                                     const void* config) {
    (void)system; (void)circuit; (void)config;
    quantum_batch_t* batch = (quantum_batch_t*)calloc(1, sizeof(quantum_batch_t));
    if (!batch) return NULL;
    batch->batch_offset = offset;
    batch->batch_size = batch_size;
    if (reg && reg->data && offset < reg->size) {
        batch->data = (char*)reg->data + offset;
    }
    return batch;
}

static inline void quantum_batch_destroy(quantum_batch_t* batch) {
    free(batch);
}

// ============================================================================
// Quantum Workspace Functions
// ============================================================================

static inline QuantumWorkspace* init_quantum_workspace(size_t chunk_size) {
    QuantumWorkspace* ws = (QuantumWorkspace*)calloc(1, sizeof(QuantumWorkspace));
    if (!ws) return NULL;
    ws->scratch_size = chunk_size;
    ws->scratch = malloc(chunk_size);
    if (!ws->scratch) {
        free(ws);
        return NULL;
    }
    return ws;
}

static inline void cleanup_quantum_workspace(QuantumWorkspace* ws) {
    if (ws) {
        free(ws->scratch);
        free(ws->auxiliary);
        free(ws);
    }
}

// ============================================================================
// Quantum Execution Stubs
// ============================================================================

static inline void quantum_execute_stage(quantum_pipeline_t* pipeline,
                                          quantum_stage_t* stage,
                                          quantum_batch_t* batch,
                                          size_t stage_idx,
                                          quantum_system_t* system,
                                          quantum_circuit_t* circuit,
                                          const void* config,
                                          QuantumWorkspace* ws) {
    (void)pipeline; (void)stage_idx; (void)system; (void)circuit; (void)config; (void)ws;
    if (stage && stage->function && batch && batch->data) {
        stage->function(batch->data, batch->data, batch->batch_size, stage->args);
    }
}

static inline void quantum_extract_batch(quantum_register_t* output,
                                          size_t offset,
                                          quantum_batch_t* batch,
                                          size_t size,
                                          quantum_system_t* system,
                                          quantum_circuit_t* circuit,
                                          const void* config,
                                          QuantumWorkspace* ws) {
    (void)system; (void)circuit; (void)config; (void)ws;
    if (output && output->data && batch && batch->data && offset + size <= output->size) {
        memcpy((char*)output->data + offset, batch->data, size);
    }
}

static inline void quantum_extract_results(void* output,
                                            quantum_register_t* reg,
                                            size_t size,
                                            quantum_system_t* system,
                                            quantum_circuit_t* circuit,
                                            const void* config) {
    (void)system; (void)circuit; (void)config;
    if (output && reg && reg->data) {
        size_t copy_size = (size < reg->size) ? size : reg->size;
        memcpy(output, reg->data, copy_size);
    }
}

// ============================================================================
// Quantum Pass Functions (Stubs)
// ============================================================================

static inline void quantum_forward_pass(quantum_register_t* input,
                                         quantum_register_t* output,
                                         size_t count,
                                         const LayerConfig* config,
                                         quantum_system_t* system,
                                         quantum_circuit_t* circuit,
                                         const void* fwd_config) {
    (void)system; (void)circuit; (void)fwd_config;
    if (input && output && input->data && output->data && config) {
        // Simple passthrough for stub - real implementation would use quantum operations
        size_t copy_size = min(input->size, output->size);
        memcpy(output->data, input->data, copy_size);
    }
}

static inline void quantum_backward_pass(quantum_register_t* input,
                                          quantum_register_t* output,
                                          size_t count,
                                          const LayerConfig* config,
                                          quantum_system_t* system,
                                          quantum_circuit_t* circuit,
                                          const void* bwd_config) {
    (void)system; (void)circuit; (void)bwd_config;
    if (input && output && input->data && output->data && config) {
        size_t copy_size = min(input->size, output->size);
        memcpy(output->data, input->data, copy_size);
    }
}

static inline void quantum_compute_gradients(quantum_register_t* input,
                                              quantum_register_t* output,
                                              size_t count,
                                              const LayerConfig* config,
                                              quantum_system_t* system,
                                              quantum_circuit_t* circuit,
                                              const void* grad_config) {
    (void)system; (void)circuit; (void)grad_config;
    if (input && output && input->data && output->data && config) {
        size_t copy_size = min(input->size, output->size);
        memcpy(output->data, input->data, copy_size);
    }
}

#ifdef __cplusplus
}
#endif

#endif // PIPELINE_PARALLEL_H
