#include "../include/quantum_geometric_core.h"
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <string.h>

/**
 * @file quantum_distributed_execution.c
 * @brief Implementation of distributed quantum execution across multiple backends
 */

/* Distribution parameters */
#define MAX_BACKENDS 32
#define MAX_CIRCUITS_PER_BACKEND 128
#define MIN_CIRCUIT_SIZE 16
#define MAX_RETRIES 3

/* Backend status */
typedef struct {
    char* name;
    size_t max_qubits;
    size_t available_qubits;
    double uptime;
    double error_rate;
    bool is_available;
} backend_status_t;

/* Circuit partition */
typedef struct {
    size_t start_qubit;
    size_t num_qubits;
    char* qasm_circuit;
    double* results;
    size_t result_size;
    bool completed;
} circuit_partition_t;

/* Distribution context */
typedef struct {
    backend_status_t backends[MAX_BACKENDS];
    size_t num_backends;
    circuit_partition_t* partitions;
    size_t num_partitions;
    MPI_Comm comm;
    int rank;
    int size;
} distribution_context_t;

/* Initialize distribution context */
static qgt_error_t init_distribution(
    distribution_context_t* context
) {
    if (!context) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Initialize MPI if not already initialized */
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            return QGT_ERROR_NOT_INITIALIZED;
        }
    }
    
    /* Get MPI info */
    MPI_Comm_dup(MPI_COMM_WORLD, &context->comm);
    MPI_Comm_rank(context->comm, &context->rank);
    MPI_Comm_size(context->comm, &context->size);
    
    /* Query available backends */
    if (context->rank == 0) {
        /* Get IBM Quantum backends */
        ibm_backend_config_t ibm_config = {0};
        qgt_error_t err = init_ibm_backend(&ibm_config);
        if (err != QGT_SUCCESS) {
            return err;
        }
        
        /* Store backend info */
        context->backends[context->num_backends++] = (backend_status_t){
            .name = strdup(ibm_config.backend_name),
            .max_qubits = ibm_config.max_qubits,
            .available_qubits = ibm_config.max_qubits,
            .uptime = 1.0,
            .error_rate = 0.001,
            .is_available = true
        };
        
        free(ibm_config.api_token);
        free(ibm_config.backend_name);
    }
    
    /* Broadcast backend info */
    MPI_Bcast(&context->num_backends, 1, MPI_SIZE_T, 0, context->comm);
    for (size_t i = 0; i < context->num_backends; i++) {
        backend_status_t* backend = &context->backends[i];
        
        if (context->rank == 0) {
            size_t name_len = strlen(backend->name);
            MPI_Bcast(&name_len, 1, MPI_SIZE_T, 0, context->comm);
            MPI_Bcast(backend->name, name_len, MPI_CHAR, 0, context->comm);
            MPI_Bcast(&backend->max_qubits, 1, MPI_SIZE_T, 0, context->comm);
            MPI_Bcast(&backend->available_qubits, 1, MPI_SIZE_T, 0, context->comm);
            MPI_Bcast(&backend->uptime, 1, MPI_DOUBLE, 0, context->comm);
            MPI_Bcast(&backend->error_rate, 1, MPI_DOUBLE, 0, context->comm);
            MPI_Bcast(&backend->is_available, 1, MPI_C_BOOL, 0, context->comm);
        } else {
            size_t name_len;
            MPI_Bcast(&name_len, 1, MPI_SIZE_T, 0, context->comm);
            backend->name = malloc(name_len + 1);
            MPI_Bcast(backend->name, name_len, MPI_CHAR, 0, context->comm);
            backend->name[name_len] = '\0';
            MPI_Bcast(&backend->max_qubits, 1, MPI_SIZE_T, 0, context->comm);
            MPI_Bcast(&backend->available_qubits, 1, MPI_SIZE_T, 0, context->comm);
            MPI_Bcast(&backend->uptime, 1, MPI_DOUBLE, 0, context->comm);
            MPI_Bcast(&backend->error_rate, 1, MPI_DOUBLE, 0, context->comm);
            MPI_Bcast(&backend->is_available, 1, MPI_C_BOOL, 0, context->comm);
        }
    }
    
    return QGT_SUCCESS;
}

/* Partition quantum circuit */
static qgt_error_t partition_circuit(
    const quantum_geometric_tensor* tensor,
    distribution_context_t* context,
    uint32_t flags
) {
    if (!tensor || !context) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Calculate total available qubits */
    size_t total_qubits = 0;
    for (size_t i = 0; i < context->num_backends; i++) {
        if (context->backends[i].is_available) {
            total_qubits += context->backends[i].available_qubits;
        }
    }
    
    /* Determine partition size */
    size_t partition_size = MIN_CIRCUIT_SIZE;
    while (partition_size * 2 <= total_qubits / context->num_backends &&
           partition_size * 2 <= tensor->num_spins) {
        partition_size *= 2;
    }
    
    /* Create partitions */
    size_t num_partitions = (tensor->num_spins + partition_size - 1) / partition_size;
    context->partitions = calloc(num_partitions, sizeof(circuit_partition_t));
    if (!context->partitions) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }
    context->num_partitions = num_partitions;
    
    /* Initialize partitions */
    for (size_t i = 0; i < num_partitions; i++) {
        circuit_partition_t* partition = &context->partitions[i];
        partition->start_qubit = i * partition_size;
        partition->num_qubits = (i == num_partitions - 1) ?
            tensor->num_spins - partition->start_qubit : partition_size;
        partition->completed = false;
        
        /* Convert partition to QASM */
        quantum_circuit_t circuit = {0};
        qgt_error_t err = tensor_to_circuit(tensor, &circuit, flags);
        if (err != QGT_SUCCESS) {
            free_quantum_circuit(&circuit);
            return err;
        }
        
        /* Extract relevant gates */
        quantum_circuit_t partition_circuit = {0};
        err = init_quantum_circuit(tensor, &partition_circuit);
        if (err != QGT_SUCCESS) {
            free_quantum_circuit(&circuit);
            return err;
        }
        
        for (size_t j = 0; j < circuit.num_layers; j++) {
            circuit_layer_t* layer = &circuit.layers[j];
            
            for (size_t k = 0; k < layer->num_gates; k++) {
                quantum_gate_t* gate = &layer->gates[k];
                bool in_partition = false;
                
                for (size_t l = 0; l < gate->num_qubits; l++) {
                    if (gate->qubits[l] >= partition->start_qubit &&
                        gate->qubits[l] < partition->start_qubit + partition->num_qubits) {
                        in_partition = true;
                        break;
                    }
                }
                
                if (in_partition) {
                    err = add_gate(&partition_circuit, gate);
                    if (err != QGT_SUCCESS) {
                        free_quantum_circuit(&circuit);
                        free_quantum_circuit(&partition_circuit);
                        return err;
                    }
                }
            }
        }
        
        /* Convert to QASM */
        err = circuit_to_qasm(&partition_circuit, &partition->qasm_circuit);
        
        free_quantum_circuit(&circuit);
        free_quantum_circuit(&partition_circuit);
        
        if (err != QGT_SUCCESS) {
            return err;
        }
    }
    
    return QGT_SUCCESS;
}

/* Execute partition on backend */
static qgt_error_t execute_partition(
    circuit_partition_t* partition,
    const backend_status_t* backend,
    uint32_t flags
) {
    if (!partition || !backend) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Execute on IBM Quantum */
    circuit_result_t result = {0};
    qgt_error_t err = execute_quantum_circuit(backend->name, partition->qasm_circuit, &result);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    /* Store results */
    partition->results = calloc(result.num_states, sizeof(double));
    if (!partition->results) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }
    memcpy(partition->results, result.probabilities,
           result.num_states * sizeof(double));
    partition->result_size = result.num_states;
    partition->completed = true;
    
    free(result.probabilities);
    free(result.counts);
    
    return QGT_SUCCESS;
}

/* Merge partition results */
static qgt_error_t merge_results(
    quantum_geometric_tensor* tensor,
    const distribution_context_t* context,
    uint32_t flags
) {
    if (!tensor || !context) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Gather all results to rank 0 */
    if (context->rank == 0) {
        for (size_t i = 1; i < context->size; i++) {
            for (size_t j = 0; j < context->num_partitions; j++) {
                circuit_partition_t* partition = &context->partitions[j];
                
                if (!partition->completed) {
                    MPI_Status status;
                    MPI_Recv(&partition->result_size, 1, MPI_SIZE_T,
                            i, 0, context->comm, &status);
                    
                    partition->results = calloc(partition->result_size, sizeof(double));
                    MPI_Recv(partition->results, partition->result_size, MPI_DOUBLE,
                            i, 1, context->comm, &status);
                    
                    partition->completed = true;
                }
            }
        }
        
        /* Update tensor with merged results */
        for (size_t i = 0; i < context->num_partitions; i++) {
            circuit_partition_t* partition = &context->partitions[i];
            
            for (size_t j = 0; j < partition->num_qubits; j++) {
                size_t global_idx = partition->start_qubit + j;
                double prob_up = 0.0;
                
                /* Sum probabilities for |1⟩ states */
                for (size_t k = 0; k < partition->result_size; k++) {
                    if (k & (1ULL << j)) {
                        prob_up += partition->results[k];
                    }
                }
                
                /* Update quantum state */
                double theta = 2 * acos(sqrt(1 - prob_up));
                tensor->spin_system.spin_states[global_idx] =
                    cos(theta/2) + I * sin(theta/2);
            }
        }
    } else {
        /* Send results to rank 0 */
        for (size_t i = 0; i < context->num_partitions; i++) {
            circuit_partition_t* partition = &context->partitions[i];
            
            if (partition->completed) {
                MPI_Send(&partition->result_size, 1, MPI_SIZE_T,
                        0, 0, context->comm);
                MPI_Send(partition->results, partition->result_size, MPI_DOUBLE,
                        0, 1, context->comm);
            }
        }
    }
    
    return QGT_SUCCESS;
}

/* Free distribution context */
static void free_distribution(distribution_context_t* context) {
    if (!context) return;
    
    for (size_t i = 0; i < context->num_backends; i++) {
        free(context->backends[i].name);
    }
    
    if (context->partitions) {
        for (size_t i = 0; i < context->num_partitions; i++) {
            free(context->partitions[i].qasm_circuit);
            free(context->partitions[i].results);
        }
        free(context->partitions);
    }
    
    MPI_Comm_free(&context->comm);
}

/* Public interface */
QGT_PUBLIC qgt_error_t execute_distributed(
    quantum_geometric_tensor* tensor,
    uint32_t flags
) {
    if (!tensor) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Initialize distribution */
    distribution_context_t context = {0};
    qgt_error_t err = init_distribution(&context);
    if (err != QGT_SUCCESS) {
        free_distribution(&context);
        return err;
    }
    
    /* Partition circuit */
    err = partition_circuit(tensor, &context, flags);
    if (err != QGT_SUCCESS) {
        free_distribution(&context);
        return err;
    }
    
    /* Execute partitions */
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < context.num_partitions; i++) {
        if (i % context.size == context.rank) {
            circuit_partition_t* partition = &context.partitions[i];
            
            /* Find available backend */
            for (size_t j = 0; j < context.num_backends; j++) {
                backend_status_t* backend = &context.backends[j];
                
                if (backend->is_available &&
                    backend->available_qubits >= partition->num_qubits) {
                    
                    /* Try to execute */
                    for (int retry = 0; retry < MAX_RETRIES; retry++) {
                        err = execute_partition(partition, backend, flags);
                        if (err == QGT_SUCCESS) break;
                    }
                    
                    if (err == QGT_SUCCESS) break;
                }
            }
        }
    }
    
    /* Merge results */
    err = merge_results(tensor, &context, flags);
    free_distribution(&context);
    
    return err;
}
