#ifndef LEARNING_CONFIG_H
#define LEARNING_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/learning/learning_task.h"

// Default configuration values
#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_DROPOUT_RATE 0.1
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_NUM_EPOCHS 100
#define DEFAULT_VAL_SPLIT 0.2
#define DEFAULT_CHECKPOINT_FREQ 10
#define DEFAULT_PATIENCE 5

// Quantum-specific configuration
#define DEFAULT_NUM_QUBITS 8  // Reduced for stability
#define DEFAULT_NUM_LAYERS 2  // Reduced for stability
#define DEFAULT_NUM_CLUSTERS 8  // Reduced for stability
#define DEFAULT_LATENT_DIM 16  // Reduced for stability

// Default configuration functions
task_config_t default_task_config(task_type_t type);

#endif // LEARNING_CONFIG_H
