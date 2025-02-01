#include "quantum_geometric/hardware/quantum_geometric_tensor_config.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdlib.h>

// Global configuration instance
static QGTensorConfig global_config = {
    .acceleration_type = QGT_DEFAULT_ACCELERATION,
    .batch_size = 32,
    .use_gpu = 1,
    .learning_rate = 0.001f
};

QGTensorConfig qgt_get_default_config(void) {
    QGTensorConfig config = {
        .acceleration_type = QGT_DEFAULT_ACCELERATION,
        .batch_size = 32,
        .use_gpu = 1,
        .learning_rate = 0.001f
    };
    return config;
}

int qgt_set_config(const QGTensorConfig* config) {
    if (!config) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Validate configuration
    if (config->batch_size <= 0 || config->learning_rate <= 0.0f) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    // Copy configuration
    global_config = *config;
    
    return QGT_SUCCESS;
}

const QGTensorConfig* qgt_get_config(void) {
    return &global_config;
}
