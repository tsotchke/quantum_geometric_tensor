#include <metal_stdlib>
#include "metal_common.h"
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Integration parameters
struct IntegrationParams {
    uint steps;
    float step_size;
    float tolerance;
    bool adaptive_step;
};

// Integration results
struct IntegrationResults {
    float4x4 result;
    float error_estimate;
    uint actual_steps;
    bool converged;
};

// Helper functions for numerical stability
inline bool is_valid_float4x4(float4x4 m) {
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            if (isnan(m[i][j]) || isinf(m[i][j]) || abs(m[i][j]) > MAX_MAGNITUDE) {
                return false;
            }
        }
    }
    return true;
}

inline float4x4 normalize_matrix(float4x4 m, thread float& max_magnitude) {
    float4x4 result = m;
    max_magnitude = 0.0f;
    
    // First pass: find maximum magnitude
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            float mag = abs(m[i][j]);
            max_magnitude = max(max_magnitude, mag);
        }
    }
    
    // Second pass: normalize if needed
    if (max_magnitude > MAX_MAGNITUDE) {
        float scale = MAX_MAGNITUDE / max_magnitude;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                result[i][j] *= scale;
            }
        }
    }
    
    return result;
}

// Runge-Kutta 4th order integration kernel with enhanced stability
kernel void rk4_integrate(
    device const float4x4* initial [[buffer(0)]],
    device const float4x4* derivative [[buffer(1)]],
    device IntegrationResults* results [[buffer(2)]],
    constant IntegrationParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1) return; // Single integration per call
    
    // Load and validate initial conditions
    float4x4 y = *initial;
    if (!is_valid_float4x4(y) || !is_valid_float4x4(*derivative)) {
        results->result = float4x4(0.0f);
        results->error_estimate = INFINITY;
        results->actual_steps = 0;
        results->converged = false;
        return;
    }
    
    // Initialize integration variables with stability checks
    float h = clamp(params.step_size, MIN_MAGNITUDE, MAX_MAGNITUDE);
    uint steps = 0;
    float max_error = 0.0f;
    bool integration_valid = true;
    
    // Normalize initial conditions for stability
    float max_magnitude;
    y = normalize_matrix(y, max_magnitude);
    
    for (uint step = 0; step < params.steps && integration_valid; step++) {
        // RK4 coefficients with stability checks
        float4x4 k1 = *derivative;
        float4x4 k2, k3, k4;
        
        // Calculate intermediate points with validation
        for (uint i = 0; i < 4 && integration_valid; i++) {
            for (uint j = 0; j < 4; j++) {
                // K2 calculation
                float k2_val = k1[i][j] + h * 0.5f;
                if (isnan(k2_val) || isinf(k2_val) || abs(k2_val) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                k2[i][j] = k2_val;
                
                // K3 calculation
                float k3_val = k2[i][j] + h * 0.5f;
                if (isnan(k3_val) || isinf(k3_val) || abs(k3_val) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                k3[i][j] = k3_val;
                
                // K4 calculation
                float k4_val = k3[i][j] + h;
                if (isnan(k4_val) || isinf(k4_val) || abs(k4_val) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                k4[i][j] = k4_val;
            }
        }
        
        if (!integration_valid) break;
        
        // Update solution with stability checks
        float4x4 new_y;
        float step_error = 0.0f;
        
        for (uint i = 0; i < 4 && integration_valid; i++) {
            for (uint j = 0; j < 4; j++) {
                float update = h/6.0f * (
                    k1[i][j] + 2.0f*k2[i][j] + 
                    2.0f*k3[i][j] + k4[i][j]
                );
                
                if (isnan(update) || isinf(update) || abs(update) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                
                new_y[i][j] = y[i][j] + update;
                step_error = max(step_error, abs(update));
            }
        }
        
        if (!integration_valid) break;
        
        // Adaptive step size with stability checks
        if (params.adaptive_step) {
            if (step_error > params.tolerance) {
                h = max(h * 0.5f, MIN_MAGNITUDE);
                continue;
            } else if (step_error < ERROR_THRESHOLD) {
                // Early convergence achieved
                integration_valid = true;
                break;
            } else if (step_error < params.tolerance * 0.1f) {
                h = min(h * 2.0f, MAX_MAGNITUDE);
            }
        }
        
        // Update state and error tracking
        y = normalize_matrix(new_y, max_magnitude);
        max_error = max(max_error, step_error);
        steps++;
    }
    
    // Store results with stability checks
    if (!integration_valid) {
        results->result = float4x4(0.0f);
        results->error_estimate = INFINITY;
        results->actual_steps = steps;
        results->converged = false;
    } else {
        results->result = y;
        results->error_estimate = max_error;
        results->actual_steps = steps;
        results->converged = max_error <= params.tolerance;
    }
}

// Symplectic integration kernel for Hamiltonian systems with enhanced stability
kernel void symplectic_integrate(
    device const float4x4* position [[buffer(0)]],
    device const float4x4* momentum [[buffer(1)]],
    device const float4x4* hamiltonian [[buffer(2)]],
    device IntegrationResults* results [[buffer(3)]],
    constant IntegrationParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1) return; // Single integration per call
    
    // Load and validate initial conditions
    float4x4 q = *position;
    float4x4 p = *momentum;
    
    if (!is_valid_float4x4(q) || !is_valid_float4x4(p) || !is_valid_float4x4(*hamiltonian)) {
        results->result = float4x4(0.0f);
        results->error_estimate = INFINITY;
        results->actual_steps = 0;
        results->converged = false;
        return;
    }
    
    // Initialize integration variables with stability checks
    float h = clamp(params.step_size, MIN_MAGNITUDE, MAX_MAGNITUDE);
    uint steps = 0;
    float max_error = 0.0f;
    bool integration_valid = true;
    
    // Normalize initial conditions for stability
    float max_magnitude;
    q = normalize_matrix(q, max_magnitude);
    p = normalize_matrix(p, max_magnitude);
    
    for (uint step = 0; step < params.steps && integration_valid; step++) {
        // Calculate Hamiltonian derivatives with stability checks
        float4x4 dH_dq, dH_dp;
        
        for (uint i = 0; i < 4 && integration_valid; i++) {
            for (uint j = 0; j < 4; j++) {
                float dq = (*hamiltonian)[i][j];
                float dp = (*hamiltonian)[j][i];
                
                if (isnan(dq) || isinf(dq) || abs(dq) > MAX_MAGNITUDE ||
                    isnan(dp) || isinf(dp) || abs(dp) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                
                dH_dq[i][j] = dq;
                dH_dp[i][j] = dp;
            }
        }
        
        if (!integration_valid) break;
        
        // Update momentum and position with stability checks
        float4x4 new_p, new_q;
        float step_error = 0.0f;
        
        for (uint i = 0; i < 4 && integration_valid; i++) {
            for (uint j = 0; j < 4; j++) {
                float dp = h * dH_dq[i][j];
                float dq = h * dH_dp[i][j];
                
                if (isnan(dp) || isinf(dp) || abs(dp) > MAX_MAGNITUDE ||
                    isnan(dq) || isinf(dq) || abs(dq) > MAX_MAGNITUDE) {
                    integration_valid = false;
                    break;
                }
                
                new_p[i][j] = p[i][j] - dp;
                new_q[i][j] = q[i][j] + dq;
                
                step_error = max(step_error, max(abs(dp), abs(dq)));
            }
        }
        
        if (!integration_valid) break;
        
        // Adaptive step size with stability checks
        if (params.adaptive_step) {
            if (step_error > params.tolerance) {
                h = max(h * 0.5f, MIN_MAGNITUDE);
                continue;
            } else if (step_error < ERROR_THRESHOLD) {
                // Early convergence achieved
                integration_valid = true;
                break;
            } else if (step_error < params.tolerance * 0.1f) {
                h = min(h * 2.0f, MAX_MAGNITUDE);
            }
        }
        
        // Update state and error tracking
        p = normalize_matrix(new_p, max_magnitude);
        q = normalize_matrix(new_q, max_magnitude);
        max_error = max(max_error, step_error);
        steps++;
    }
    
    // Store results with stability checks
    if (!integration_valid) {
        results->result = float4x4(0.0f);
        results->error_estimate = INFINITY;
        results->actual_steps = steps;
        results->converged = false;
    } else {
        // Combine position and momentum with stability checks
        float4x4 combined;
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                combined[i][j] = i < 2 ? q[i][j] : p[i-2][j];
            }
        }
        
        results->result = combined;
        results->error_estimate = max_error;
        results->actual_steps = steps;
        results->converged = max_error <= params.tolerance;
    }
}
