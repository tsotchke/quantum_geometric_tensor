#include <metal_stdlib>
using namespace metal;

// Constants for numerical stability
constant float MAX_MAGNITUDE = 1e3f;
constant float MIN_MAGNITUDE = 1e-6f;
constant float ERROR_THRESHOLD = 1e-6f;

// Complex number operations with stability
struct Complex {
    float2 val;
    
    Complex() : val(float2(0.0f)) {}
    Complex(float2 v) : val(v) {}
    Complex(float r, float i) : val(float2(r, i)) {}
    
    // Validation
    bool is_valid() const {
        float2 abs_val = abs(val);
        return !any(isnan(val)) && !any(isinf(val)) && 
               all((abs_val == float2(0.0f)) || 
                   (abs_val >= float2(MIN_MAGNITUDE) && abs_val <= float2(MAX_MAGNITUDE)));
    }
    
    // Magnitude control
    float magnitude() const {
        return length(val);
    }
    
    void normalize(float max_allowed = MAX_MAGNITUDE) {
        float mag = magnitude();
        if (mag > max_allowed) {
            val *= (max_allowed / mag);
        } else if (mag > 0.0f && mag < MIN_MAGNITUDE) {
            val *= (MIN_MAGNITUDE / mag);
        }
    }
    
    // Complex multiplication with stability
    friend Complex multiply(thread const Complex& a, thread const Complex& b) {
        if (!a.is_valid() || !b.is_valid()) {
            return Complex();
        }
        
        // Scale inputs if needed
        Complex a_norm = a;
        Complex b_norm = b;
        a_norm.normalize();
        b_norm.normalize();
        
        Complex result(float2(
            a_norm.val.x * b_norm.val.x - a_norm.val.y * b_norm.val.y,
            a_norm.val.x * b_norm.val.y + a_norm.val.y * b_norm.val.x
        ));
        
        float mag = result.magnitude();
        if (mag > 0.0f && mag < MIN_MAGNITUDE) {
            result.val *= (MIN_MAGNITUDE / mag);
        } else if (mag > MAX_MAGNITUDE) {
            result.val *= (MAX_MAGNITUDE / mag);
        }
        
        return result;
    }
    
    friend Complex multiply(device const Complex* a, thread const Complex& b) {
        Complex a_thread((*a).val);
        return multiply(a_thread, b);
    }
    
    Complex conj() const {
        return Complex(float2(val.x, -val.y));
    }
    
    // Addition with stability
    Complex operator+(thread const Complex& rhs) const {
        if (!rhs.is_valid()) return *this;
        Complex result = *this;
        result.val += rhs.val;
        
        float mag = result.magnitude();
        if (mag > 0.0f && mag < MIN_MAGNITUDE) {
            result.val *= (MIN_MAGNITUDE / mag);
        } else if (mag > MAX_MAGNITUDE) {
            result.val *= (MAX_MAGNITUDE / mag);
        }
        
        return result;
    }
    
    void add_to(thread Complex& result) const {
        if (!this->is_valid()) return;
        result.val += this->val;
        
        float mag = result.magnitude();
        if (mag > 0.0f && mag < MIN_MAGNITUDE) {
            result.val *= (MIN_MAGNITUDE / mag);
        } else if (mag > MAX_MAGNITUDE) {
            result.val *= (MAX_MAGNITUDE / mag);
        }
    }
};

// Helper functions
inline bool validate_field_params(
    uint idx,
    uint field_size,
    uint num_components,
    uint qubit = 0
) {
    return idx < field_size && 
           qubit < 32 && // Maximum supported qubits
           num_components > 0 && 
           num_components <= 256; // Reasonable upper limit
}

// Rotation kernel with enhanced stability
kernel void apply_rotation_kernel(
    device Complex* field [[buffer(0)]],
    device const Complex* rotation [[buffer(1)]],
    constant uint& field_size [[buffer(2)]],
    constant uint& num_components [[buffer(3)]],
    constant uint& qubit [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    // Validate parameters
    if (!validate_field_params(idx, field_size, num_components, qubit)) return;
    
    // Validate rotation matrix
    bool valid_rotation = true;
    for (uint i = 0; i < 4; i++) {
        if (!Complex(rotation[i].val).is_valid()) {
            valid_rotation = false;
            break;
        }
    }
    if (!valid_rotation) return;
    
    uint mask = 1u << qubit;
    if (idx & mask) {
        // Get field components with validation
        Complex psi_0(field[idx ^ mask].val);
        Complex psi_1(field[idx].val);
        
        if (!psi_0.is_valid() || !psi_1.is_valid()) {
            field[idx ^ mask].val = float2(0.0f);
            field[idx].val = float2(0.0f);
            return;
        }
        
        // Apply rotation with stability checks
        Complex new_psi_0 = multiply(&rotation[0], psi_0);
        Complex temp = multiply(&rotation[1], psi_1);
        new_psi_0 = new_psi_0 + temp;
        
        Complex new_psi_1 = multiply(&rotation[2], psi_0);
        temp = multiply(&rotation[3], psi_1);
        new_psi_1 = new_psi_1 + temp;
        
        // Normalize results
        new_psi_0.normalize();
        new_psi_1.normalize();
        
        // Update field
        field[idx ^ mask].val = new_psi_0.val;
        field[idx].val = new_psi_1.val;
    }
}

// Energy calculation kernel with enhanced stability
kernel void calculate_field_energy_kernel(
    device const Complex* field [[buffer(0)]],
    device const Complex* momentum [[buffer(1)]],
    device atomic_float* energy [[buffer(2)]],
    constant uint& field_size [[buffer(3)]],
    constant uint& num_components [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    // Validate parameters
    if (!validate_field_params(idx, field_size, num_components)) return;
    
    float local_energy = 0.0f;
    float max_term = 0.0f;
    
    // First pass: find maximum terms
    for (uint i = 0; i < num_components; i++) {
        Complex pi(momentum[idx * num_components + i].val);
        Complex phi(field[idx * num_components + i].val);
        
        if (!pi.is_valid() || !phi.is_valid()) continue;
        
        float pi_term = pi.magnitude();
        float phi_term = phi.magnitude();
        
        max_term = max(max_term, max(pi_term, phi_term));
    }
    
    // Second pass: accumulate with scaling if needed
    if (max_term > 0.0f && max_term < MAX_MAGNITUDE) {
        float scale = min(1.0f, MAX_MAGNITUDE / max_term);
        
        // Kinetic energy with scaling
        for (uint i = 0; i < num_components; i++) {
            Complex pi(momentum[idx * num_components + i].val);
            if (!pi.is_valid()) continue;
            
            float2 scaled_pi = pi.val * scale;
            local_energy += dot(scaled_pi, scaled_pi);
        }
        
        // Potential energy with scaling
        float phi_squared = 0.0f;
        for (uint i = 0; i < num_components; i++) {
            Complex phi(field[idx * num_components + i].val);
            if (!phi.is_valid()) continue;
            
            float2 scaled_phi = phi.val * scale;
            phi_squared += dot(scaled_phi, scaled_phi);
        }
        
        local_energy += 0.5f * phi_squared;
        
        // Rescale final result
        local_energy /= (scale * scale);
    }
    
    // Only add valid energy contributions
    if (local_energy > MIN_MAGNITUDE && local_energy < MAX_MAGNITUDE) {
        atomic_fetch_add_explicit(energy, local_energy, memory_order_relaxed);
    }
}

// Field equations kernel with enhanced stability
kernel void calculate_field_equations_kernel(
    device const Complex* field [[buffer(0)]],
    device Complex* equations [[buffer(1)]],
    constant uint& field_size [[buffer(2)]],
    constant uint& num_components [[buffer(3)]],
    constant float& mass [[buffer(4)]],
    constant float& coupling [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    // Validate parameters with error threshold
    if (!validate_field_params(idx, field_size, num_components)) return;
    
    if (mass < ERROR_THRESHOLD || !isfinite(mass) || 
        coupling < ERROR_THRESHOLD || !isfinite(coupling) ||
        abs(mass) > MAX_MAGNITUDE || abs(coupling) > MAX_MAGNITUDE) {
        return;
    }
    
    // First pass: calculate field magnitude with stability
    float phi_squared = 0.0f;
    float max_magnitude = 0.0f;
    
    for (uint i = 0; i < num_components; i++) {
        Complex phi(field[idx * num_components + i].val);
        if (!phi.is_valid()) continue;
        
        float mag = phi.magnitude();
        max_magnitude = max(max_magnitude, mag);
        phi_squared += dot(phi.val, phi.val);
    }
    
    // Apply scaling if needed
    float scale = 1.0f;
    if (max_magnitude > MAX_MAGNITUDE) {
        scale = MAX_MAGNITUDE / max_magnitude;
        phi_squared *= (scale * scale);
    }
    
    // Second pass: calculate equations with scaling
    for (uint i = 0; i < num_components; i++) {
        Complex phi(field[idx * num_components + i].val);
        if (!phi.is_valid()) {
            equations[idx * num_components + i].val = float2(0.0f);
            continue;
        }
        
        // Scale field component
        float2 scaled_phi = phi.val * scale;
        
        // Mass term with stability
        Complex eq(mass * mass * scaled_phi);
        float eq_mag = eq.magnitude();
        if (eq_mag > 0.0f && eq_mag < MIN_MAGNITUDE) {
            eq.val *= (MIN_MAGNITUDE / eq_mag);
        }
        
        // Interaction term with stability
        if (phi_squared > MIN_MAGNITUDE && phi_squared < MAX_MAGNITUDE) {
            Complex interaction(coupling * phi_squared * scaled_phi);
            float int_mag = interaction.magnitude();
            if (int_mag > 0.0f && int_mag < MIN_MAGNITUDE) {
                interaction.val *= (MIN_MAGNITUDE / int_mag);
            }
            eq = eq + interaction;
        }
        
        // Rescale result
        eq.val /= scale;
        
        // Final stability check
        float final_mag = eq.magnitude();
        if (final_mag > 0.0f && final_mag < MIN_MAGNITUDE) {
            eq.val *= (MIN_MAGNITUDE / final_mag);
        } else if (final_mag > MAX_MAGNITUDE) {
            eq.val *= (MAX_MAGNITUDE / final_mag);
        }
        
        equations[idx * num_components + i].val = eq.val;
    }
}
