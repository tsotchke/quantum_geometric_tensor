.text
.align 2

// AMX instruction encodings for Apple Silicon
#define AMX_LDX     0xE5808000  // Load X registers
#define AMX_LDY     0xE5808400  // Load Y registers
#define AMX_STX     0xE5808800  // Store X registers
#define AMX_STY     0xE5808C00  // Store Y registers
#define AMX_LDZ     0xE5809000  // Load Z registers
#define AMX_STZ     0xE5809400  // Store Z registers
#define AMX_FMA64   0xE580A000  // Fused multiply-add
#define AMX_EXTRX   0xE580AC00  // Extract row from X
#define AMX_EXTRY   0xE580B000  // Extract row from Y
#define AMX_MAC64   0xE580B400  // Multiply-accumulate
#define AMX_INIT    0xE5800000  // Initialize AMX
#define AMX_STOP    0xE5800400  // Stop AMX

// AMX register state offsets (aligned to 64-byte cache lines)
#define AMX_X_OFFSET    0
#define AMX_Y_OFFSET    256  // 4 cache lines
#define AMX_Z_OFFSET    512  // 8 cache lines

// Register aliases for better readability
x_ptr       .req x0   // Source/destination pointer for X
y_ptr       .req x1   // Source/destination pointer for Y
z_ptr       .req x2   // Source/destination pointer for Z
offset      .req x3   // Offset for loads/stores
row_idx     .req x4   // Row index for extracts
col_idx     .req x5   // Column index for extracts
temp        .req x6   // Temporary register

.global _amx_init_asm
_amx_init_asm:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Enable AMX coprocessor with full capabilities
    MSR     S3_5_C15_C12_0, XZR    // Initialize AMX
    ISB                             // Instruction barrier
    
    // Verify AMX is enabled
    MRS     temp, S3_5_C15_C12_0
    CMP     temp, XZR
    B.NE    1f
    
    // AMX init failed
    MOV     x0, #1
    B       2f
    
1:  // AMX init succeeded
    MOV     x0, XZR
    
2:  // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_stop
amx_stop:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Disable AMX coprocessor
    MSR     S3_5_C15_C12_1, XZR
    ISB
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_ldx
amx_ldx:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, x_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     x_ptr, x_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Load X registers with prefetch
    PRFM    PLDL1KEEP, [x_ptr, #64]   // Prefetch next cache line
    .inst   AMX_LDX | (0 << 16)       // Load into X[0]
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_ldy
amx_ldy:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, y_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     y_ptr, y_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Load Y registers with prefetch
    PRFM    PLDL1KEEP, [y_ptr, #64]   // Prefetch next cache line
    .inst   AMX_LDY | (0 << 16)       // Load into Y[0]
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_stx
amx_stx:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, x_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     x_ptr, x_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Store X registers
    .inst   AMX_STX | (0 << 16)       // Store from X[0]
    DC      CVAC, [x_ptr]             // Clean cache line
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_sty
amx_sty:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, y_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     y_ptr, y_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Store Y registers
    .inst   AMX_STY | (0 << 16)       // Store from Y[0]
    DC      CVAC, [y_ptr]             // Clean cache line
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_ldz
amx_ldz:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, z_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     z_ptr, z_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Load Z registers with prefetch
    PRFM    PLDL1KEEP, [z_ptr, #64]   // Prefetch next cache line
    .inst   AMX_LDZ | (0 << 16)       // Load into Z[0]
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_stz
amx_stz:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Ensure pointer is 64-byte aligned
    AND     temp, z_ptr, #0x3F
    CBZ     temp, 1f
    
    // Align pointer down and adjust offset
    BIC     z_ptr, z_ptr, #0x3F
    ADD     offset, offset, temp
    
1:  // Store Z registers
    .inst   AMX_STZ | (0 << 16)       // Store from Z[0]
    DC      CVAC, [z_ptr]             // Clean cache line
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET

.global amx_fma64
amx_fma64:
    // Save registers
    STP     x29, x30, [sp, #-16]!
    MOV     x29, sp
    
    // Perform FMA operation: Z += X * Y
    .inst   AMX_FMA64 | (0 << 16)     // Use X[0], Y[0], Z[0]
    
    // Optional: Extract specific rows for better precision
    .inst   AMX_EXTRX | (0 << 16)     // Extract from X[0]
    .inst   AMX_EXTRY | (0 << 16)     // Extract from Y[0]
    .inst   AMX_MAC64 | (0 << 16)     // Additional multiply-accumulate
    
    // Restore registers and return
    LDP     x29, x30, [sp], #16
    RET
