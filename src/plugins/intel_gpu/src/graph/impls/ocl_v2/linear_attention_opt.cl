// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Optimized LinearAttention (GatedDeltaNet) OCL kernel.
// Target: K_HEAD_DIMS=128, SUBGROUP_SIZE=16 (Xe2/Xe3)
//
// OPT_ALGO modes:
//   0: Simple — native_exp/rsqrt, no redundant broadcast, unrolled iv loop
//   1: Algebraic fold — decay folded into FMA, two independent dots per iv,
//      pre-computed kq_dot shared across all iv iterations
//   2: Algebraic fold + K/Q double-buffering (prefetch next token's K/Q)
//
// Same dispatch model as ref: global={B, V_HEADS*v_blocks, SG_SIZE}, local={1,1,SG_SIZE}
//
// JIT Constants: K_HEAD_DIMS, V_HEAD_NUMS, Q_HEAD_NUMS, SUBGROUP_SIZE,
//                SCALE_FACTOR, IO_TYPE, OUTPUT_STATE, OPT_ALGO

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#ifndef TO_INPUT5_TYPE8
#    define TO_INPUT5_TYPE8(x) CAT(convert_, MAKE_VECTOR_TYPE(INPUT5_TYPE, 8))(x)
#endif

#define V_BLOCK_SIZE 4

inline float sum8_o(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_opt)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* g_input,
 __global INPUT4_TYPE* beta_input,
 __global INPUT5_TYPE* initial_state,
 __global OUTPUT_TYPE* output,
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,
#endif
 int seq_len,
 int key_offset,
 int value_offset) {

    const int b = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int STEP_STRIDE = Q_HEAD_NUMS * K_HEAD_DIMS;
    const int OUTPUT_STEP_STRIDE = V_HEAD_NUMS * K_HEAD_DIMS;
    const int KEY_STEP_STRIDE = (Q_HEAD_NUMS + key_offset) * K_HEAD_DIMS;
    const int VALUE_STEP_STRIDE = (V_HEAD_NUMS + value_offset) * K_HEAD_DIMS;
    const int v_blocks = K_HEAD_DIMS / V_BLOCK_SIZE;
    const int h = gid1 / v_blocks;
    const int group_size = V_HEAD_NUMS / Q_HEAD_NUMS;
    const int qk_h = h / group_size;
    const int v_block_id = gid1 - h * v_blocks;
    const int i_v_base = v_block_id * V_BLOCK_SIZE;
    const int id_sg_local = get_sub_group_local_id();

    // Per-batch pointers
    const __global INPUT0_TYPE* q_ptr = q + b * (Q_HEAD_NUMS * seq_len * K_HEAD_DIMS);
    const __global INPUT1_TYPE* k_ptr = k + b * (KEY_STEP_STRIDE * seq_len);
    const __global INPUT2_TYPE* v_ptr = v + b * (VALUE_STEP_STRIDE * seq_len);
    const __global INPUT3_TYPE* g_ptr = g_input + b * (V_HEAD_NUMS * seq_len);
    const __global INPUT4_TYPE* beta_ptr = beta_input + b * (V_HEAD_NUMS * seq_len);
    const int out_base = b * V_HEAD_NUMS * seq_len * K_HEAD_DIMS + h * K_HEAD_DIMS;

    // State registers: float8 per v_block (8 components × 16 SIMD lanes = 128 elements)
    float8 state[V_BLOCK_SIZE];

    // Load initial state
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS
                      + h * K_HEAD_DIMS * K_HEAD_DIMS
                      + i_v * K_HEAD_DIMS;
        state[iv] = convert_float8(BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base));
    }

    // Base addresses for token 0
    const int q_base0 = qk_h * K_HEAD_DIMS;
    const int k_base0 = (qk_h + key_offset) * K_HEAD_DIMS;
    const int v_base0 = (h + value_offset) * K_HEAD_DIMS;

#if (OPT_ALGO >= 1)
    // ========================================================================
    // ALGEBRAIC FOLD MODE (OPT_ALGO=1,2):
    // Key insight: output = g * dot(state_old, q) + v_new * dot(k, q)
    //              state = fma(state_old, g, k * v_new)
    //
    // Benefits:
    // - Eliminates state *= g multiply (saves 8 FLOPs/lane × 4 iv = 512 FLOPs/token)
    // - dot(state,k) and dot(state,q) are INDEPENDENT (shorter dependency chain)
    // - dot(k,q) computed once per token, shared across all iv iterations
    // ========================================================================

#if (OPT_ALGO == 2)
    // Pre-load first token's K/Q for double-buffering
    float8 next_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base0));
    float8 next_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base0));
    // Normalize ahead
    {
        float ks = sum8_o(next_k * next_k);
        ks = sub_group_reduce_add(ks);
        next_k *= native_rsqrt(ks + 0.000001f);
        float qs = sum8_o(next_q * next_q);
        qs = sub_group_reduce_add(qs);
        next_q *= native_rsqrt(qs + 0.000001f) * SCALE_FACTOR;
    }
#endif

    {
        int q_off = q_base0;
        int k_off = k_base0;
        int v_off = v_base0;
        int out_off = out_base;

        for (int i = 0; i < seq_len; i++) {
            const float b_g = native_exp(convert_float(g_ptr[i * V_HEAD_NUMS + h]));
            const float b_beta = convert_float(beta_ptr[i * V_HEAD_NUMS + h]);

#if (OPT_ALGO == 2)
            // Double-buffered: use pre-loaded and pre-normalized K/Q
            float8 b_k = next_k;
            float8 b_q = next_q;
            // Prefetch next token's K/Q
            if (i + 1 < seq_len) {
                next_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_off + KEY_STEP_STRIDE));
                next_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_off + STEP_STRIDE));
                float ks = sum8_o(next_k * next_k);
                ks = sub_group_reduce_add(ks);
                next_k *= native_rsqrt(ks + 0.000001f);
                float qs = sum8_o(next_q * next_q);
                qs = sub_group_reduce_add(qs);
                next_q *= native_rsqrt(qs + 0.000001f) * SCALE_FACTOR;
            }
#else
            // Standard load + normalize
            float8 b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_off));
            float8 b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_off));
            {
                float k_sum = sum8_o(b_k * b_k);
                k_sum = sub_group_reduce_add(k_sum);
                b_k *= native_rsqrt(k_sum + 0.000001f);
            }
            {
                float q_sum = sum8_o(b_q * b_q);
                q_sum = sub_group_reduce_add(q_sum);
                b_q *= native_rsqrt(q_sum + 0.000001f) * SCALE_FACTOR;
            }
#endif

            // Pre-compute dot(k, q) — shared across all iv iterations
            float kq_local = sum8_o(b_k * b_q);
            float kq_dot = sub_group_reduce_add(kq_local);

            #pragma unroll
            for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
                const int i_v = i_v_base + iv;

                // Two INDEPENDENT dot products using old (un-decayed) state
                float dsk_local = sum8_o(state[iv] * b_k);
                float dsq_local = sum8_o(state[iv] * b_q);
                float dsk = sub_group_reduce_add(dsk_local);
                float dsq = sub_group_reduce_add(dsq_local);

                // v_new = (v - g * dot(state_old, k)) * beta
                float hk = b_g * dsk;
                const int v_base_aligned = v_off + (i_v & ~(SUBGROUP_SIZE - 1));
                const int v_lane = i_v & (SUBGROUP_SIZE - 1);
                INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                float b_v = sub_group_broadcast(convert_float(v_val_h), v_lane);
                float v_new = (b_v - hk) * b_beta;

                // Output: g * dot(state_old, q) + v_new * dot(k, q)
                float out_acc = fma(b_g, dsq, v_new * kq_dot);
                if (id_sg_local == 0) {
                    output[out_off + i_v] = TO_OUTPUT_TYPE(out_acc);
                }

                // State: state_old * g + k * v_new  (decay folded into FMA)
                state[iv] = fma(state[iv], (float8)(b_g), b_k * (float8)(v_new));
            }

            q_off += STEP_STRIDE;
            k_off += KEY_STEP_STRIDE;
            v_off += VALUE_STEP_STRIDE;
            out_off += OUTPUT_STEP_STRIDE;
        }
    }

#else
    // ========================================================================
    // SIMPLE MODE (OPT_ALGO=0): Same structure as ref kernel, with
    // native_exp/native_rsqrt, removed redundant broadcasts, unrolled iv loop.
    // ========================================================================
    {
        int q_base = q_base0;
        int k_base = k_base0;
        int v_base = v_base0;
        int out_i_base = out_base;

        for (int i = 0; i < seq_len; i++) {
            const float b_g = native_exp(convert_float(g_ptr[i * V_HEAD_NUMS + h]));
            const float b_beta = convert_float(beta_ptr[i * V_HEAD_NUMS + h]);

            // Load K and Q
            float8 b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base));
            float8 b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base));

            // Normalize K: L2-norm with native_rsqrt
            {
                float k_sum = sum8_o(b_k * b_k);
                k_sum = sub_group_reduce_add(k_sum);
                b_k *= native_rsqrt(k_sum + 0.000001f);
            }

            // Normalize Q: L2-norm with native_rsqrt + scale factor
            {
                float q_sum = sum8_o(b_q * b_q);
                q_sum = sub_group_reduce_add(q_sum);
                b_q *= native_rsqrt(q_sum + 0.000001f) * SCALE_FACTOR;
            }

            #pragma unroll
            for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
                const int i_v = i_v_base + iv;

                state[iv] *= b_g;

                float hk_acc = sum8_o(state[iv] * b_k);
                hk_acc = sub_group_reduce_add(hk_acc);

                const int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
                const int v_lane = i_v & (SUBGROUP_SIZE - 1);
                INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                float b_v = sub_group_broadcast(convert_float(v_val_h), v_lane);
                b_v -= hk_acc;
                b_v *= b_beta;
                state[iv] = fma(b_k, (float8)(b_v), state[iv]);

                float out_acc = sum8_o(state[iv] * b_q);
                out_acc = sub_group_reduce_add(out_acc);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
            }

            q_base += STEP_STRIDE;
            k_base += KEY_STEP_STRIDE;
            v_base += VALUE_STEP_STRIDE;
            out_i_base += OUTPUT_STEP_STRIDE;
        }
    }
#endif

    // Store final state
    __global INPUT5_TYPE* state_out = initial_state;
#if OUTPUT_STATE
    state_out = (__global INPUT5_TYPE*)output_state;
#endif
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS
                      + h * K_HEAD_DIMS * K_HEAD_DIMS
                      + i_v * K_HEAD_DIMS;
        BLOCK_WRITEN(INPUT5_TYPE, 8, state_out, init_base, TO_INPUT5_TYPE8(state[iv]));
    }
}
