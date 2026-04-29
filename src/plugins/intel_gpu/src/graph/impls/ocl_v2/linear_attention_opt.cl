// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Optimized LinearAttention (GatedDeltaNet) OCL kernel.
// Target: K_HEAD_DIMS=128, SUBGROUP_SIZE=16 (Xe2/Xe3)
//
// Optimizations over ref kernel:
// 1. native_exp / native_rsqrt for cheaper transcendentals
// 2. Removed redundant sub_group_broadcast after sub_group_reduce_add
// 3. #pragma unroll on V_BLOCK_SIZE inner loop
// 4. When OPT_CHUNK_SIZE > 0: pre-compute K/Q norm scales, exp(g), beta
//    in register-cached chunks to shorten the recurrent critical path.
//
// Same dispatch model as ref: global={B, V_HEADS*v_blocks, SG_SIZE}, local={1,1,SG_SIZE}
//
// JIT Constants: K_HEAD_DIMS, V_HEAD_NUMS, Q_HEAD_NUMS, SUBGROUP_SIZE,
//                SCALE_FACTOR, IO_TYPE, OUTPUT_STATE, OPT_CHUNK_SIZE

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

#if (OPT_CHUNK_SIZE > 0)
    // ========================================================================
    // CHUNK MODE: Pre-compute norm scales, exp(g), beta in register caches,
    // then run recurrent update using cached scalars (no reduce/rsqrt on
    // the critical path in Phase 2).
    // ========================================================================
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += OPT_CHUNK_SIZE) {
        const int chunk_end = min(chunk_start + OPT_CHUNK_SIZE, seq_len);
        const int chunk_len = chunk_end - chunk_start;

        // Phase 1: Pre-compute per-token scalar values
        float c_k_scale[OPT_CHUNK_SIZE];
        float c_q_scale[OPT_CHUNK_SIZE];
        float c_g_exp[OPT_CHUNK_SIZE];
        float c_beta[OPT_CHUNK_SIZE];

        {
            int q_off = q_base0 + chunk_start * STEP_STRIDE;
            int k_off = k_base0 + chunk_start * KEY_STEP_STRIDE;

            for (int t = 0; t < chunk_len; t++) {
                const int gi = (chunk_start + t) * V_HEAD_NUMS + h;
                c_g_exp[t] = native_exp(convert_float(g_ptr[gi]));
                c_beta[t] = convert_float(beta_ptr[gi]);

                // Load K, compute L2-norm scale
                float8 tmp_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_off));
                float k_ss = sum8_o(tmp_k * tmp_k);
                k_ss = sub_group_reduce_add(k_ss);
                c_k_scale[t] = native_rsqrt(k_ss + 0.000001f);

                // Load Q, compute L2-norm scale with SCALE_FACTOR
                float8 tmp_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_off));
                float q_ss = sum8_o(tmp_q * tmp_q);
                q_ss = sub_group_reduce_add(q_ss);
                c_q_scale[t] = native_rsqrt(q_ss + 0.000001f) * SCALE_FACTOR;

                q_off += STEP_STRIDE;
                k_off += KEY_STEP_STRIDE;
            }
        }

        // Phase 2: Recurrent state update with pre-computed values
        {
            int q_off = q_base0 + chunk_start * STEP_STRIDE;
            int k_off = k_base0 + chunk_start * KEY_STEP_STRIDE;
            int v_off = v_base0 + chunk_start * VALUE_STEP_STRIDE;
            int out_off = out_base + chunk_start * OUTPUT_STEP_STRIDE;

            for (int t = 0; t < chunk_len; t++) {
                const float b_g = c_g_exp[t];
                const float b_beta = c_beta[t];

                // Re-load K/Q (L1-cached from Phase 1) and apply pre-computed scales
                float8 b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_off));
                b_k *= c_k_scale[t];
                float8 b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_off));
                b_q *= c_q_scale[t];

                #pragma unroll
                for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
                    const int i_v = i_v_base + iv;

                    // Decay state
                    state[iv] *= b_g;

                    // h_k = dot(state, k)
                    float hk_acc = sum8_o(state[iv] * b_k);
                    hk_acc = sub_group_reduce_add(hk_acc);

                    // Load V element and broadcast
                    const int v_base_aligned = v_off + (i_v & ~(SUBGROUP_SIZE - 1));
                    const int v_lane = i_v & (SUBGROUP_SIZE - 1);
                    INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                    float b_v = sub_group_broadcast(convert_float(v_val_h), v_lane);

                    // v_new = (v - h_k) * beta
                    b_v -= hk_acc;
                    b_v *= b_beta;

                    // State update: state += outer(k, v_new)
                    state[iv] = fma(b_k, (float8)(b_v), state[iv]);

                    // Output = dot(state, q)
                    float out_acc = sum8_o(state[iv] * b_q);
                    out_acc = sub_group_reduce_add(out_acc);
                    if (id_sg_local == 0) {
                        output[out_off + i_v] = TO_OUTPUT_TYPE(out_acc);
                    }
                }

                q_off += STEP_STRIDE;
                k_off += KEY_STEP_STRIDE;
                v_off += VALUE_STEP_STRIDE;
                out_off += OUTPUT_STEP_STRIDE;
            }
        }
    }

#else
    // ========================================================================
    // SIMPLE MODE: Same structure as ref kernel, with native_exp/native_rsqrt,
    // removed redundant broadcasts, and unrolled iv loop.
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
