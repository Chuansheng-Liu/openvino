// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Chunk-wise LinearAttention (GatedDeltaNet) OCL kernel.
//
// Phase 1: Recurrent within chunks, with g/beta pre-caching per chunk.
//          Produces identical output to the ref kernel.
//          Provides infrastructure for Phase 2 (parallel intra-chunk ops + DPAS).
//
// Same dispatch model as ref: global={B, V_HEADS*v_blocks, SG_SIZE}, local={1,1,SG_SIZE}
//
// JIT Constants: CHUNK_SIZE, K_HEAD_DIMS, V_HEAD_NUMS, Q_HEAD_NUMS,
//                SUBGROUP_SIZE, SCALE_FACTOR, IO_TYPE, OUTPUT_STATE

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#ifndef TO_INPUT5_TYPE2
#    define TO_INPUT5_TYPE2(x) CAT(convert_, MAKE_VECTOR_TYPE(INPUT5_TYPE, 2))(x)
#endif

#ifndef TO_INPUT5_TYPE8
#    define TO_INPUT5_TYPE8(x) CAT(convert_, MAKE_VECTOR_TYPE(INPUT5_TYPE, 8))(x)
#endif

#define V_BLOCK_SIZE 4

float sum8_c(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

inline float l2norm_scale_c(float sum, float extra_scale) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + 0.000001f) * extra_scale;
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_chunk)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* g,
 __global INPUT4_TYPE* beta,
 __global INPUT5_TYPE* initial_state,
 __global OUTPUT_TYPE* output,
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,
#endif
 int seq_len,
 int key_offset,
 int value_offset) {

    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int BATCH_STRIDE = Q_HEAD_NUMS * seq_len * K_HEAD_DIMS;
    int STEP_STRIDE = Q_HEAD_NUMS * K_HEAD_DIMS;
    int OUTPUT_STEP_STRIDE = V_HEAD_NUMS * K_HEAD_DIMS;
    int KEY_STEP_STRIDE = (Q_HEAD_NUMS + key_offset) * K_HEAD_DIMS;
    int VALUE_STEP_STRIDE = (V_HEAD_NUMS + value_offset) * K_HEAD_DIMS;
    int KEY_BATCH_STRIDE = KEY_STEP_STRIDE * seq_len;
    int VALUE_BATCH_STRIDE = VALUE_STEP_STRIDE * seq_len;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int group_size = V_HEAD_NUMS / Q_HEAD_NUMS;
    int qk_h = h / group_size;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    const __global INPUT0_TYPE* q_ptr = q + b * BATCH_STRIDE;
    const __global INPUT1_TYPE* k_ptr = k + b * KEY_BATCH_STRIDE;
    const __global INPUT2_TYPE* v_ptr = v + b * VALUE_BATCH_STRIDE;
    const __global INPUT3_TYPE* g_ptr = g + b * V_HEAD_NUMS * seq_len;
    const __global INPUT4_TYPE* beta_ptr = beta + b * V_HEAD_NUMS * seq_len;
    int out_base = b * V_HEAD_NUMS * seq_len * K_HEAD_DIMS + h * K_HEAD_DIMS;

#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
    float8 init_state[V_BLOCK_SIZE][2];
    float8 b_k[2];
    float8 b_q[2];
#    else
    float8 init_state[V_BLOCK_SIZE];
    float8 b_k;
    float8 b_q;
#    endif
#elif (K_HEAD_DIMS % 32) == 0
    float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
    float2 b_k[K_HEAD_DIMS / 32];
    float2 b_q[K_HEAD_DIMS / 32];
#else
    float init_state[V_BLOCK_SIZE][CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
    float b_k[CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
    float b_q[CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
#endif
    int id_sg_local = get_sub_group_local_id();

    // Load initial state (same as ref kernel)
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
#        define DATA_VEC_LS MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
        DATA_VEC_LS h8_0 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
        DATA_VEC_LS h8_1 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base + (SUBGROUP_SIZE * 8));
#        undef DATA_VEC_LS
        init_state[iv][0] = convert_float8(h8_0);
        init_state[iv][1] = convert_float8(h8_1);
#    else
#        define DATA_VEC_LS MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
        DATA_VEC_LS h8 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
#        undef DATA_VEC_LS
        init_state[iv] = convert_float8(h8);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
#            define DATA_VEC_LS MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
            DATA_VEC_LS h2 = BLOCK_READN(INPUT5_TYPE, 2, initial_state, init_base + (j - id_sg_local));
#            undef DATA_VEC_LS
            init_state[iv][idx] = convert_float2(h2);
        }
#    else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            init_state[iv][idx] = convert_float(initial_state[init_base + j]);
        }
#    endif
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            init_state[iv][idx] = convert_float(initial_state[init_base + j]);
        }
#endif
    }

    // Chunk-wise iteration over the sequence
    int num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    int q_base_step = qk_h * K_HEAD_DIMS;
    int k_base_step = (qk_h + key_offset) * K_HEAD_DIMS;
    int v_base_step = (h + value_offset) * K_HEAD_DIMS;
    int out_i_base_step = out_base;

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int chunk_start = chunk_idx * CHUNK_SIZE;
        int chunk_end = min(chunk_start + CHUNK_SIZE, seq_len);
        int chunk_len = chunk_end - chunk_start;

        // Pre-cache g and beta for this chunk to reduce global memory reads
        float g_cache[CHUNK_SIZE];
        float beta_cache[CHUNK_SIZE];
        for (int t = 0; t < chunk_len; t++) {
            g_cache[t] = convert_float(g_ptr[(chunk_start + t) * V_HEAD_NUMS + h]);
            beta_cache[t] = convert_float(beta_ptr[(chunk_start + t) * V_HEAD_NUMS + h]);
        }

        // Process tokens within this chunk (recurrent, same as ref)
        int q_base = q_base_step + chunk_start * STEP_STRIDE;
        int k_base = k_base_step + chunk_start * KEY_STEP_STRIDE;
        int v_base = v_base_step + chunk_start * VALUE_STEP_STRIDE;
        int out_i_base = out_i_base_step + chunk_start * OUTPUT_STEP_STRIDE;

        for (int t = 0; t < chunk_len; t++, q_base += STEP_STRIDE, k_base += KEY_STEP_STRIDE, v_base += VALUE_STEP_STRIDE, out_i_base += OUTPUT_STEP_STRIDE) {
            float b_g = exp(g_cache[t]);
            float b_beta = beta_cache[t];

            // Load k and q
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
            b_k[0] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base));
            b_k[1] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base + (SUBGROUP_SIZE * 8)));
            b_q[0] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base));
            b_q[1] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base + (SUBGROUP_SIZE * 8)));
#    else
            b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base));
            b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base));
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
#        pragma unroll
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_k[idx] = convert_float2(BLOCK_READN(INPUT1_TYPE, 2, k_ptr, k_base + (j - id_sg_local)));
                b_q[idx] = convert_float2(BLOCK_READN(INPUT0_TYPE, 2, q_ptr, q_base + (j - id_sg_local)));
            }
#    else
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] = convert_float(k_ptr[k_base + j]);
                b_q[idx] = convert_float(q_ptr[q_base + j]);
            }
#    endif
#else
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] = convert_float(k_ptr[k_base + j]);
                b_q[idx] = convert_float(q_ptr[q_base + j]);
            }
#endif

            // Normalize k and q
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
            {
                float k_sum = sum8_c(b_k[0] * b_k[0]) + sum8_c(b_k[1] * b_k[1]);
                float k_scale = l2norm_scale_c(k_sum, 1.0f);
                b_k[0] *= k_scale; b_k[1] *= k_scale;
                float q_sum = sum8_c(b_q[0] * b_q[0]) + sum8_c(b_q[1] * b_q[1]);
                float q_scale = l2norm_scale_c(q_sum, SCALE_FACTOR);
                b_q[0] *= q_scale; b_q[1] *= q_scale;
            }
#    else
            {
                float k_sum = sum8_c(b_k * b_k);
                float k_scale = l2norm_scale_c(k_sum, 1.0f);
                b_k *= k_scale;
                float q_sum = sum8_c(b_q * b_q);
                float q_scale = l2norm_scale_c(q_sum, SCALE_FACTOR);
                b_q *= q_scale;
            }
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
            {
                float k_sum = 0.0f, q_sum = 0.0f;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    k_sum += b_k[idx].s0 * b_k[idx].s0 + b_k[idx].s1 * b_k[idx].s1;
                    q_sum += b_q[idx].s0 * b_q[idx].s0 + b_q[idx].s1 * b_q[idx].s1;
                }
                float k_scale = l2norm_scale_c(k_sum, 1.0f);
                float q_scale = l2norm_scale_c(q_sum, SCALE_FACTOR);
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    b_k[idx] *= k_scale;
                    b_q[idx] *= q_scale;
                }
            }
#    else
            {
                float k_sum = 0.0f, q_sum = 0.0f;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                    int idx = j / SUBGROUP_SIZE;
                    k_sum = fma(b_k[idx], b_k[idx], k_sum);
                    q_sum = fma(b_q[idx], b_q[idx], q_sum);
                }
                float k_scale = l2norm_scale_c(k_sum, 1.0f);
                float q_scale = l2norm_scale_c(q_sum, SCALE_FACTOR);
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                    int idx = j / SUBGROUP_SIZE;
                    b_k[idx] *= k_scale;
                    b_q[idx] *= q_scale;
                }
            }
#    endif
#else
            {
                float k_sum = 0.0f, q_sum = 0.0f;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                    int idx = j / SUBGROUP_SIZE;
                    k_sum = fma(b_k[idx], b_k[idx], k_sum);
                    q_sum = fma(b_q[idx], b_q[idx], q_sum);
                }
                float k_scale = l2norm_scale_c(k_sum, 1.0f);
                float q_scale = l2norm_scale_c(q_sum, SCALE_FACTOR);
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                    int idx = j / SUBGROUP_SIZE;
                    b_k[idx] *= k_scale;
                    b_q[idx] *= q_scale;
                }
            }
#endif

            // Per-v_block recurrent update (identical to ref kernel)
            for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
                int i_v = i_v_base + iv;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
                init_state[iv][0] *= b_g;
                init_state[iv][1] *= b_g;
                float hk_acc = sum8_c(init_state[iv][0] * b_k[0]) + sum8_c(init_state[iv][1] * b_k[1]);
                hk_acc = sub_group_reduce_add(hk_acc);
                hk_acc = sub_group_broadcast(hk_acc, 0);

                int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
                int v_lane = i_v & (SUBGROUP_SIZE - 1);
                INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                float v_val = convert_float(v_val_h);
                float b_v = sub_group_broadcast(v_val, v_lane);
                b_v -= hk_acc;
                b_v *= b_beta;
                init_state[iv][0] = fma(b_k[0], (float8)(b_v), init_state[iv][0]);
                init_state[iv][1] = fma(b_k[1], (float8)(b_v), init_state[iv][1]);

                float out_acc = sum8_c(init_state[iv][0] * b_q[0]) + sum8_c(init_state[iv][1] * b_q[1]);
                out_acc = sub_group_reduce_add(out_acc);
                out_acc = sub_group_broadcast(out_acc, 0);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
#    else
                init_state[iv] *= b_g;
                float hk_acc = sum8_c(init_state[iv] * b_k);
                hk_acc = sub_group_reduce_add(hk_acc);
                hk_acc = sub_group_broadcast(hk_acc, 0);

                int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
                int v_lane = i_v & (SUBGROUP_SIZE - 1);
                INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                float v_val = convert_float(v_val_h);
                float b_v = sub_group_broadcast(v_val, v_lane);
                b_v -= hk_acc;
                b_v *= b_beta;
                init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

                float out_acc = sum8_c(init_state[iv] * b_q);
                out_acc = sub_group_reduce_add(out_acc);
                out_acc = sub_group_broadcast(out_acc, 0);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    init_state[iv][idx] *= b_g;
                }
                float hk_acc = 0.0f;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    hk_acc += init_state[iv][idx].s0 * b_k[idx].s0 + init_state[iv][idx].s1 * b_k[idx].s1;
                }
                hk_acc = sub_group_reduce_add(hk_acc);
                hk_acc = sub_group_broadcast(hk_acc, 0);

                int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
                int v_lane = i_v & (SUBGROUP_SIZE - 1);
                INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
                float v_val = convert_float(v_val_h);
                float b_v = sub_group_broadcast(v_val, v_lane);
                b_v -= hk_acc;
                b_v *= b_beta;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
                }

                float out_acc = 0.0f;
                for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                    int idx = j >> 5;
                    out_acc += init_state[iv][idx].s0 * b_q[idx].s0 + init_state[iv][idx].s1 * b_q[idx].s1;
                }
                out_acc = sub_group_reduce_add(out_acc);
                out_acc = sub_group_broadcast(out_acc, 0);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
#    else
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    init_state[iv][idx] *= b_g;
                }
                float hk_acc = 0.0f;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    hk_acc += init_state[iv][idx].s0 * b_k[idx].s0 + init_state[iv][idx].s1 * b_k[idx].s1;
                }
                hk_acc = sub_group_reduce_add(hk_acc);
                hk_acc = sub_group_broadcast(hk_acc, 0);

                float b_v = convert_float(v_ptr[v_base + i_v]);
                b_v -= hk_acc;
                b_v *= b_beta;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
                }

                float out_acc = 0.0f;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    out_acc += init_state[iv][idx].s0 * b_q[idx].s0 + init_state[iv][idx].s1 * b_q[idx].s1;
                }
                out_acc = sub_group_reduce_add(out_acc);
                out_acc = sub_group_broadcast(out_acc, 0);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
#    endif
#else
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    init_state[iv][idx] *= b_g;
                }
                float hk_acc = 0.0f;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
                }
                hk_acc = sub_group_reduce_add(hk_acc);
                hk_acc = sub_group_broadcast(hk_acc, 0);

                float b_v = convert_float(v_ptr[v_base + i_v]);
                b_v -= hk_acc;
                b_v *= b_beta;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
                }

                float out_acc = 0.0f;
                for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                    int idx = n / SUBGROUP_SIZE;
                    out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
                }
                out_acc = sub_group_reduce_add(out_acc);
                out_acc = sub_group_broadcast(out_acc, 0);
                if (id_sg_local == 0) {
                    output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
                }
#endif
            }
        }  // end token loop within chunk
    }  // end chunk loop

    // Store final state
    __global INPUT5_TYPE* state_out = initial_state;
#if OUTPUT_STATE
    state_out = (__global INPUT5_TYPE*)output_state;
#endif
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
#        define DATA_VEC_SS MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
        DATA_VEC_SS h8_0_out = TO_INPUT5_TYPE8(init_state[iv][0]);
        DATA_VEC_SS h8_1_out = TO_INPUT5_TYPE8(init_state[iv][1]);
        BLOCK_WRITEN(INPUT5_TYPE, 8, state_out, init_base, h8_0_out);
        BLOCK_WRITEN(INPUT5_TYPE, 8, state_out, init_base + (SUBGROUP_SIZE * 8), h8_1_out);
#        undef DATA_VEC_SS
#    else
#        define DATA_VEC_SS MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
        DATA_VEC_SS h8_out = TO_INPUT5_TYPE8(init_state[iv]);
        BLOCK_WRITEN(INPUT5_TYPE, 8, state_out, init_base, h8_out);
#        undef DATA_VEC_SS
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
#            define DATA_VEC_SS MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
            DATA_VEC_SS h2_out = TO_INPUT5_TYPE2(init_state[iv][idx]);
            BLOCK_WRITEN(INPUT5_TYPE, 2, state_out, init_base + (j - id_sg_local), h2_out);
#            undef DATA_VEC_SS
        }
#    else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            state_out[init_base + j] = TO_INPUT5_TYPE(init_state[iv][idx]);
        }
#    endif
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            state_out[init_base + j] = TO_INPUT5_TYPE(init_state[iv][idx]);
        }
#endif
    }
}
