// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"


#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_kv_cache.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)


inline uint FUNC(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
    return OUTPUT1_GET_INDEX(b, f, y, x);
}

inline uint FUNC(get_scales_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
#ifdef SCALES_OUTPUT_ORDER
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR SCALES_OUTPUT_ORDER);
#else
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
#endif
}

#define SUBGROUP_SIZE 16
#define INNERMOST_DIM_VALUE INPUT0_SIZE_X
#ifdef INNERMOST_GROUP_SIZE
// Sub-group quantization: override innermost dim to process only group_size elements
#undef INNERMOST_DIM_VALUE
#define INNERMOST_DIM_VALUE INNERMOST_GROUP_SIZE
#endif
#define INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#if !QUANTIZE_4BIT
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#endif

__attribute__((reqd_work_group_size(SUBGROUP_SIZE, SUBGROUPS_NUMBER, 1)))
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(dynamic_quantize_gpu_kv_cache)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if QUANTIZE_4BIT
    __global uchar* output,
#else
    __global OUTPUT_TYPE* output,
#endif
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION && !GROUP_SCALES_WITH_ZP
    , __global OUTPUT2_TYPE* output_zp
#endif
#ifdef APPEND_MODE
    , const uint axis_offset
#endif
    )
{
    const uint sglid = get_sub_group_local_id();
    const uint grouped_indexes = get_global_id(1);
    const uint batch_indexes = get_global_id(2);

    DECLARE_BATCHED_DIMS_INDEXES(batch_indexes);
    DECLARE_GROUPED_DIMS_INDEXES(grouped_indexes);

#ifdef NUM_INNERMOST_GROUPS
    // Sub-group quantization: grouped_indexes encodes both other-grouped-dims and inner group
    const uint inner_group_id = grouped_indexes / SUBGROUPS_NUMBER;
    const uint x = inner_group_id * INNERMOST_DIM_VALUE;
    const uint scale_x = inner_group_id;
#else
    // The innermost dimension is always processed in the loop inside the kernel
    const uint x = 0;
    const uint scale_x = 0;
#endif

    half grp_max = 0.001h;
    half max_value = INPUT0_VAL_MIN;
    half min_value = INPUT0_VAL_MAX;

    half val[INNERMOST_DIM_VALUE / SUBGROUP_SIZE];

    const uint input_offset = INPUT0_GET_INDEX(b, f, y, x);
#if QUANTIZE_4BIT
    // TurboQuant-matching i4 encode: L2-normalize in f32, then quantize with fixed 7/3.
    // Matches CPU TQ: x_norm = x/||x||; x_scaled = x_norm*sqrt(D); code = round(clip(x_scaled,±3)*7/3)
    // dequant_scale = ||x|| * 3/(7*sqrt(D))  (stored, SDPA multiplies code * dequant_scale)
    float sum_sq = 0.0f;
    float val_f32[INNERMOST_DIM_VALUE / SUBGROUP_SIZE];
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
        val[i] = INPUT_BLOCK_READ(input, input_offset + i * SUBGROUP_SIZE);
        val_f32[i] = convert_float(val[i]);
        sum_sq += val_f32[i] * val_f32[i];
    }
#else
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
        val[i] = INPUT_BLOCK_READ(input, input_offset + i * SUBGROUP_SIZE);
#if ASYMMETRIC_QUANTIZATION
        max_value = fmax(max_value, val[i]);
        min_value = fmin(min_value, val[i]);
#else
        max_value = fmax(max_value, fabs(val[i]));
#endif
    }
    max_value = fmax(max_value, grp_max);
#endif

#if ASYMMETRIC_QUANTIZATION
    min_value = work_group_reduce_min(min_value);
    max_value = work_group_reduce_max(max_value);

    // If the range of input data is zero, it is adjusted to the minimum value(0.001).
    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)(scale_tmp);
    OUTPUT1_TYPE zp = (OUTPUT1_TYPE)(zp_tmp);

#elif QUANTIZE_4BIT
    // L2-norm + fixed 7/3 quantization (matches CPU TurboQuant exactly).
    // L2_norm = sqrt(sum_sq) over sub-group; normalize, scale by sqrt(D), clip ±3, quant 7/3.
    // dequant_scale = L2_norm * 3 / (7 * sqrt(D)) — stores inverse of encode scale.
    float sum_sq_total = work_group_reduce_add(sum_sq);
    float l2_norm = sqrt(sum_sq_total);
    l2_norm = fmax(l2_norm, 1e-6f);
    float inv_norm = 1.0f / l2_norm;
    float sqrt_D = sqrt((float)INNERMOST_DIM_VALUE);
    // scale_to_int = 7/3, applied after normalization+sqrt(D) scaling
    // dequant_scale stored so SDPA can do: code * dequant_scale ≈ original_value
    OUTPUT1_TYPE dequant_scale = convert_half(3.0f * l2_norm / (7.0f * sqrt_D));
#else
    max_value = work_group_reduce_max(max_value);
    OUTPUT1_TYPE scale = 127.0h / max_value;
#endif

#ifdef APPEND_MODE
    APPEND_AXIS_NAME += axis_offset;
#endif

#if QUANTIZE_4BIT
    // i4 packing: pair adjacent elements (even sglid = low nibble, odd = high nibble)
    // Compute base byte offset for this token's output row
    // For bfyx layout with i4, byte offset = element_offset / 2
    const uint elem_offset = OUTPUT_GET_INDEX(b, f, y, x);

    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
        // Exact CPU TQ encode: normalize → sqrt(D) → clip ±3 → 7/3
        float v_norm = val_f32[i] * inv_norm;       // unit vector
        float v_scaled = v_norm * sqrt_D;            // ~ N(0,1)
        float v_clipped = clamp(v_scaled, -3.0f, 3.0f);
        char q = clamp(convert_char_rte(v_clipped * (7.0f / 3.0f)), (char)-7, (char)7);
        // Exchange with neighbor (even<->odd) via subgroup shuffle
        char q_neighbor = intel_sub_group_shuffle(q, sglid ^ 1);
        if ((sglid & 1) == 0) {
            // Even work item: pack (my_value=low nibble, neighbor=high nibble)
            uchar lo = (uchar)q & 0x0F;
            uchar hi = ((uchar)q_neighbor & 0x0F) << 4;
            uchar packed = lo | hi;
            uint byte_idx = (elem_offset + i * SUBGROUP_SIZE + sglid) >> 1;
            output[byte_idx] = packed;
        }
    }
#else
    const uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
#if ASYMMETRIC_QUANTIZATION
        OUTPUT_TYPE res = convert_char_rte(val[i] * scale + zp);
#else
        OUTPUT_TYPE res = convert_char_rte(val[i] * scale);
#endif
        OUTPUT_BLOCK_WRITE(output, output_offset + i * SUBGROUP_SIZE, res);
    }
#endif

    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, scale_x);

    // Use get_local_id(1) instead of grouped_indexes to handle sub-group dispatch correctly:
    // each inner-group work group should write its own scale independently.
    if (get_local_id(1) == 0 && sglid == 0) {
#if ASYMMETRIC_QUANTIZATION
        output_scale[scale_idx] = 1.0h / scale;
#if GROUP_SCALES_WITH_ZP
        output_scale[scale_idx + 1] = zp;
#else

    #if OUTPUT2_IS_FP
        output_zp[scale_idx] = zp;
    #else
        output_zp[scale_idx] = convert_char_rte(zp);
    #endif

#endif
#elif QUANTIZE_4BIT
        // Store dequant scale directly (not 1/scale) — SDPA does: code * dequant_scale
        output_scale[scale_idx] = dequant_scale;
#else
        output_scale[scale_idx] = 1.0h / scale;
#endif
    }
}
