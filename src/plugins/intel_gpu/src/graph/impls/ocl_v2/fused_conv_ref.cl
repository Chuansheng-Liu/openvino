#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// FusedConv kernel: fuses Gather(beam_idx) + Concat + DepthwiseConv1D + SiLU + Slice
//
// Inputs:
//   INPUT0 (input):       [B, CONV_DIM, S]
//   INPUT1 (conv_weight): [CONV_DIM, KERNEL_SIZE]
//   INPUT2 (beam_idx):    [B]
//   INPUT3 (state_in):    [B, CONV_DIM, KERNEL_SIZE]
//
// Outputs:
//   OUTPUT  (output):     [B, CONV_DIM, S]
//   OUTPUT1 (state_out):  [B, CONV_DIM, KERNEL_SIZE]
//
// Dispatch: global = {batch, conv_dim, 1}, local = {1, WG_SIZE, 1}

KERNEL(fused_conv_ref)(
    __global INPUT0_TYPE* input,
    __global INPUT1_TYPE* conv_weight,
    __global INPUT2_TYPE* beam_idx,
    __global INPUT3_TYPE* state_in,
    __global INPUT4_TYPE* state_update_mode,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* state_out,
#if OUTPUT_SNAPSHOTS
    __global OUTPUT2_TYPE* state_snapshots,
#endif
    int seq_len)
{
    const int b  = get_global_id(0);
    const int ch = get_global_id(1);

    if (ch >= CONV_DIM)
        return;

    const int mode = convert_int(state_update_mode[0]);

#if OUTPUT_SNAPSHOTS
#ifndef SNAP_SEQ_LEN
#define SNAP_SEQ_LEN seq_len
#endif
    // Snapshot commit-by-index: mode < 0 means commit snapshot[(-mode)-1]
    // to the variable state before processing.
    if (mode < 0) {
        const int commit_idx = (-mode) - 1;
        const int snap_src = b * SNAP_SEQ_LEN * CONV_DIM * KERNEL_SIZE
                           + commit_idx * CONV_DIM * KERNEL_SIZE
                           + ch * KERNEL_SIZE;
        const int state_dst = b * CONV_DIM * KERNEL_SIZE + ch * KERNEL_SIZE;
        for (int k = 0; k < KERNEL_SIZE; k++)
            state_out[state_dst + k] = state_snapshots[snap_src + k];
    }
#endif

    // 1. Beam search reorder: read state from beam_idx[b] source batch
    const int src_b = convert_int(beam_idx[b]);

    // 2. Load state (KERNEL_SIZE values)
    float state[KERNEL_SIZE];
    const int state_in_base = src_b * CONV_DIM * KERNEL_SIZE + ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        state[k] = convert_float(state_in[state_in_base + k]);

    // 3. Load conv weight
    float w[KERNEL_SIZE];
    const int w_base = ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        w[k] = convert_float(conv_weight[w_base + k]);

    // 4. For each sequence position: depthwise conv + SiLU
    const int io_base = b * CONV_DIM * seq_len + ch * seq_len;
    for (int s = 0; s < seq_len; s++) {
        float x_new = convert_float(input[io_base + s]);

        // Conv window = [state[1], state[2], ..., state[K-1], x_new]
        // This corresponds to concatenating state with input and applying valid conv
        float acc = 0.0f;
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            acc += state[k + 1] * w[k];
        acc += x_new * w[KERNEL_SIZE - 1];

        // SiLU activation: x * sigmoid(x)
        float sig = native_recip(1.0f + native_exp(-acc));
        output[io_base + s] = TO_OUTPUT_TYPE(acc * sig);

        // Shift state left, append new input
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            state[k] = state[k + 1];
        state[KERNEL_SIZE - 1] = x_new;

#if OUTPUT_SNAPSHOTS
        // Write per-step state snapshot only during verify (mode <= 0).
        // Prefill (mode > 0) skips snapshot writes to avoid O(prompt_len) GPU memory.
        if (mode <= 0) {
            const int snap_base = b * SNAP_SEQ_LEN * CONV_DIM * KERNEL_SIZE
                                + s * CONV_DIM * KERNEL_SIZE
                                + ch * KERNEL_SIZE;
            for (int k = 0; k < KERNEL_SIZE; k++)
                state_snapshots[snap_base + k] = TO_OUTPUT2_TYPE(state[k]);
        }
#endif
    }

    // 5. Write back updated state only when mode > 0 (normal commit).
    //    mode == 0: no commit (deferred verify)
    //    mode <  0: snapshot was committed above; current step is also deferred
    if (mode > 0) {
        const int state_out_base = b * CONV_DIM * KERNEL_SIZE + ch * KERNEL_SIZE;
        for (int k = 0; k < KERNEL_SIZE; k++)
            state_out[state_out_base + k] = TO_OUTPUT1_TYPE(state[k]);
    }
}
