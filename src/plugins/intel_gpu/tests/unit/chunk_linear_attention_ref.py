"""
Numpy reference implementation of chunk-wise GatedDeltaNet (LinearAttention).

This serves as the test oracle for validating the OCL chunk kernel.
It implements both the recurrent and chunk-wise algorithms and verifies
they produce equivalent results.

The data layout matches the OCL kernel interface:
  q:             [B, seq_len, Q_HEAD_NUMS, K_HEAD_DIMS]
  k:             [B, seq_len, Q_HEAD_NUMS, K_HEAD_DIMS]
  v:             [B, seq_len, V_HEAD_NUMS, K_HEAD_DIMS]
  g:             [B, seq_len, V_HEAD_NUMS]
  beta:          [B, seq_len, V_HEAD_NUMS]
  initial_state: [B, V_HEAD_NUMS, K_HEAD_DIMS, K_HEAD_DIMS]

Output:
  output:        [B, seq_len, V_HEAD_NUMS, K_HEAD_DIMS]
  output_state:  [B, V_HEAD_NUMS, K_HEAD_DIMS, K_HEAD_DIMS]

Note: In Qwen3.5, Q_HEAD_NUMS (kv_heads) = 16, V_HEAD_NUMS = 32,
      so group_size = V_HEAD_NUMS / Q_HEAD_NUMS = 2.
      Each k-head is shared by 2 v-heads.
"""

import numpy as np
from typing import Tuple, Optional


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """L2 normalize along given axis."""
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def recurrent_gated_delta_rule(
    q: np.ndarray,       # [B, S, Q_HEADS, K_DIM]
    k: np.ndarray,       # [B, S, Q_HEADS, K_DIM]
    v: np.ndarray,       # [B, S, V_HEADS, V_DIM]
    g: np.ndarray,       # [B, S, V_HEADS]
    beta: np.ndarray,    # [B, S, V_HEADS]
    initial_state: np.ndarray,  # [B, V_HEADS, K_DIM, V_DIM]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recurrent (per-token) GatedDeltaNet — the reference ground truth.
    Matches the OCL linear_attention_ref kernel exactly.
    """
    B, S, Q_HEADS, K_DIM = q.shape
    V_HEADS = v.shape[2]
    V_DIM = v.shape[3]
    group_size = V_HEADS // Q_HEADS

    # L2 normalize q and k
    q = l2norm(q, axis=-1)
    k = l2norm(k, axis=-1)

    # Scale q
    scale = 1.0 / np.sqrt(K_DIM)
    q = q * scale

    # State: [B, V_HEADS, K_DIM, V_DIM]
    state = initial_state.copy().astype(np.float32)
    output = np.zeros((B, S, V_HEADS, V_DIM), dtype=np.float32)

    for i in range(S):
        for b in range(B):
            for h in range(V_HEADS):
                qk_h = h // group_size  # which q/k head this v-head maps to
                q_t = q[b, i, qk_h]    # [K_DIM]
                k_t = k[b, i, qk_h]    # [K_DIM]
                v_t = v[b, i, h]        # [V_DIM]
                g_t = np.exp(g[b, i, h])  # scalar
                beta_t = beta[b, i, h]    # scalar

                # Decay state
                state[b, h] *= g_t

                # Memory readout: h·k → sum over K_DIM
                hk = np.dot(state[b, h].T, k_t)  # [V_DIM]

                # Delta rule update
                delta = (v_t - hk) * beta_t  # [V_DIM]

                # Outer product update: state += k ⊗ delta
                state[b, h] += np.outer(k_t, delta)

                # Query output: q·state → sum over K_DIM
                output[b, i, h] = np.dot(state[b, h].T, q_t)  # [V_DIM]

    return output, state


def chunk_gated_delta_rule(
    q: np.ndarray,       # [B, S, Q_HEADS, K_DIM]
    k: np.ndarray,       # [B, S, Q_HEADS, K_DIM]
    v: np.ndarray,       # [B, S, V_HEADS, V_DIM]
    g: np.ndarray,       # [B, S, V_HEADS]
    beta: np.ndarray,    # [B, S, V_HEADS]
    initial_state: np.ndarray,  # [B, V_HEADS, K_DIM, V_DIM]
    chunk_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chunk-wise parallel GatedDeltaNet — matches HF torch_chunk_gated_delta_rule.
    This is what the OCL chunk kernel must produce.
    """
    B, S, Q_HEADS, K_DIM = q.shape
    V_HEADS = v.shape[2]
    V_DIM = v.shape[3]
    group_size = V_HEADS // Q_HEADS

    # L2 normalize q and k
    q = l2norm(q, axis=-1).astype(np.float32)
    k = l2norm(k, axis=-1).astype(np.float32)

    # Scale q
    scale = 1.0 / np.sqrt(K_DIM)
    q = q * scale

    # Pad sequence to multiple of chunk_size
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    if pad_size > 0:
        q = np.pad(q, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
        k = np.pad(k, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
        v = np.pad(v, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
        g = np.pad(g, ((0, 0), (0, pad_size), (0, 0)))
        beta = np.pad(beta, ((0, 0), (0, pad_size), (0, 0)))

    total_S = S + pad_size
    num_chunks = total_S // chunk_size
    C = chunk_size

    # Process per v-head (g and beta are per-v_head)
    state = initial_state.copy().astype(np.float32)
    output = np.zeros((B, total_S, V_HEADS, V_DIM), dtype=np.float32)

    for b in range(B):
        for h in range(V_HEADS):
            qk_h = h // group_size

            # Extract per-head sequences
            q_seq = q[b, :, qk_h, :]    # [total_S, K_DIM]
            k_seq = k[b, :, qk_h, :]    # [total_S, K_DIM]
            v_seq = v[b, :, h, :]        # [total_S, V_DIM]
            g_seq = g[b, :, h]           # [total_S]
            beta_seq = beta[b, :, h]     # [total_S]

            # Reshape to chunks
            q_chunks = q_seq.reshape(num_chunks, C, K_DIM)
            k_chunks = k_seq.reshape(num_chunks, C, K_DIM)
            v_chunks = v_seq.reshape(num_chunks, C, V_DIM)
            g_chunks = g_seq.reshape(num_chunks, C)
            beta_chunks = beta_seq.reshape(num_chunks, C)

            # --- Pre-chunk computation (parallelizable across chunks) ---

            # g_cumsum within each chunk
            g_cumsum = np.cumsum(g_chunks, axis=-1)  # [num_chunks, C]

            # decay_mask[c, i, j] = exp(g_cumsum[c,i] - g_cumsum[c,j]) for i>=j
            # Shape: [num_chunks, C, C]
            decay_mask = np.zeros((num_chunks, C, C), dtype=np.float32)
            for c in range(num_chunks):
                for i in range(C):
                    for j in range(i + 1):
                        decay_mask[c, i, j] = np.exp(g_cumsum[c, i] - g_cumsum[c, j])

            # v_beta and k_beta
            v_beta = v_chunks * beta_chunks[:, :, np.newaxis]  # [num_chunks, C, V_DIM]
            k_beta = k_chunks * beta_chunks[:, :, np.newaxis]  # [num_chunks, C, K_DIM]

            # WY fixup: forward substitution to compute v_resolved and k_cumdecay
            # attn_wy = -((k_beta @ key.T) * decay_mask), masked upper-triangular to 0
            # Then forward substitution, then +I
            for c in range(num_chunks):
                # Raw attention coefficient matrix
                # k_beta[c] @ k[c].T → [C, C]
                raw_attn = -(k_beta[c] @ k_chunks[c].T) * decay_mask[c]
                # Zero out upper triangle (keep lower strict triangle)
                raw_attn = np.tril(raw_attn, k=-1)

                # Forward substitution (WY fixup)
                for i in range(1, C):
                    row = raw_attn[i, :i].copy()
                    sub = raw_attn[:i, :i].copy()
                    raw_attn[i, :i] = row + np.sum(row[:, np.newaxis] * sub, axis=0)

                # Add identity
                attn_wy = raw_attn + np.eye(C, dtype=np.float32)

                # Resolved values
                v_resolved = attn_wy @ v_beta[c]  # [C, V_DIM]
                k_cumdecay = attn_wy @ (k_beta[c] * np.exp(g_cumsum[c])[:, np.newaxis])  # [C, K_DIM]

                # --- Per-chunk computation (serial across chunks) ---

                # Intra-chunk attention: QK^T * decay_mask
                qk_attn = (q_chunks[c] @ k_chunks[c].T) * decay_mask[c]  # [C, C]

                # State contribution correction
                v_prime = k_cumdecay @ state[b, h]  # [C, V_DIM]
                v_new = v_resolved - v_prime  # [C, V_DIM]

                # Inter-chunk: Q @ state (with per-token decay from chunk start)
                # q[i] * exp(g_cumsum[i]) @ state
                q_decayed = q_chunks[c] * np.exp(g_cumsum[c])[:, np.newaxis]  # [C, K_DIM]
                attn_inter = q_decayed @ state[b, h]  # [C, V_DIM]

                # Combine
                chunk_output = attn_inter + qk_attn @ v_new  # [C, V_DIM]
                output[b, c * C:(c + 1) * C, h, :] = chunk_output

                # Update state for next chunk
                chunk_total_decay = np.exp(g_cumsum[c, -1])  # scalar: total decay in chunk
                # k_i * exp(g_cumsum[-1] - g_cumsum[i]) for state accumulation
                k_state_decay = k_chunks[c] * np.exp(g_cumsum[c, -1] - g_cumsum[c])[:, np.newaxis]  # [C, K_DIM]
                state[b, h] = state[b, h] * chunk_total_decay + k_state_decay.T @ v_new  # [K_DIM, V_DIM]

    # Trim padding
    output = output[:, :S, :, :]
    return output, state


def test_chunk_vs_recurrent(
    B: int = 1,
    S: int = 128,
    Q_HEADS: int = 16,
    V_HEADS: int = 32,
    K_DIM: int = 128,
    V_DIM: int = 128,
    chunk_size: int = 64,
    seed: int = 42,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """Test that chunk-wise and recurrent produce the same results."""
    np.random.seed(seed)

    # Generate random inputs with realistic ranges
    q = np.random.randn(B, S, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    k = np.random.randn(B, S, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    v = np.random.randn(B, S, V_HEADS, V_DIM).astype(np.float32) * 0.1
    # g should be small negative values (gate decay)
    g = np.random.randn(B, S, V_HEADS).astype(np.float32) * 0.1 - 0.5
    beta = np.random.rand(B, S, V_HEADS).astype(np.float32) * 0.5 + 0.1
    initial_state = np.random.randn(B, V_HEADS, K_DIM, V_DIM).astype(np.float32) * 0.01

    # Run both implementations
    out_rec, state_rec = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state)
    out_chunk, state_chunk = chunk_gated_delta_rule(q, k, v, g, beta, initial_state, chunk_size)

    # Compare outputs
    out_match = np.allclose(out_rec, out_chunk, rtol=rtol, atol=atol)
    state_match = np.allclose(state_rec, state_chunk, rtol=rtol, atol=atol)

    if not out_match:
        max_diff = np.max(np.abs(out_rec - out_chunk))
        mean_diff = np.mean(np.abs(out_rec - out_chunk))
        print(f"  Output MISMATCH: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    if not state_match:
        max_diff = np.max(np.abs(state_rec - state_chunk))
        mean_diff = np.mean(np.abs(state_rec - state_chunk))
        print(f"  State MISMATCH: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

    return out_match and state_match


def test_prefill_then_decode(
    B: int = 1,
    prefill_len: int = 128,
    decode_steps: int = 5,
    Q_HEADS: int = 16,
    V_HEADS: int = 32,
    K_DIM: int = 128,
    V_DIM: int = 128,
    chunk_size: int = 64,
    seed: int = 123,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """
    Test prefill (chunk) → decode (recurrent) continuity.
    The chunk kernel's output_state must be correct for decode to work.
    """
    np.random.seed(seed)

    # Generate prefill inputs
    q_pf = np.random.randn(B, prefill_len, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    k_pf = np.random.randn(B, prefill_len, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    v_pf = np.random.randn(B, prefill_len, V_HEADS, V_DIM).astype(np.float32) * 0.1
    g_pf = np.random.randn(B, prefill_len, V_HEADS).astype(np.float32) * 0.1 - 0.5
    beta_pf = np.random.rand(B, prefill_len, V_HEADS).astype(np.float32) * 0.5 + 0.1
    init_state = np.zeros((B, V_HEADS, K_DIM, V_DIM), dtype=np.float32)

    # Generate decode inputs
    q_dec = np.random.randn(B, decode_steps, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    k_dec = np.random.randn(B, decode_steps, Q_HEADS, K_DIM).astype(np.float32) * 0.1
    v_dec = np.random.randn(B, decode_steps, V_HEADS, V_DIM).astype(np.float32) * 0.1
    g_dec = np.random.randn(B, decode_steps, V_HEADS).astype(np.float32) * 0.1 - 0.5
    beta_dec = np.random.rand(B, decode_steps, V_HEADS).astype(np.float32) * 0.5 + 0.1

    # Path A: Full recurrent (ground truth)
    full_q = np.concatenate([q_pf, q_dec], axis=1)
    full_k = np.concatenate([k_pf, k_dec], axis=1)
    full_v = np.concatenate([v_pf, v_dec], axis=1)
    full_g = np.concatenate([g_pf, g_dec], axis=1)
    full_beta = np.concatenate([beta_pf, beta_dec], axis=1)
    out_full, _ = recurrent_gated_delta_rule(full_q, full_k, full_v, full_g, full_beta, init_state)

    # Path B: Chunk prefill → recurrent decode
    _, state_after_prefill = chunk_gated_delta_rule(q_pf, k_pf, v_pf, g_pf, beta_pf, init_state, chunk_size)
    out_decode, _ = recurrent_gated_delta_rule(q_dec, k_dec, v_dec, g_dec, beta_dec, state_after_prefill)

    # Compare decode outputs
    out_full_decode = out_full[:, prefill_len:, :, :]
    match = np.allclose(out_full_decode, out_decode, rtol=rtol, atol=atol)

    if not match:
        max_diff = np.max(np.abs(out_full_decode - out_decode))
        mean_diff = np.mean(np.abs(out_full_decode - out_decode))
        print(f"  Prefill→Decode MISMATCH: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

    return match


if __name__ == "__main__":
    print("=" * 70)
    print("Chunk-wise GatedDeltaNet (LinearAttention) Numpy Reference Tests")
    print("=" * 70)

    test_cases = [
        # (description, seq_len, chunk_size)
        ("seq_len=64 (1 chunk)", 64, 64),
        ("seq_len=128 (2 chunks)", 128, 64),
        ("seq_len=786 (13 chunks, tail=26)", 786, 64),
        ("seq_len=32 (partial chunk < C)", 32, 64),
        ("seq_len=65 (1 full + 1 partial)", 65, 64),
        ("seq_len=1 (single token)", 1, 64),
        ("chunk_size=32, seq=128", 128, 32),
    ]

    all_passed = True
    for desc, seq_len, chunk_size in test_cases:
        print(f"\nTest: {desc}")
        passed = test_chunk_vs_recurrent(S=seq_len, chunk_size=chunk_size)
        status = "PASSED" if passed else "FAILED"
        print(f"  Result: {status}")
        all_passed = all_passed and passed

    print(f"\nTest: prefill(128)→decode(5) continuity")
    passed = test_prefill_then_decode()
    status = "PASSED" if passed else "FAILED"
    print(f"  Result: {status}")
    all_passed = all_passed and passed

    print(f"\nTest: prefill(786)→decode(10) continuity (chunk=64)")
    passed = test_prefill_then_decode(prefill_len=786, decode_steps=10)
    status = "PASSED" if passed else "FAILED"
    print(f"  Result: {status}")
    all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
