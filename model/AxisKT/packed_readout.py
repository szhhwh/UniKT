"""Fused packed-space readout and event construction for inference.

One Triton kernel per (batch, position) replaces the aten ``scatter_reduce``
chain of the training path; inference-only, so gradient semantics are
untouched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _readout_event_kernel(
    state_ptr,
    skemb_ptr,
    skchg_ptr,
    question_ptr,
    question_ids_ptr,
    qdiff_ptr,
    valid_ptr,
    inv_ptr,
    out_ptr,
    event_ptr,
    wl_ptr,
    ws_ptr,
    S,
    K,
    P,
    H,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (batch row, position): fused segment-softmax readout
    plus pooled event-embedding construction."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    h_offs = tl.arange(0, BLOCK_H)
    k_mask = k_offs < K
    h_mask = h_offs < H

    # An invalid slot's packed index may fall outside the trimmed packed
    # width; clamp before any load (the load mask keeps its values unused).
    flat = pid_b * S * K + pid_s * K + k_offs
    kv = tl.load(valid_ptr + flat, mask=k_mask, other=0) != 0
    j = tl.load(inv_ptr + flat, mask=k_mask, other=0).to(tl.int32)
    j = tl.where(kv & (j < P), j, 0)

    offs = pid_b * P * H + j[:, None] * H + h_offs[None, :]
    m2 = (k_mask & kv)[:, None] & h_mask[None, :]
    state = tl.load(state_ptr + offs, mask=m2, other=0.0)
    sk = tl.load(skemb_ptr + offs, mask=m2, other=0.0)
    ch = tl.load(skchg_ptr + offs, mask=m2, other=0.0)

    wl = tl.load(wl_ptr + h_offs, mask=h_mask, other=0.0)
    ws = tl.load(ws_ptr + h_offs, mask=h_mask, other=0.0)
    q_off = (pid_b * S + pid_s) * H + h_offs
    question = tl.load(question_ptr + q_off, mask=h_mask, other=0.0)
    score = tl.sum(state * wl[None, :], 1) + tl.sum(sk * ws[None, :], 1)
    score = tl.where(kv, score, float("-inf"))

    m = tl.max(score, 0)
    e = tl.exp(score - m)
    e = tl.where(kv, e, 0.0)
    z = tl.sum(e, 0)
    w = e / tl.where(z > 0, z, 1.0)

    out = tl.sum(w[:, None] * state, 0)
    count = tl.sum(kv.to(tl.float32), 0)
    denom = tl.maximum(count, 1.0)
    pooled_sk = tl.sum(tl.where(kv[:, None], sk, 0.0), 0) / denom
    pooled_ch = tl.sum(tl.where(kv[:, None], ch, 0.0), 0) / denom

    qid = tl.load(question_ids_ptr + pid_b * S + pid_s).to(tl.int32)
    qdiff = tl.load(qdiff_ptr + qid)
    event = question + pooled_sk + qdiff * pooled_ch
    tl.store(out_ptr + q_off, out, mask=h_mask)
    tl.store(event_ptr + q_off, event, mask=h_mask)


def fused_readout_event(
    packed_state: torch.Tensor,
    packed_skill_embedding: torch.Tensor,
    packed_skill_change: torch.Tensor,
    question_vector: torch.Tensor,
    question_ids: torch.Tensor,
    question_diff_weight: torch.Tensor,
    slot_valid: torch.Tensor,
    kc_inverse: torch.Tensor,
    max_skills: int,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run the fused readout and event-construction kernel.

    Args:
        packed_state: Scanned per-occurrence states, [B, P, H].
        packed_skill_embedding: Skill embeddings at the packed slots, [B, P, H].
        packed_skill_change: Change embeddings at the packed slots, [B, P, H].
        question_vector: Per-position question embedding, [B, S, H].
        question_ids: Per-position question ids, [B, S].
        question_diff_weight: Question-difficulty embedding table, [Q, 1].
        slot_valid: Occurrence validity over the flat [B, S*K] slot domain.
        kc_inverse: Inverse of the packing permutation over the same flat
            domain: flat slot -> packed index, [B, S*K].
        max_skills: K, the flattened KC width per position.
        weight: Row vector of ``local_readout``, [1, 2H]: first H entries
            multiply the state and the final H multiply the skill embedding.

    Returns:
        (readout [B, S, H], event_embedding [B, S, H]).
    """
    batch, packed_len, hidden = packed_state.shape
    seq_len = question_vector.shape[1]
    device = packed_state.device

    # One pooled allocation for the two [B, S, H] outputs; each slice stays
    # contiguous.
    n_out = batch * seq_len * hidden
    pool = torch.empty(2 * n_out, device=device, dtype=torch.float32)
    readout = pool[:n_out].view(batch, seq_len, hidden)
    event = pool[n_out:].view(batch, seq_len, hidden)
    major, _ = torch.cuda.get_device_capability(device)
    num_warps = 4 if major >= 9 else 1

    _readout_event_kernel[(batch, seq_len)](
        packed_state,
        packed_skill_embedding,
        packed_skill_change,
        question_vector,
        question_ids,
        question_diff_weight,
        slot_valid,
        kc_inverse,
        readout,
        event,
        weight[:, :hidden].reshape(-1),
        weight[:, hidden:].reshape(-1),
        seq_len,
        max_skills,
        packed_len,
        hidden,
        BLOCK_K=triton.next_power_of_2(max_skills),
        BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=num_warps,
    )
    return readout, event


__all__ = ["fused_readout_event"]
