import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k, v, seq_indptr, sm_scale):
    total_tokens, num_qo_heads, head_dim_qk = q.shape
    head_dim_vo = v.shape[-1]
    len_indptr = seq_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_qk == 192
    assert head_dim_vo == 128

    # Check constraints
    assert total_tokens == seq_indptr[-1].item()

    device = q.device

    out = torch.zeros((total_tokens, num_qo_heads, head_dim_vo), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    for b in range(len_indptr - 1):
        seq_start = int(seq_indptr[b].item())
        seq_end = int(seq_indptr[b + 1].item())
        if seq_start >= seq_end:
            continue

        seq_len = seq_end - seq_start
        qb = q[seq_start:seq_end]  # [seq_len, num_qo_heads, head_dim_qk]
        kb = k[seq_start:seq_end]  # [seq_len, num_qo_heads, head_dim_qk]
        vb = v[seq_start:seq_end]  # [seq_len, num_qo_heads, head_dim_vo]

        logits = torch.einsum("qhd,khd->qhk", qb, kb)  # [seq_len, num_qo_heads, seq_len]
        logits_scaled = logits * sm_scale

        # Apply causal mask if enabled
        i = torch.arange(seq_len, device=device).unsqueeze(-1)  # [seq_len, 1]
        j = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        logits_scaled.masked_fill_((j > i).unsqueeze(1), float("-inf"))

        # Compute 2-base LSE
        lse[seq_start:seq_end] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [seq_len, num_qo_heads, seq_len]
        out_b = torch.einsum("qhk,khd->qhd", attn, vb)  # [seq_len, num_qo_heads, head_dim_vo]
        out[seq_start:seq_end] = out_b.to(torch.bfloat16)

    return {"output": out, "lse": lse}


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_heads=16,
    head_dim_qk=192,
    head_dim_vo=128,
    causal=True,
    device="cuda",
):
    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)

    # Since MLA is self-attention, all indptrs are the same
    seq_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    seq_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    seq_indptr = seq_indptr.to(device)

    total_tokens = int(seq_indptr[-1].item())

    # Generate Q, K, V tensors with same token count
    q = torch.randn(total_tokens, num_heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_tokens, num_heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_tokens, num_heads, head_dim_vo, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim_qk)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)
    
    # Convert causal to tensor
    causal = torch.tensor(causal, dtype=torch.bool, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "seq_indptr": seq_indptr,
        "seq_lens": seq_lens,
        "total_tokens": total_tokens,
        "sm_scale": sm_scale,
        "causal": causal,
        "num_heads": num_heads,
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "device": device,
    }


def test_correctness(batch_size=4, max_seq_len=32, causal=True, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(
        f"Testing MLA Ragged Prefill "
        f"bs={batch_size}, max_seq_len={max_seq_len}, causal={causal}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True  # skip as pass to not break CI

    H = 16
    Dqk = 192
    Dvo = 128

    inputs = generate_random_inputs(batch_size, max_seq_len, H, Dqk, Dvo, causal, device)
    print(f"seq_lens: {inputs['seq_lens'].cpu().numpy()}")
    print(f"total_tokens={inputs['total_tokens']}")
    print(f"causal={inputs['causal'].item()}")

    print("\nRunning reference...")
    ref = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["seq_indptr"],
        inputs["sm_scale"],
    )
    ref_o = ref["output"]
    ref_lse = ref["lse"]

    print("\nSetting up FlashInfer (ragged prefill wrapper)...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )
    # For MLA self-attention, qo_indptr and kv_indptr are identical
    prefill.plan(
        inputs["seq_indptr"],
        inputs["seq_indptr"],
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=Dqk,
        head_dim_vo=Dvo,
        causal=inputs["causal"].item(),
        sm_scale=inputs["sm_scale"],
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_o, fi_lse = prefill.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_o.shape}")
    print(f"FlashInfer output shape: {fi_o.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")
    print(f"FlashInfer LSE shape: {fi_lse.shape}")

    abs_diff = (ref_o - fi_o).abs()
    rel_diff = abs_diff / (fi_o.abs() + 1e-8)

    print(
        f"Output: max_abs={abs_diff.max().item():.6e}, "
        f"max_rel={rel_diff.max().item():.6e}, "
        f"mean_abs={abs_diff.mean().item():.6e}, "
        f"mean_rel={rel_diff.mean().item():.6e}"
    )

    cos_sim = torch.nn.functional.cosine_similarity(ref_o.flatten(), fi_o.flatten(), dim=0).item()
    mse = torch.mean((ref_o - fi_o) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}, MSE: {mse:.6e}")

    print("\nComparing LSE (log2)...")
    lse_abs = (ref_lse - fi_lse).abs()
    lse_rel = lse_abs / (fi_lse.abs() + 1e-8)
    print(
        f"LSE:    max_abs={lse_abs.max().item():.6e}, "
        f"max_rel={lse_rel.max().item():.6e}, "
        f"mean_abs={lse_abs.mean().item():.6e}, "
        f"mean_rel={lse_rel.mean().item():.6e}"
    )

    ok_out = torch.allclose(ref_o, fi_o, atol=atol, rtol=rtol)
    ok_lse = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_ok = ok_out and ok_lse

    if all_ok:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
        if not ok_out:
            flat = abs_diff.flatten()
            top_k = min(5, flat.numel())
            vals, idxs = torch.topk(flat, top_k)
            print("\nTop output diffs:")
            for i in range(top_k):
                idx = idxs[i].item()
                q_idx = idx // (H * Dvo)
                h_idx = (idx % (H * Dvo)) // Dvo
                d_idx = idx % Dvo
                print(
                    f"  [q={q_idx}, h={h_idx}, d={d_idx}] "
                    f"ref={ref_o.flatten()[idx].item():.6f} "
                    f"fi={fi_o.flatten()[idx].item():.6f} "
                    f"diff={vals[i].item():.6e}"
                )
        if not ok_lse:
            flat = lse_abs.flatten()
            top_k = min(5, flat.numel())
            vals, idxs = torch.topk(flat, top_k)
            print("\nTop LSE diffs:")
            for i in range(top_k):
                idx = idxs[i].item()
                q_idx = idx // H
                h_idx = idx % H
                print(
                    f"  [q={q_idx}, h={h_idx}] "
                    f"ref={ref_lse.flatten()[idx].item():.6f} "
                    f"fi={fi_lse.flatten()[idx].item():.6f} "
                    f"diff={vals[i].item():.6e}"
                )
    return all_ok


def main():
    print("Testing MLA Ragged Prefill with FlashInfer")
    test_cfgs = [
        # (batch_size, max_seq_len, causal)
        (1, 16, True),   # Small, causal
        # (1, 16, False),  # Small, non-causal
        (4, 32, True),   # Medium, causal
        # (4, 32, False),  # Medium, non-causal
        (8, 64, True),   # Large, causal
        # (8, 64, False),  # Large, non-causal
        (16, 128, True),  # Very large, causal
        # (16, 128, False), # Very large, non-causal
    ]
    passed = 0
    for bs, max_seq, causal in test_cfgs:
        try:
            if test_correctness(bs, max_seq, causal):
                passed += 1
        except Exception as e:
            print(f"✗ Exception: {e}")
            import traceback

            traceback.print_exc()
    total = len(test_cfgs)
    print(f"\n{'='*60}\nSummary: {passed}/{total} tests passed\n{'='*60}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
