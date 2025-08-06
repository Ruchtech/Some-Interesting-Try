import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device="cpu"):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim)
        )

    def forward(self, x, position_ids, seq_len=None):
        if seq_len is not None:
            print("Warning: seq_len is deprecated.")
        # Force float32 since bfloat16 loses precision on long contexts
        # see https://github.com/huggingface/transformers/pull/29285
        # x: [bs, num_heads, seq_len, head_dim]
        # position_ids: [bs, seq_len]
        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"

        inv_freq_expanded = self.inv_freq[None, :, None]  # [1, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [bs, 1, seq_len]

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [bs, seq_len, dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [bs, seq_len, dim]
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to the query and key tensors...."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def test_rotary_embedding():
    bs = 2                   # batch size
    seq_len = 4              # sequence length
    num_heads = 3            # number of attention heads
    head_dim = 8             # dimension per head (must be even)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(bs, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(bs, num_heads, seq_len, head_dim, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)  # [bs, seq_len]

    rotary_emb = RotaryEmbedding(head_dim, device=device).to(device)
    cos, sin = rotary_emb(q, position_ids)

    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"q shape:        {q.shape}")
    print(f"cos shape:      {cos.shape}")
    print(f"q_embed shape:  {q_embed.shape}")
    print(f"k_embed shape:  {k_embed.shape}")

    assert q.shape == q_embed.shape
    assert k.shape == k_embed.shape
    print("✅ Passed!")

test_rotary_embedding()
