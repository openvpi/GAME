import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def compute_inv_freq(dim: int, theta: float = 10000.0):
    """pre-compute inv_freq, dim is fixed"""
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))


def compute_freqs_cis_dynamic(x: torch.Tensor, inv_freq: torch.Tensor):
    """ONNX兼容：动态计算cos/sin，序列长度从xa tensor shape获取"""
    # 用tensor操作获取seq_len，让ONNX能动态trace
    seq_len = x.shape[-2]
    # 生成位置索引 [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    # outer product: [seq_len, dim//2]
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


# 旋转位置编码计算 (ONNX兼容版本)
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
) :
    """ONNX兼容：手动实现复数乘法 (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 分离实部和虚部
    xq_r, xq_i = xq_[..., 0], xq_[..., 1]
    xk_r, xk_i = xk_[..., 0], xk_[..., 1]

    # 复数乘法: (xq_r + xq_i*j) * (cos + sin*j) = (xq_r*cos - xq_i*sin) + (xq_r*sin + xq_i*cos)*j
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 合并实部和虚部
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPosEmb(nn.Module):
    def __init__(self, dim, max_len=5000, theta=10000.0, use_cache=True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.use_cache = use_cache
        # inv_freq是固定的，可以预计算
        self.register_buffer('inv_freq', compute_inv_freq(dim, theta))
        # 缓存模式下预计算pe
        if use_cache:
            pe_cos, pe_sin = compute_freqs_cis_dynamic(
                torch.zeros(1, max_len, dim), self.inv_freq)
            self.register_buffer('pe_cos', pe_cos[None, :, :])
            self.register_buffer('pe_sin', pe_sin[None, :, :])

    def extend_pe(self, x):
        """Reset the positional encodings (only for use_cache=True mode)."""
        if self.pe_cos.size(1) >= x.size(1):
            return
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        self.pe_cos = pe_cos[None, :, :].to(device=x.device)
        self.pe_sin = pe_sin[None, :, :].to(device=x.device)

    def forward(self, q, k):
        if self.use_cache:
            self.extend_pe(q)
            pe_cos = self.pe_cos[:, :q.size(1)]
            pe_sin = self.pe_sin[:, :q.size(1)]
        else:
            # ONNX模式：完全动态计算
            pe_cos, pe_sin = compute_freqs_cis_dynamic(q, self.inv_freq)
        return apply_rotary_emb(q, k, pe_cos, pe_sin)



def single_apply_rotary_emb(
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
) :
    """ONNX兼容：手动实现复数乘法"""
    x_ = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()

    # 分离实部和虚部
    x_r, x_i = x_[..., 0], x_[..., 1]

    # 复数乘法: (x_r + x_i*j) * (cos + sin*j) = (x_r*cos - x_i*sin) + (x_r*sin + x_i*cos)*j
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # 合并实部和虚部
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)

    return x_out.type_as(x)

class SingleRoPosEmb(nn.Module):
    def __init__(self, dim:int, max_len=5000, theta=10000.0, use_cache=True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.use_cache = use_cache
        # inv_freq是固定的，可以预计算
        self.register_buffer('inv_freq', compute_inv_freq(dim, theta))
        # 缓存模式下预计算pe
        if use_cache:
            pe_cos, pe_sin = compute_freqs_cis_dynamic(
                torch.zeros(1, max_len, dim), self.inv_freq)
            self.register_buffer('pe_cos', pe_cos[None, :, :])
            self.register_buffer('pe_sin', pe_sin[None, :, :])

    def extend_pe(self, x):
        """Reset the positional encodings (only for use_cache=True mode)."""
        if self.pe_cos.size(1) >= x.size(-2):
            return
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        self.pe_cos = pe_cos[None, :, :].to(device=x.device)
        self.pe_sin = pe_sin[None, :, :].to(device=x.device)

    def get_pe_dynamic(self, x):
        """ONNX模式：完全动态计算"""
        ndim = x.ndim
        seq_len = x.shape[-2]
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        pe_cos = pe_cos.view(*((1,)*(ndim-2)), seq_len, self.dim//2)
        pe_sin = pe_sin.view(*((1,)*(ndim-2)), seq_len, self.dim//2)
        return pe_cos, pe_sin

    def get_pe_cached(self, x):
        """Cache模式：从缓存切片"""
        ndim = x.ndim
        seq_len = x.size(-2)
        pe_cos = self.pe_cos[:, :seq_len]
        pe_sin = self.pe_sin[:, :seq_len]
        pe_cos = pe_cos.view(*((1,)*(ndim-2)), seq_len, self.dim//2)
        pe_sin = pe_sin.view(*((1,)*(ndim-2)), seq_len, self.dim//2)
        return pe_cos, pe_sin

    def forward(self, x):
        if self.use_cache:
            self.extend_pe(x)
            pe_cos, pe_sin = self.get_pe_cached(x)
        else:
            # ONNX模式：完全动态计算
            pe_cos, pe_sin = self.get_pe_dynamic(x)
        return single_apply_rotary_emb(x, pe_cos, pe_sin)
if __name__ == '__main__':
    import numpy as np

    print("=" * 50)
    print("测试1: 对比原版复数实现和ONNX实现的数值一致性")
    print("=" * 50)

    # 原版复数实现（用于对比）
    def precompute_freqs_cis_original(dim: int, seq_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def single_apply_rotary_emb_original(x, freqs_cis):
        x_ = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
        x_ = torch.view_as_complex(x_)
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)

    # 测试数据
    torch.manual_seed(42)
    dim = 40
    seq_len = 5
    x_test = torch.randn(1, 2, 8, seq_len, dim)

    # 原版输出
    freqs_cis_orig = precompute_freqs_cis_original(dim, seq_len)
    freqs_cis_orig = freqs_cis_orig[None, :, :].view(1, 1, 1, seq_len, dim // 2)
    out_original = single_apply_rotary_emb_original(x_test, freqs_cis_orig)

    # 新版输出 (use_cache=True)
    sro_cached = SingleRoPosEmb(dim, use_cache=True)
    out_new_cached = sro_cached(x_test.clone())

    # 新版输出 (use_cache=False, ONNX模式)
    sro_no_cache = SingleRoPosEmb(dim, use_cache=False)
    out_new_no_cache = sro_no_cache(x_test.clone())

    # 对比
    diff_cached = (out_original - out_new_cached).abs().max().item()
    diff_no_cache = (out_original - out_new_no_cache).abs().max().item()
    print(f"原版 vs 新版(use_cache=True) 最大差异: {diff_cached:.2e}")
    print(f"原版 vs 新版(use_cache=False) 最大差异: {diff_no_cache:.2e}")
    assert diff_cached < 1e-5, "缓存模式数值不一致!"
    assert diff_no_cache < 1e-5, "无缓存模式数值不一致!"
    print("[PASS] numerical consistency test passed!")

    print("\n" + "=" * 50)
    print("测试2: 导出ONNX并验证")
    print("=" * 50)

    # 用无缓存模式导出ONNX
    model_onnx = SingleRoPosEmb(dim, use_cache=False)
    model_onnx.eval()

    dummy_input = torch.randn(1, 2, 8, seq_len, dim)
    onnx_path = "rope_test.onnx"

    torch.onnx.export(
        model_onnx,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 3: "seq_len"}, "output": {0: "batch", 3: "seq_len"}},
        opset_version=14,
    )
    print(f"[PASS] ONNX export success: {onnx_path}")

    # 用ONNX Runtime验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {"input": dummy_input.numpy()})[0]
        torch_out = model_onnx(dummy_input).detach().numpy()
        diff_onnx = np.abs(ort_out - torch_out).max()
        print(f"PyTorch vs ONNX Runtime (seq_len={seq_len}): {diff_onnx:.2e}")
        assert diff_onnx < 1e-5, "ONNX vs PyTorch mismatch!"
        
        # Test dynamic seq_len: use different lengths
        for test_seq in [3, 10, 20,100,5000,7000]:
            test_input = torch.randn(1, 2, 8, test_seq, dim)
            ort_out = sess.run(None, {"input": test_input.numpy()})[0]
            torch_out = model_onnx(test_input).detach().numpy()
            diff = np.abs(ort_out - torch_out).max()
            print(f"PyTorch vs ONNX Runtime (seq_len={test_seq}): {diff:.2e}")
            assert diff < 1e-5, f"ONNX dynamic seq_len={test_seq} failed!"
        
        print("[PASS] ONNX Runtime verification passed (dynamic seq_len works!)")
    except ImportError:
        print("(skip ONNX Runtime test, onnxruntime not installed)")

    print("\n" + "=" * 50)
    print("测试3: RoPosEmb双输入版本")
    print("=" * 50)
    q = torch.randn(1, seq_len, dim)
    k = torch.randn(1, seq_len, dim)
    rope = RoPosEmb(dim, use_cache=True)
    q_out, k_out = rope(q, k)
    print(f"q_out.shape: {q_out.shape}, k_out.shape: {k_out.shape}")
    print("[PASS] RoPosEmb test passed!")

    print("\n=== All tests passed! ===")

