#!/usr/bin/env python3
"""Muon optimizer (Keller Jordan et al., 2024) for the 2D LoRA matrices.

Muon momentum-averages the gradient, then ORTHOGONALIZES the update via a
Newton-Schulz iteration (approximates U V^T of the update's SVD). The resulting
step has bounded spectral norm regardless of gradient scale -- which is exactly
the property we want after the AdamW run diverged (grad blow-up -> weight
runaway): an orthogonalized update can't blow up the weights.

LoRA's trainable params (lora_A, lora_B) are all 2D, so Muon applies cleanly;
any non-2D param falls back to momentum SGD.
"""
import torch


@torch.no_grad()
def _newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz: approximate the orthogonal factor of G (==U V^T)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom, lr = grp["momentum"], grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                buf = st.get("buf")
                if buf is None:
                    buf = torch.zeros_like(g)
                    st["buf"] = buf
                buf.mul_(mom).add_(g)
                if g.ndim != 2:                       # no LoRA param hits this
                    p.add_(buf, alpha=-lr)
                    continue
                u = g.add(buf, alpha=mom) if grp["nesterov"] else buf
                u = _newtonschulz5(u, grp["ns_steps"])
                # match the update RMS to the matrix aspect ratio (Jordan's scaling)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u, alpha=-lr * scale)
