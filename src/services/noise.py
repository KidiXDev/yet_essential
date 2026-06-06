from __future__ import annotations

import torch


def slerp_noise(noise_a: torch.Tensor, noise_b: torch.Tensor, strength: float) -> torch.Tensor:
    t = float(min(max(strength, 0.0), 1.0))
    if t <= 0.0:
        return noise_a
    if t >= 1.0:
        return noise_b

    a = noise_a.reshape(noise_a.shape[0], -1)
    b = noise_b.reshape(noise_b.shape[0], -1)

    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    dot = torch.sum(a_norm * b_norm, dim=1, keepdim=True).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    near_linear = torch.abs(sin_omega) < 1e-6
    interp = (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b
    lerp = (1.0 - t) * a + t * b
    out = torch.where(near_linear, lerp, interp)
    return out.reshape_as(noise_a)
