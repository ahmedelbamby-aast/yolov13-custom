# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(g: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of matrix g via quintic Newton-Schulz."""
    assert len(g.shape) == 2
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T

    for a, b, c in [
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.T
    return x


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    """Compute Muon update with momentum and orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1.0, grad.size(-2) / max(grad.size(-1), 1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """Hybrid optimizer combining Muon and SGD updates."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    lr = group["lr"]

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_sgd"] = torch.zeros_like(p)

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_sgd"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_sgd"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_sgd"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    lr = group["lr"]
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(optim.Optimizer):
    """Muon optimizer for non-distributed usage."""

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
