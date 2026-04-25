"""Gradient reversal layer used in adversarial domain learning."""

from __future__ import annotations

import torch


class GradientReverseFunction(torch.autograd.Function):
    """Identity in forward, negative-scaled gradient in backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReverseLayer(torch.nn.Module):
    """A torch.nn.Module wrapper around GradientReverseFunction."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor, lambda_: float | None = None) -> torch.Tensor:
        scale = self.lambda_ if lambda_ is None else lambda_
        return GradientReverseFunction.apply(x, float(scale))
