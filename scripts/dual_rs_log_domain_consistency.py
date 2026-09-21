"""Explicit Dual RS numerical compatibility variant; author checkout unchanged.

Same differentiable mean-probability KL and clamped entropy in real arithmetic.
Computes log(mean softmax) via logsumexp, avoiding the derivative of log(0)
in kl_div's probability-target path. No detach, probability floor, LR change,
mixed-precision switch or alternative statistical/training objective.
"""
import math
import torch
import torch.nn.functional as F


def consistency_loss(logits, lbd, eta=0.5, loss="default"):
    if loss != "default" or not logits:
        raise ValueError("compatibility scope is the author's default consistency loss")
    if any(x.shape != logits[0].shape or x.dtype != logits[0].dtype or
           not torch.isfinite(x).all() for x in logits):
        raise ValueError("incompatible/nonfinite logit batches")
    log_probs = torch.stack([F.log_softmax(x, dim=1) for x in logits])
    log_avg = torch.logsumexp(log_probs, dim=0) - math.log(len(logits))
    avg = log_avg.exp()
    kl = (avg.unsqueeze(0) * (log_avg.unsqueeze(0) - log_probs)).sum(-1).mean(0)
    ent = -(avg * log_avg.clamp(min=math.log(1e-20))).sum(-1)
    value = lbd * kl + eta * ent
    return value.clamp(min=1e-10), kl.mean().item(), ent.mean().item()
