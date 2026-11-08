from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GateFMModelConfig:
    n_snps: int
    latent_rank: int
    gate_temperature: float = 0.7
    gate_activation: str = "hard_sigmoid"
    gate_init: float = 0.05
    embedding_init_std: float = 0.1
    gamma_tri: float = 0.05


class ThirdOrderGateFM(nn.Module):
    """Third-order genotype-specific GateFM reference model.

    Each SNP has three genotype embeddings. A SNP-level clipped linear
    hard-sigmoid gate is shared by its three genotype states. Only the
    third-order FM branch is included.
    """

    def __init__(self, config: GateFMModelConfig):
        super().__init__()
        if config.n_snps < 3:
            raise ValueError("ThirdOrderGateFM requires at least three SNPs.")
        if config.latent_rank <= 0:
            raise ValueError("latent_rank must be positive.")
        if config.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if config.gate_activation not in {"sigmoid", "hard_sigmoid"}:
            raise ValueError(
                "gate_activation must be 'sigmoid' or 'hard_sigmoid'."
            )

        self.config = config
        self.n_snps = int(config.n_snps)
        self.latent_rank = int(config.latent_rank)
        self.gate_temperature = float(config.gate_temperature)
        self.gate_activation = str(config.gate_activation)
        self.gamma_tri = float(config.gamma_tri)

        self.E_tri = nn.Parameter(
            torch.empty(self.n_snps, 3, self.latent_rank)
        )
        nn.init.normal_(
            self.E_tri,
            mean=0.0,
            std=float(config.embedding_init_std),
        )

        self.gate_logit = nn.Parameter(
            torch.full((self.n_snps,), float(config.gate_init))
            + 0.01 * torch.randn(self.n_snps)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "snp_offsets",
            torch.arange(self.n_snps, dtype=torch.long) * 3,
            persistent=False,
        )

    def gates(self) -> torch.Tensor:
        """Return the configured SNP-level gate."""
        scaled = self.gate_logit / self.gate_temperature
        if self.gate_activation == "sigmoid":
            return torch.sigmoid(scaled)
        return torch.clamp((scaled + 1.0) / 2.0, min=0.0, max=1.0)

    def selected_embeddings(
        self,
        genotypes: torch.Tensor,
        *,
        apply_gate: bool = True,
    ) -> torch.Tensor:
        if genotypes.dtype != torch.long:
            raise TypeError("genotypes must be torch.long.")
        if genotypes.ndim != 2 or genotypes.shape[1] != self.n_snps:
            raise ValueError(
                f"Expected genotypes [batch,{self.n_snps}], got {tuple(genotypes.shape)}."
            )

        flat_index = genotypes + self.snp_offsets.view(1, -1)
        embedding_table = self.E_tri.reshape(
            self.n_snps * 3,
            self.latent_rank,
        )
        selected = F.embedding(flat_index, embedding_table)
        if apply_gate:
            selected = selected * self.gates().view(1, -1, 1)
        return selected

    def third_order_term(self, genotypes: torch.Tensor) -> torch.Tensor:
        """Newton-Girard/FM trick for all u<v<w third-order interactions."""
        u = self.selected_embeddings(genotypes, apply_gate=True)
        s1 = u.sum(dim=1)
        u2 = u.square()
        s2 = u2.sum(dim=1)
        s3 = (u2 * u).sum(dim=1)
        per_rank = (s1.pow(3) - 3.0 * s1 * s2 + 2.0 * s3) / 6.0
        return per_rank.sum(dim=1)

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        return self.bias + self.gamma_tri * self.third_order_term(genotypes)

    @torch.no_grad()
    def feature_artifacts(self) -> Dict[str, np.ndarray]:
        """Return all values needed to reproduce ranking and Stage-2 patterns.

        SNP priority score:
          rho[j,k] = gate[j] * ||E_tri[j,k,:]||_2
          q[j] = max_k rho[j,k]
        """
        gate = self.gates()
        embedding_norm = torch.linalg.vector_norm(self.E_tri, ord=2, dim=2)
        rho = gate[:, None] * embedding_norm
        q_score, representative_genotype = torch.max(rho, dim=1)
        order = torch.argsort(q_score, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(
            1,
            self.n_snps + 1,
            device=order.device,
        )

        return {
            "gate_logit": self.gate_logit.detach().cpu().numpy().astype(np.float32),
            "gate": gate.detach().cpu().numpy().astype(np.float32),
            "embedding": self.E_tri.detach().cpu().numpy().astype(np.float32),
            "embedding_norm": embedding_norm.detach().cpu().numpy().astype(np.float32),
            "rho": rho.detach().cpu().numpy().astype(np.float32),
            "q_score": q_score.detach().cpu().numpy().astype(np.float32),
            "representative_genotype": (
                representative_genotype.detach().cpu().numpy().astype(np.int8)
            ),
            "q_rank": rank.detach().cpu().numpy().astype(np.int32),
        }

    @torch.no_grad()
    def direct_third_order_term(self, genotypes: torch.Tensor) -> torch.Tensor:
        """Slow reference implementation used only by verification tests."""
        u = self.selected_embeddings(genotypes, apply_gate=True)
        out = torch.zeros(
            genotypes.shape[0],
            device=genotypes.device,
            dtype=u.dtype,
        )
        for a in range(self.n_snps - 2):
            for b in range(a + 1, self.n_snps - 1):
                for c in range(b + 1, self.n_snps):
                    out = out + (u[:, a, :] * u[:, b, :] * u[:, c, :]).sum(dim=1)
        return out


def model_config_from_dict(payload: dict, n_snps: int) -> GateFMModelConfig:
    model_cfg = payload["model"]
    return GateFMModelConfig(
        n_snps=int(n_snps),
        latent_rank=int(model_cfg["latent_rank"]),
        gate_temperature=float(model_cfg["gate_temperature"]),
        gate_activation=str(model_cfg.get("gate_activation", "hard_sigmoid")),
        gate_init=float(model_cfg.get("gate_init", 0.05)),
        embedding_init_std=float(model_cfg.get("embedding_init_std", 0.1)),
        gamma_tri=float(model_cfg.get("gamma_tri", 0.05)),
    )
