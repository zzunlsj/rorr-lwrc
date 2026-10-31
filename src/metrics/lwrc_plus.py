"""LwRC → LwRC+ 정규화 (정본 v1.0 §5).

LwRC+ = 100 × rate ÷ denom[시즌, 포지션]
- 기준 집단: 2026 전체 풀 × 포지션 (확정 2026-10-27)
- 음수 허용, 바닥 처리 없음 (확정 2026-10-27)
- 분모: 포지션별 rate를 ±3σ 윈저라이즈한 평균, prior가 있으면 w=(n·실측+k·prior)/(n+k)
"""

import numpy as np
import polars as pl


def winsorized_mean(x: np.ndarray, sigma: float) -> float:
    if x.size < 2:
        return float(x.mean()) if x.size else float("nan")
    mu, sd = x.mean(), x.std(ddof=1)
    return float(np.clip(x, mu - sigma * sd, mu + sigma * sd).mean())


def denominators(players: pl.DataFrame, cfg: dict, prior: dict[str, float] | None = None) -> pl.DataFrame:
    norm = cfg["normalize"]
    rows = []
    for (role,), grp in players.group_by(["role"], maintain_order=True):
        x = grp["rate"].to_numpy()
        measured = winsorized_mean(x, norm["winsorize_sigma"])
        n, k = x.size, norm["prior_k"]
        denom = (n * measured + k * prior[role]) / (n + k) if prior and role in prior else measured
        rows.append({
            "role": role, "n": n, "mean_raw": float(x.mean()), "std": float(x.std(ddof=1)) if n > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "mean_winsorized": measured,
            "prior": prior.get(role) if prior else None, "denom": denom,
        })
    return pl.DataFrame(rows).sort("role")


def apply(players: pl.DataFrame, denom: pl.DataFrame) -> pl.DataFrame:
    return (players.join(denom.select("role", "denom"), on="role", how="left")
            .with_columns((100 * pl.col("rate") / pl.col("denom")).alias("lwrc_plus")))
