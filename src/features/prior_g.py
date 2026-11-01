"""비교 기준선 prior G (spec/01 §1.5) — timeline.build_timeline의 engine으로 끼워 같은 분할·귀속을 산출한다.

  G_prior(t, side) = gold[side, t] + R(t) × M(t, side)
  R(t) = r̄ × min(H, 남은 분)        r̄ = 팀 분당 평균 수입 (자체 데이터)
  M    = 1 + w_alive·(생존차/5) + w_dragon·드래곤 스택차 + w_tower·(상대 타워 손실 − 우리 타워 손실) + w_baron·바론(우리)

해석 (기준선 전용, 사양 문구가 열린 곳):
- 타워 = 외곽·2차·억제기 타워 (넥서스 타워 제외). 바론(우리) = 버프 지속 중이면 1.
- G_prior(END) = 종료 골드 (남은 분 0 → R = 0).
- 시간항 = R의 감소(종료 10분 전부터) × 직전 M. 결정적 감쇠 = 부활(생존 +1)과 바론 버프 종료의 R × 가중.
- Risk = 데스 순간 생존 −1의 G 변화 = R(t_d⁻) × (−w_alive/5). 현상금은 상대 자기 골드라 우리 G_prior와 무관.
"""

import numpy as np
import polars as pl

from common.paths import L2
from features.timeline import SIDES, death_risk


def team_income_rate() -> float:
    """r̄ = 팀·게임별 (종료 골드 − 시작 골드) / 분의 평균."""
    s = pl.scan_parquet(L2 / "state" / "**" / "*.parquet", hive_partitioning=False).select("game_id", "side", "t_ms", "gold_us")
    e = s.group_by("game_id", "side").agg(
        ((pl.col("gold_us").sort_by("t_ms").last() - pl.col("gold_us").sort_by("t_ms").first())
         / ((pl.col("t_ms").max() - pl.col("t_ms").min()) / 60_000)).alias("r")).collect()
    return float(e["r"].mean())


def prior_engine(meta: dict, cfg: dict, coefs, state: pl.DataFrame, kills: pl.DataFrame, fp: pl.DataFrame, rbar: float):
    w, h = cfg["prior_g"], cfg["g"]["label_horizon_min"]
    G, TT, DEC, R = {}, {}, {}, None
    for s in SIDES:
        x = state.filter(pl.col("side") == s).sort("t_ms")
        t = x["t_ms"].to_numpy().astype(float)
        R = rbar * np.minimum(h, (t[-1] - t) / 60_000)
        tower = sum(x[f"tower_{k}_taken"].to_numpy() - x[f"tower_{k}_lost"].to_numpy() for k in ("outer", "inner", "base"))
        baron = (x["baron_buff_us"].to_numpy() > 0).astype(float)
        alive_us, alive_opp = x["alive_us"].to_numpy(), x["alive_opp"].to_numpy()
        m = (1 + w["w_alive"] * (alive_us - alive_opp) / 5
             + w["w_dragon"] * (x["dragon_stack_us"].to_numpy() - x["dragon_stack_opp"].to_numpy())
             + w["w_tower"] * tower + w["w_baron"] * baron)
        G[s] = x["gold_us"].to_numpy() + R * m
        TT[s] = np.concatenate([[0.0], np.cumsum(np.diff(R) * m[:-1])])
        step = R[1:] * (w["w_alive"] / 5 * (np.clip(np.diff(alive_us), 0, None) - np.clip(np.diff(alive_opp), 0, None))
                        + w["w_baron"] * np.clip(np.diff(baron), None, 0))
        DEC[s] = np.concatenate([[0.0], np.cumsum(step)])
    deaths = death_risk(kills, fp, {"alive_us": 0.0, "gold_opp": 0.0, "respawn_sum_us": 0.0})
    i = np.clip(np.searchsorted(t, deaths["t_ms"].to_numpy(), side="left") - 1, 0, t.size - 1)   # t_d⁻
    deaths = deaths.with_columns(pl.Series("risk", R[i] * (-w["w_alive"] / 5)))
    return t, G, TT, DEC, deaths
