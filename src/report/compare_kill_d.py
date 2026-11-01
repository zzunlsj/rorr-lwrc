"""킬 → 킬러 D 비교 (결정 27 보류분, 2026-11-01 Jun 요청) — A(결정 27 원안) vs B(상대 데스 비용도 D 파이에 포함, 결정 30 채택).

A: D 파이 = max(−잔여_opp, 0),  잔여_opp = ΔV_opp − Σ Risk_opp − 시간항_opp − 감쇠항_opp      (결정 27 원안, d_pie_excl_death)
B: D 파이 = max(−(잔여_opp + Σ Risk_opp), 0) = max(−(ΔV_opp − 시간항_opp − 감쇠항_opp), 0)
   → 상대 감속 전체(데스로 생긴 몫 포함)를 우리 팀 관여 지분(w_reg)으로 배분. 상대 Risk는 그대로 둔다. (정본, d_pie)

C·Risk·팀 계정·항등식은 바뀌지 않는다 (D는 항등식 밖). 바뀌는 것은 선수 D와 LwRC뿐이다.
비교 항목: D 크기, split-half 신뢰도(#9), 벤치 방향성(#10), 실현 골드 상관, 게임 길이 상관(#11), LwRC+ 순위 변화.

결과: outputs/compare_kill_d.json (리포트 specs/kill_d_comparison_2026-11-01.md의 수치 원본)
사용: PYTHONPATH=src uv run python -m report.compare_kill_d
"""

import json

import numpy as np
import polars as pl
import yaml

from common.paths import L2, OUTPUTS
from ingest.leaguepedia import LWRC_YAML
from metrics.season import bench_direction, load_player_games, lwrc_plus, scope_rows, split_half

COLS = {"A": "lwrc", "B": "lwrc_B"}   # 아래 main에서 lwrc·d = A, lwrc_B·d_B = B로 맞춘다


def d_pies() -> pl.DataFrame:
    """클러스터 × 팀별 A·B D 파이 (timeline 산출: d_pie = B, d_pie_excl_death = A)."""
    c = pl.read_parquet(L2 / "timeline" / "clusters" / "**" / "*.parquet", hive_partitioning=False)
    return c.select("game_id", "k", pl.col("side").alias("team_id"), "n_kills", "c_pie",
                    pl.col("d_pie_excl_death").alias("d_pie_A"), pl.col("d_pie").alias("d_pie_B"))


def allocate_d(pies: pl.DataFrame) -> pl.DataFrame:
    """game_lwrc.allocate와 같은 규칙 (share = w / Σ_팀 w, Σw = 0이면 미배분)을 전 게임에 한 번에."""
    w = pl.read_parquet(L2 / "alloc" / "weights" / "**" / "*.parquet", hive_partitioning=False)
    x = (w.join(pies, on=["game_id", "k", "team_id"])
         .with_columns(pl.col("w_reg").sum().over("game_id", "k", "team_id").alias("sw"))
         .with_columns(pl.when(pl.col("sw") > 0).then(pl.col("w_reg") / pl.col("sw")).otherwise(0.0).alias("share")))
    return x.group_by("game_id", "participant_id").agg((pl.col("d_pie_A") * pl.col("share")).sum().alias("d_A"),
                                                       (pl.col("d_pie_B") * pl.col("share")).sum().alias("d_B_chk"))


def diff_ci(p: pl.DataFrame, sigma: float, min_half: int, mode: str, reps: int, seed: int) -> dict:
    """B − A split-half 상관 차이, 선수 재표집 부트스트랩."""
    p = p.sort("game_start_utc").with_columns(pl.int_range(pl.len()).over("puuid").alias("_i"), pl.len().over("puuid").alias("_n"))
    a, b = ((p.filter(pl.col("_i") < pl.col("_n") // 2), p.filter(pl.col("_i") >= pl.col("_n") // 2)) if mode == "half"
            else (p.filter(pl.col("_i") % 2 == 0), p.filter(pl.col("_i") % 2 == 1)))
    cols = {}
    for v, c in COLS.items():
        j = lwrc_plus(a, c, sigma).join(lwrc_plus(b, c, sigma), on="puuid", suffix="_b").filter(
            (pl.col("g") >= min_half) & (pl.col("g_b") >= min_half)).sort("puuid")
        cols[v] = (j["ix_f"].to_numpy(), j["ix_f_b"].to_numpy())
    rng, n, diffs = np.random.default_rng(seed), cols["A"][0].size, []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        diffs.append(np.corrcoef(cols["B"][0][i], cols["B"][1][i])[0, 1] - np.corrcoef(cols["A"][0][i], cols["A"][1][i])[0, 1])
    return {"diff": float(np.mean(diffs)), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
            "p_B_higher": float((np.array(diffs) > 0).mean()), "n_players": int(n)}


def season_table(p: pl.DataFrame, scope: str, cfg: dict, sigma: float, min_rank: int) -> pl.DataFrame:
    x = scope_rows(p, scope, cfg)
    names = p.sort("game_start_utc").group_by("puuid").agg(pl.col("display_name").last().alias("name"), pl.col("team_code").last().alias("team"))
    a = lwrc_plus(x, "lwrc", sigma).select("puuid", "role", "g", "minutes", pl.col("ix_f").alias("ix_A"))
    b = lwrc_plus(x, "lwrc_B", sigma).select("puuid", pl.col("ix_f").alias("ix_B"))
    t = a.join(b, on="puuid").join(names, on="puuid").filter(pl.col("g") >= min_rank)
    return t.with_columns(pl.col("ix_A").rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("rk_A"),
                          pl.col("ix_B").rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("rk_B"))


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    sc = cfg["season"]
    sigma, min_rank = sc["winsorize_sigma"], sc["min_games_for_ranking"]
    pies = d_pies()
    p = load_player_games(cfg).join(allocate_d(pies), on=["game_id", "participant_id"], how="left").with_columns(
        pl.col("d_A", "d_B_chk").fill_null(0.0))
    assert (p["d_B_chk"] - p["d"]).abs().max() < 1e-6, "정본 D 재현 실패"
    p = p.with_columns(pl.col("d").alias("d_B"), pl.col("lwrc").alias("lwrc_B"),
                       (pl.col("lwrc") - pl.col("d") + pl.col("d_A")).alias("lwrc"), pl.col("d_A").alias("d"))
    out = {}

    # 1. D 크기
    tg = p.group_by("game_id", "team_id").agg(pl.col("d", "d_B", "c_pie", "c", "risk").sum())
    fights = pies.filter(pl.col("n_kills") >= 3)
    out["size"] = {
        "sum": {"c_gold": p["c_gold"].sum(), "c_pie": p["c_pie"].sum(), "risk": p["risk"].sum(), "d_A": p["d"].sum(), "d_B": p["d_B"].sum()},
        "team_game_median_d_over_cpie": {"A": float((tg["d"] / tg["c_pie"]).median()), "B": float((tg["d_B"] / tg["c_pie"]).median())},
        "fight_clusters": fights.height,
        "fight_d_positive": {"A": float((fights["d_pie_A"] > 0).mean()), "B": float((fights["d_pie_B"] > 0).mean())},
        "lwrc_sum": {"A": p["lwrc"].sum(), "B": p["lwrc_B"].sum()},
    }
    won = p.with_columns((pl.col("team_id") == pl.col("winning_team")).alias("won")).group_by("won").agg(
        (pl.col("d").sum() / pl.col("play_min").sum()).alias("d_A_pm"), (pl.col("d_B").sum() / pl.col("play_min").sum()).alias("d_B_pm"),
        (pl.col("lwrc").sum() / pl.col("play_min").sum()).alias("lwrc_A_pm"), (pl.col("lwrc_B").sum() / pl.col("play_min").sum()).alias("lwrc_B_pm"))
    out["win_loss_per_min"] = {("win" if r["won"] else "loss"): {k: v for k, v in r.items() if k != "won"} for r in won.to_dicts()}
    role = p.group_by("role").agg(*[(pl.col(c).sum() / pl.col("play_min").sum()).alias(c) for c in ("c_gold", "c_pie", "d", "d_B", "risk", "lwrc", "lwrc_B")])
    out["role_per_min"] = {r["role"]: {k: v for k, v in r.items() if k != "role"} for r in role.sort("role").to_dicts()}

    # 2. #9 신뢰도
    rel = {}
    for mode in ("half", "oddeven"):
        rel[mode] = {}
        for v, c in {**COLS, "gold_only": "c_gold"}.items():
            r, n = split_half(p, c, sigma, min_rank, mode)
            rel[mode][v] = {"r": r, "spearman_brown": 2 * r / (1 + r), "n": n}
        rel[mode]["B_minus_A"] = diff_ci(p, sigma, min_rank, mode, cfg["allocation"]["bootstrap_reps"], 0)
    out["reliability"] = rel

    # 3. #10 벤치 방향성 · 실현 골드 상관, #11 길이 상관
    out["bench"] = {"A": bench_direction(p, min_rank), "B": bench_direction(p.with_columns(pl.col("lwrc_B").alias("lwrc")), min_rank)}
    out["length_corr"] = {v: float(np.corrcoef(p["play_min"], p[c] / p["play_min"])[0, 1]) for v, c in COLS.items()}
    risk_pm = p.group_by("puuid").agg(pl.col("role").mode().first(), pl.len().alias("g"),
                                      (pl.col("risk").sum() / pl.col("play_min").sum()).alias("risk_r"),
                                      (pl.col("lwrc").sum() / pl.col("play_min").sum()).alias("a_r"),
                                      (pl.col("lwrc_B").sum() / pl.col("play_min").sum()).alias("b_r")).filter(pl.col("g") >= min_rank)
    out["corr_with_risk_rate"] = {r: {"A": float(np.corrcoef(d["risk_r"], d["a_r"])[0, 1]), "B": float(np.corrcoef(d["risk_r"], d["b_r"])[0, 1])}
                                  for (r,), d in sorted(risk_pm.partition_by("role", as_dict=True).items())}

    # 4. 시즌 LwRC+ 순위 변화
    out["season"] = {}
    for scope in ("all", "lck"):
        t = season_table(p, scope, cfg, sigma, min_rank)
        corr = {r: float(np.corrcoef(d["ix_A"], d["ix_B"])[0, 1]) for (r,), d in sorted(t.partition_by("role", as_dict=True).items())}
        moved = t.with_columns((pl.col("rk_A") - pl.col("rk_B")).alias("up"), (pl.col("ix_B") - pl.col("ix_A")).alias("dix"))
        out["season"][scope] = {
            "corr_ix_by_role": corr, "n": t.height,
            "mean_abs_rank_change": float(moved["up"].abs().mean()), "max_abs_rank_change": int(moved["up"].abs().max()),
            "sd_ix": {"A": float(t["ix_A"].std()), "B": float(t["ix_B"].std())},
            "top_up": moved.sort("up", descending=True).head(8).select("name", "team", "role", "g", "ix_A", "ix_B", "rk_A", "rk_B").to_dicts(),
            "top_down": moved.sort("up").head(8).select("name", "team", "role", "g", "ix_A", "ix_B", "rk_A", "rk_B").to_dicts(),
            "top3": {r: d.sort("rk_B").head(3).select("name", "team", "ix_A", "ix_B", "rk_A").to_dicts()
                     for (r,), d in sorted(t.partition_by("role", as_dict=True).items())}}
    (OUTPUTS / "compare_kill_d.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("size", "reliability", "length_corr")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
