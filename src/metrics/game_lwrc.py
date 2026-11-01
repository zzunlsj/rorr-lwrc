"""Step 5 — 파이 배분 · 보존 항등식 → 게임별 선수 LwRC (spec/03 §3.3, spec/05 §5.1·§5.2, 검증 spec/07 #4·#5·#6).

변형 4가지를 같은 게임에서 나란히 산출한다 (spec/07 "비교 기준선"):
  main     학습 G · 속도형 파이(결정 27·28) · 회귀 지분           ← 정본 후보
  prior_w  학습 G · 속도형 파이 · prior 지분                      ← 가중 방식 기준선 (#9)
  legacy   학습 G · 기존 ΔG 파이(c_pie_legacy) · 회귀 지분        ← 결정 27 이전 정의
  prior_g  prior G(spec/01 §1.5) · 속도형 파이 · 회귀 지분         ← G 기준선 (#5)

배분 (spec/03 §3.3): share_i = w_i / Σ_팀 w. Σw = 0이면 C 파이 → 팀 계정, D 파이 → D_unattributed.
선수 C = 실현 골드(속도형: 클러스터 + 드리프트 / legacy: 드리프트만) + C 파이 배분.
항등식 (팀별): Σ_i (실현 골드 + C 파이 + Risk) + 팀 계정 = G(END) − G(0). |잔차| ≥ identity.epsilon이면 그 게임은 집계 제외.

결과: data/l3/game_player_lwrc/tournament=<t>/stage=<s>/<game_id>.parquet, data/l3/game_team/…, data/l3/step5_validation.json,
      data/l3/excluded_identity.csv
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.config import data_snapshot
from common.paths import L1, L2, L3
from features.g_train import SPEC_VERSION
from features.prior_g import prior_engine, team_income_rate
from features.timeline import build_timeline, load_coefs
from ingest.leaguepedia import LWRC_YAML

TL, W_DIR = L2 / "timeline", L2 / "alloc" / "weights"
OUT_P, OUT_T = L3 / "game_player_lwrc", L3 / "game_team"
VARIANTS = ("main", "prior_w", "legacy", "prior_g")


def _read(base, table: str, meta: dict) -> pl.DataFrame:
    return pl.read_parquet(base / table / f"tournament={meta['tournament']}" / "**" / f"{meta['game_id']}.parquet",
                           hive_partitioning=False)


def allocate(clusters: pl.DataFrame, weights: pl.DataFrame, w_col: str, c_col: str, d_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """클러스터 파이를 지분으로 배분. 선수별 (c_pie, d) 와 팀별 (미배분 C, D_unattributed)."""
    pie = clusters.select("k", pl.col("side").alias("team_id"), pl.col(c_col).alias("cp"), pl.col(d_col).alias("dp"))
    x = (weights.select("k", "participant_id", "team_id", pl.col(w_col).alias("w")).join(pie, on=["k", "team_id"])
         .with_columns(pl.col("w").sum().over("k", "team_id").alias("sw")))
    x = x.with_columns(pl.when(pl.col("sw") > 0).then(pl.col("w") / pl.col("sw")).otherwise(0.0).alias("share"))
    player = x.group_by("participant_id").agg((pl.col("cp") * pl.col("share")).sum().alias("c_pie"),
                                              (pl.col("dp") * pl.col("share")).sum().alias("d"))
    team = (x.filter(pl.col("sw") == 0).unique(["k", "team_id"]).group_by("team_id")
            .agg(pl.col("cp").sum().alias("c_unalloc"), pl.col("dp").sum().alias("d_unattributed")))
    team = (pl.DataFrame({"team_id": [100, 200]}, schema={"team_id": pl.Int64}).join(team, on="team_id", how="left")
            .with_columns(pl.col("c_unalloc", "d_unattributed").fill_null(0.0)))
    return player, team


def ledger(variant: str, tl: dict[str, pl.DataFrame], weights: pl.DataFrame, eps: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """한 변형의 선수 성분과 팀 항등식."""
    c, s, gp, deaths = tl["clusters"], tl["summary"], tl["gold_player"], tl["deaths"]
    legacy = variant == "legacy"
    w_col = "w_prior" if variant == "prior_w" else "w_reg"
    player, team = allocate(c, weights, w_col, *(("c_pie_legacy", "d_pie_legacy") if legacy else ("c_pie", "d_pie")))
    risk = deaths.group_by(pl.col("victim").alias("participant_id")).agg(pl.col("risk").sum())
    gold = gp.select("participant_id", "team_id",
                     (pl.col("gold_drift") if legacy else pl.col("gold_cluster") + pl.col("gold_drift")).alias("c_gold"))
    p = (gold.join(player, on="participant_id", how="left").join(risk, on="participant_id", how="left")
         .with_columns(pl.col("c_pie", "d", "risk").fill_null(0.0))
         .with_columns((pl.col("c_gold") + pl.col("c_pie")).alias("c"))
         .with_columns((pl.col("c") + pl.col("d") + pl.col("risk")).alias("lwrc")))
    if legacy:
        acct_cluster = c.group_by(pl.col("side").alias("team_id")).agg(
            (pl.col("dG") - pl.col("risk_sum") - pl.col("c_pie_legacy")).sum().alias("acct_cluster"))
    else:
        acct_cluster = c.group_by(pl.col("side").alias("team_id")).agg(pl.col("team_acct").sum().alias("acct_cluster"))
    sums = p.group_by("team_id").agg(pl.col("c_gold", "c_pie", "d", "risk").sum())
    t = (s.select(pl.col("side").alias("team_id"), "G0", "G_end", "dG_total", "team_acct_drift")
         .join(acct_cluster, on="team_id").join(team, on="team_id").join(sums, on="team_id")
         .with_columns((pl.col("acct_cluster") + pl.col("team_acct_drift") + pl.col("c_unalloc")).alias("team_acct"))
         .with_columns((pl.col("c_gold") + pl.col("c_pie") + pl.col("risk") + pl.col("team_acct") - pl.col("dG_total")).alias("identity_resid"))
         .with_columns((pl.col("identity_resid").abs() < eps).alias("identity_pass"), pl.lit(variant).alias("variant")))
    return p.with_columns(pl.lit(variant).alias("variant")), t


def build_one(meta: dict, cfg: dict, coefs: dict, rbar: float, snapshot: str) -> dict:
    eps = cfg["identity"]["epsilon"]
    tl = {name: _read(TL, name, meta) for name in ("clusters", "summary", "gold_player", "deaths")}
    tl_prior = build_timeline(meta, cfg, coefs, engine=partial(prior_engine, rbar=rbar))
    weights = pl.read_parquet(
        W_DIR / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet")
    parts, teams = [], []
    for v in VARIANTS:
        p, t = ledger(v, tl_prior if v == "prior_g" else tl, weights, eps)
        parts.append(p)
        teams.append(t)
    teams = pl.concat(teams, how="diagonal_relaxed")
    wide = parts[0].drop("variant")
    for v, p in zip(VARIANTS[1:], parts[1:]):
        wide = wide.join(p.select("participant_id", *[pl.col(x).alias(f"{x}_{v}") for x in ("c", "d", "risk", "lwrc")]),
                         on="participant_id")
    g = meta["game"]
    out = (wide.join(meta["participants"], on="participant_id")
           .with_columns(pl.lit(meta["game_id"]).alias("game_id"), pl.lit(g["match_id"]).alias("match_id"),
                         pl.lit(meta["tournament"]).alias("tournament"), pl.lit(meta["stage"]).alias("stage"),
                         pl.lit(g["patch"]).alias("patch"), pl.lit(tl["summary"]["game_min"][0]).alias("play_min"),
                         pl.lit(bool(teams.filter(pl.col("variant") == "main")["identity_pass"].all())).alias("identity_pass"),
                         pl.lit(SPEC_VERSION).alias("spec_version"), pl.lit(snapshot).alias("data_snapshot")))
    for base, df in ((OUT_P, out), (OUT_T, teams.with_columns(pl.lit(meta["game_id"]).alias("game_id")))):
        path = base / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")
    return {"game_id": meta["game_id"], "tournament": meta["tournament"], "stage": meta["stage"],
            "platform_game_id": g["platform_game_id"],
            "max_resid": {v: float(teams.filter(pl.col("variant") == v)["identity_resid"].abs().max()) for v in VARIANTS}}


def game_list() -> list[dict]:
    games = (pl.scan_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True).filter(~pl.col("excluded"))
             .select("game_id", "tournament", "stage", "match_id", "patch", "platform_game_id").collect())
    parts = pl.read_parquet(L1 / "participants" / "**" / "*.parquet", hive_partitioning=False).select(
        "game_id", "participant_id", "puuid", "display_name", "role", "esports_team_id", "team_code", "champion")
    by_game = {g: d.drop("game_id") for (g,), d in parts.partition_by(["game_id"], as_dict=True).items()}
    return [{"game_id": r["game_id"], "tournament": r["tournament"], "stage": r["stage"], "game": r,
             "participants": by_game[r["game_id"]]} for r in games.to_dicts()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    t0 = time.time()
    rbar = team_income_rate()
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(partial(build_one, cfg=cfg, coefs=load_coefs(), rbar=rbar, snapshot=data_snapshot()),
                                       game_list(), chunksize=4))
    eps = cfg["identity"]["epsilon"]
    failed = sorted((r for r in res if r["max_resid"]["main"] >= eps), key=lambda r: r["game_id"])
    pl.DataFrame([{"game_id": r["game_id"], "platform_game_id": r["platform_game_id"], "tournament": r["tournament"],
                   "stage": r["stage"], "step": "5_identity", "reason": f"identity_resid={r['max_resid']['main']:.3g}"}
                  for r in failed], schema=["game_id", "platform_game_id", "tournament", "stage", "step", "reason"]
                 ).write_csv(L3 / "excluded_identity.csv")
    summary = {"games": len(res), "identity_failed_main": len(failed), "rbar_team_gold_per_min": rbar,
               "max_resid": {v: max(r["max_resid"][v] for r in res) for v in VARIANTS}, "epsilon": eps,
               "spec_version": SPEC_VERSION, "data_snapshot": data_snapshot()}
    (L3 / "step5_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"게임 {len(res)}개, 항등식 미통과 {len(failed)}, 최대 잔차 {summary['max_resid']}, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
