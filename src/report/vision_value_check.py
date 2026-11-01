"""시야 설치·제거의 가치 점검 (2026-11-01 Jun 요청) — 시야 피처 정의 후보를 같은 절차로 비교한다.

후보 (나머지 6피처 dmg·assist·absorb·presence·cc·protect는 그대로):
  V0 현행     vision_provided(교전 구역 와드 겹친 초) + vision_denied(교전 구역 제거 개수)        — β 각각
  V1 제거=초  vision_provided + denied_s(제거로 없앤 시야 초 = min(만료, 창 끝) − 제거 시각)       — β 각각
  V2 시야초   vision_s = vision_provided + denied_s  (설치 1초 = 제거로 없앤 1초, 같은 β)          — "시야 초" 하나
  V3 시야점수 vs_delta = 시야창 [s − 30초, e] 동안 늘어난 게임 시야 점수(VISION_SCORE)             — 라이엇 가중
비교: 표본 외 R²(경기 단위 크로스피팅), β, 포지션별 시야 몫, split-half 신뢰도(#9), 시즌 LwRC+ 순위 상관.
배분 재계산: 같은 클러스터 파이(C 파이 + D)를 후보 지분으로 다시 나눈다 — 실현 골드·Risk·팀 계정·항등식은 그대로.

결과: outputs/vision_value_check.json, data/l2/vision_score/cluster_vs_delta.parquet
사용: PYTHONPATH=src uv run python -m report.vision_value_check [--workers 6]
"""

import argparse
import csv
import json
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.paths import L1, L2, MANIFEST, OUTPUTS
from features.alloc_features import FEATURES, OUT as FEAT_DIR
from features.alloc_train import CONTROLS, NNLSRidge, choose_alpha, r2
from ingest.leaguepedia import LWRC_YAML
from ingest.raw_reader import iter_events
from metrics.season import load_player_games, lwrc_plus, scope_rows, split_half

VS_OUT = L2 / "vision_score" / "cluster_vs_delta.parquet"
BASE = ["dmg", "assist", "absorb", "presence", "cc", "protect"]
VARIANTS = {"V0": ["vision_provided", "vision_denied"], "V1": ["vision_provided", "denied_s"],
            "V2": ["vision_s"], "V3": ["vs_delta"]}


def vs_deltas(args: tuple[dict, list[tuple[int, int, int]]]) -> list[dict]:
    """한 게임: 클러스터 시야창별 선수 시야 점수 증가분 (누적값의 창 끝 − 창 시작)."""
    meta, wins = args
    t, vals = [], []
    for schema, _, p in iter_events(meta):
        if schema != "stats_update" or p.get("gameTime") is None:
            continue
        row = np.full(10, np.nan)
        for pp in p.get("participants", []):
            for s in pp.get("stats", []):
                if s.get("name") == "VISION_SCORE":
                    row[pp["participantID"] - 1] = s["value"]
        t.append(p["gameTime"])
        vals.append(row)
    order = np.argsort(t)
    t, v = np.array(t)[order], np.array(vals)[order]
    v = np.fmax.accumulate(np.nan_to_num(v, nan=0.0), axis=0)
    out = []
    for k, w0, w1 in wins:
        i0 = max(int(np.searchsorted(t, w0, side="right")) - 1, 0)
        i1 = max(int(np.searchsorted(t, w1, side="right")) - 1, 0)
        for pid in range(10):
            out.append({"game_id": meta["game_id"], "k": k, "participant_id": pid + 1, "vs_delta": float(v[i1, pid] - v[i0, pid])})
    return out


def extract_vs(cfg: dict, workers: int) -> pl.DataFrame:
    if VS_OUT.exists():
        return pl.read_parquet(VS_OUT)
    pre = cfg["allocation"]["vision_pre_window_sec"] * 1000
    received = {r["game_id"]: r for r in csv.DictReader(open(MANIFEST / "received.csv")) if not r.get("superseded_by")}
    cl = pl.read_parquet(L2 / "timeline" / "clusters" / "**" / "*.parquet", hive_partitioning=False).filter(pl.col("side") == 100)
    jobs = [(received[g], [(k, s - pre, e) for k, s, e in d.select("k", "s_ms", "e_ms").iter_rows()])
            for (g,), d in cl.partition_by(["game_id"], as_dict=True).items()]
    with Pool(workers) as pool:
        df = pl.DataFrame([r for part in pool.imap_unordered(vs_deltas, jobs, chunksize=4) for r in part])
    VS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(VS_OUT)
    return df


def player_features(cfg: dict, workers: int) -> pl.DataFrame:
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=False)
    fk = pl.read_parquet(L2 / "vision_score" / "fight_ward_kills.parquet").group_by("game_id", "k", pl.col("killer").alias("participant_id")).agg(
        pl.col("denied_s").sum())
    vs = extract_vs(cfg, workers)
    games = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("game_id", "match_id")
    folds = pl.read_parquet(L2 / "g_model" / "folds.parquet")
    return (feat.join(fk, on=["game_id", "k", "participant_id"], how="left").join(vs, on=["game_id", "k", "participant_id"], how="left")
            .with_columns(pl.col("denied_s", "vs_delta").fill_null(0.0))
            .with_columns((pl.col("vision_provided") + pl.col("denied_s")).alias("vision_s"))
            .join(games, on="game_id").join(folds, on="match_id"))


def design(pf: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """클러스터 1행: 팀 피처 차이 + Swing·통제항 (alloc_train.load_design의 Swing·통제항 재사용)."""
    from features.alloc_train import load_design
    team = pf.group_by("game_id", "k", "team_id").agg(*[pl.col(c).sum() for c in cols])
    a, b = team.filter(pl.col("team_id") == 100).drop("team_id"), team.filter(pl.col("team_id") == 200).drop("team_id")
    diff = a.join(b, on=["game_id", "k"], suffix="_o").select("game_id", "k", *[(pl.col(c) - pl.col(f"{c}_o")).alias(c) for c in cols])
    return load_design().select("game_id", "k", "swing", "match_id", "fold", *CONTROLS).join(diff, on=["game_id", "k"])


def fit_variant(pf: pl.DataFrame, cols: list[str], k: int, alphas: list[float]) -> tuple[dict, np.ndarray, float]:
    d = design(pf, cols)
    x, c, y = d.select(cols).to_numpy(), d.select(CONTROLS).to_numpy(), d["swing"].to_numpy()
    groups, fold = d["match_id"].to_numpy(), d["fold"].to_numpy()
    oos, betas = np.empty_like(y), {}
    for f in range(k):
        tr, te = fold != f, fold == f
        a = choose_alpha(x[tr], c[tr], y[tr], groups[tr], k, alphas)
        m = NNLSRidge().fit(x[tr], c[tr], y[tr], a)
        betas[f], oos[te] = m.coef_, m.predict(x[te], c[te])
    a_all = choose_alpha(x, c, y, groups, k, alphas)
    return betas, NNLSRidge().fit(x, c, y, a_all).coef_, r2(y, oos)


def reallocate(pf: pl.DataFrame, cols: list[str], betas: dict, vision_cols: list[str]) -> pl.DataFrame:
    """fold β로 지분 → 클러스터 파이(C 파이 + D) 재배분. 선수-게임별 배분 합과 시야 몫."""
    xf, fold = pf.select(cols).to_numpy(), pf["fold"].to_numpy()
    w, wv = np.zeros(pf.height), np.zeros(pf.height)
    vi = [cols.index(v) for v in vision_cols]
    for f, b in betas.items():
        m = fold == f
        w[m] = xf[m] @ b
        wv[m] = xf[m][:, vi] @ b[vi]
    cl = pl.read_parquet(L2 / "timeline" / "clusters" / "**" / "*.parquet", hive_partitioning=False).select(
        "game_id", "k", pl.col("side").alias("team_id"), "c_pie", "d_pie")
    x = (pf.select("game_id", "k", "participant_id", "team_id").with_columns(pl.Series("w", w), pl.Series("wv", wv))
         .join(cl, on=["game_id", "k", "team_id"])
         .with_columns(pl.col("w").sum().over("game_id", "k", "team_id").alias("sw"))
         .with_columns(pl.when(pl.col("sw") > 0).then(pl.col("w") / pl.col("sw")).otherwise(0.0).alias("share")))
    return x.group_by("game_id", "participant_id").agg(
        (pl.col("c_pie") * pl.col("share")).sum().alias("c_pie_v"), (pl.col("d_pie") * pl.col("share")).sum().alias("d_v"),
        ((pl.col("c_pie") + pl.col("d_pie")) * pl.when(pl.col("w") > 0).then(pl.col("share") * pl.col("wv") / pl.col("w")).otherwise(0.0))
        .sum().alias("vision_gold_v"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    k, alphas, sigma, min_rank = cfg["g"]["crossfit_k"], cfg["allocation"]["ridge_alphas"], cfg["season"]["winsorize_sigma"], cfg["season"]["min_games_for_ranking"]
    pf = player_features(cfg, args.workers)
    p0 = load_player_games(cfg)
    sd = pf.select([pl.col(c).std() for c in ("vision_provided", "vision_denied", "denied_s", "vision_s", "vs_delta")]).to_dicts()[0]
    out, season = {"feature_sd_player_cluster": sd}, {}
    for name, vcols in VARIANTS.items():
        cols = BASE + vcols
        betas, b_all, oos = fit_variant(pf, cols, k, alphas)
        alloc = reallocate(pf, cols, betas, vcols)
        p = (p0.join(alloc, on=["game_id", "participant_id"], how="left").with_columns(pl.col("c_pie_v", "d_v", "vision_gold_v").fill_null(0.0))
             .with_columns((pl.col("c_gold") + pl.col("c_pie_v") + pl.col("d_v") + pl.col("risk")).alias("lwrc_v")))
        rel = {m: split_half(p, "lwrc_v", sigma, min_rank, m)[0] for m in ("half", "oddeven")}
        role = p.group_by("role").agg((pl.col("vision_gold_v").sum() / pl.col("play_min").sum()).alias("vg_pm"),
                                      pl.col("vision_gold_v").sum().alias("vg"))
        tot = role["vg"].sum()
        out[name] = {"features": cols, "oos_r2": oos, "beta_all": dict(zip(cols, map(float, b_all))),
                     "beta_x_sd_vision": {c: float(b_all[cols.index(c)] * sd[c]) for c in vcols},
                     "split_half": rel,
                     "vision_gold_per_min": {r["role"]: r["vg_pm"] for r in role.to_dicts()},
                     "vision_gold_share_pct": {r["role"]: 100 * r["vg"] / tot for r in role.to_dicts()},
                     "check_identity_pie_total": float(p["c_pie_v"].sum() + p["d_v"].sum())}
        x = scope_rows(p, "lck", cfg)
        season[name] = lwrc_plus(x, "lwrc_v", sigma).filter(pl.col("g") >= min_rank).select("puuid", "role", pl.col("ix_f").alias(name))
    # 시즌 LwRC+ (LCK) — 현행(V0) 대비 포지션 내 상관·순위 변화
    s = season["V0"]
    for name in ("V1", "V2", "V3"):
        s = s.join(season[name].drop("role"), on="puuid")
    names = p0.sort("game_start_utc").group_by("puuid").agg(pl.col("display_name").last().alias("name"))
    s = s.join(names, on="puuid")
    out["lck_season_vs_V0"] = {}
    for name in ("V1", "V2", "V3"):
        r = s.with_columns(pl.col("V0").rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("r0"),
                           pl.col(name).rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("r1"))
        out["lck_season_vs_V0"][name] = {
            "corr_by_role": {ro: float(np.corrcoef(d["V0"], d[name])[0, 1]) for (ro,), d in r.partition_by("role", as_dict=True).items()},
            "mean_abs_rank_change": float((r["r0"] - r["r1"]).abs().mean()),
            "movers": r.with_columns((pl.col(name) - pl.col("V0")).alias("dix")).sort("dix").select(
                "name", "role", pl.col("V0").round(1), pl.col(name).round(1), "r0", "r1").to_dicts()[:3]
            + r.with_columns((pl.col(name) - pl.col("V0")).alias("dix")).sort("dix", descending=True).select(
                "name", "role", pl.col("V0").round(1), pl.col(name).round(1), "r0", "r1").to_dicts()[:3]}
    (OUTPUTS / "vision_value_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for name in VARIANTS:
        o = out[name]
        print(name, f"R² {o['oos_r2']:.4f}", "split-half", {m: round(v, 3) for m, v in o["split_half"].items()},
              "시야β×SD", {c: round(v) for c, v in o["beta_x_sd_vision"].items()},
              "시야몫%", {r: round(v, 1) for r, v in sorted(o["vision_gold_share_pct"].items())},
              "분당", {r: round(v, 1) for r, v in sorted(o["vision_gold_per_min"].items())})


if __name__ == "__main__":
    main()
