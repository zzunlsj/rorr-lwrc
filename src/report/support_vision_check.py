"""서폿 기여 · 시야 기여 점검 (2026-11-01 Jun 요청) — 배분(spec/04)이 서폿 고유 기여와 시야를 얼마나 인정하는가.

  1. 포지션별 관여 피처 프로파일 — 팀 합 중 포지션 몫
  2. 지분 w_i = Σ β_k x_ik 의 피처별 구성 (포지션별)
  3. 배분된 파이(C 파이 + D)를 피처별로 나눈 값 — 포지션별 분당, 시야 몫은 선수별 순위까지
  4. 모델 비교 (경기 단위 크로스피팅 표본 외 R²): 피처 빼기 · 단독 · 포지션별 β
  5. CC가 0인 이유 — 팀 차이 피처 간 상관, CC 단독 β
  6. 서폿 안에서 무엇이 LwRC를 가르나 — 선수별 피처율과 LwRC/분 상관 (골드 통제)
  7. 시야 데이터 — 포지션·종류별 설치, unknown 제외 비율

결과: outputs/support_vision_check.json (리포트 specs/support_vision_check_2026-11-01.md의 수치 원본)
사용: PYTHONPATH=src uv run python -m report.support_vision_check
"""

import json

import numpy as np
import polars as pl
import yaml

from common.paths import L1, L2, OUTPUTS
from features.alloc_features import FEATURES, OUT as FEAT_DIR
from features.alloc_train import CONTROLS, NNLSRidge, choose_alpha, load_design, r2
from ingest.leaguepedia import LWRC_YAML
from metrics.season import load_player_games, partial_corr

ROLES = ["TOP", "JG", "MID", "BOT", "SUP"]
VISION = ["vision"]   # 결정 32: 설치·제거를 라이엇 비율로 합친 한 피처


def player_clusters(cfg: dict) -> pl.DataFrame:
    """선수 × 클러스터: 피처, 포지션, fold β로 분해한 지분, 배분된 파이(C 파이 + D)와 피처별 몫."""
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=False)
    roles = pl.read_parquet(L1 / "participants" / "**" / "*.parquet", hive_partitioning=False).select(
        "game_id", "participant_id", pl.col("role").replace_strict(cfg["season"]["role_codes"]))
    games = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("game_id", "match_id")
    folds = pl.read_parquet(L2 / "g_model" / "folds.parquet")
    beta = pl.read_parquet(L2 / "alloc" / "model" / "beta.parquet").filter(pl.col("fold") != "all").with_columns(
        pl.col("fold").cast(pl.Int64))
    x = (feat.join(roles, on=["game_id", "participant_id"]).join(games, on="game_id").join(folds, on="match_id")
         .join(beta.select("fold", *[pl.col(f).alias(f"b_{f}") for f in FEATURES]), on="fold"))
    x = x.with_columns(*[(pl.col(f) * pl.col(f"b_{f}")).alias(f"w_{f}") for f in FEATURES])
    x = x.with_columns(pl.sum_horizontal([f"w_{f}" for f in FEATURES]).alias("w"))
    cl = pl.read_parquet(L2 / "timeline" / "clusters" / "**" / "*.parquet", hive_partitioning=False).select(
        "game_id", "k", pl.col("side").alias("team_id"), (pl.col("c_pie") + pl.col("d_pie")).alias("pie"))
    x = (x.join(cl, on=["game_id", "k", "team_id"])
         .with_columns(pl.col("w").sum().over("game_id", "k", "team_id").alias("sw"))
         .with_columns(pl.when(pl.col("sw") > 0).then(pl.col("pie") * pl.col("w") / pl.col("sw")).otherwise(0.0).alias("alloc")))
    return x.with_columns(*[pl.when(pl.col("w") > 0).then(pl.col("alloc") * pl.col(f"w_{f}") / pl.col("w")).otherwise(0.0)
                            .alias(f"a_{f}") for f in FEATURES])


def oos_r2(d: pl.DataFrame, cols: list[str], k: int, alphas: list[float]) -> float:
    """alloc_train과 같은 절차(경기 단위 fold, 내부 CV α)의 표본 외 R². cols가 비면 통제항만."""
    c, y = d.select(CONTROLS).to_numpy(), d["swing"].to_numpy()
    groups, fold = d["match_id"].to_numpy(), d["fold"].to_numpy()
    oos = np.empty_like(y)
    for f in range(k):
        tr, te = fold != f, fold == f
        if not cols:
            c1 = np.hstack([c, np.ones((c.shape[0], 1))])
            oos[te] = c1[te] @ np.linalg.lstsq(c1[tr], y[tr], rcond=None)[0]
            continue
        x = d.select(cols).to_numpy()
        a = choose_alpha(x[tr], c[tr], y[tr], groups[tr], k, alphas)
        oos[te] = NNLSRidge().fit(x[tr], c[tr], y[tr], a).predict(x[te], c[te])
    return r2(y, oos)


def role_design(pc: pl.DataFrame, d: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """포지션별 β용 설계: 피처 × 포지션의 팀 차이(우리 100 − 상대 200)."""
    cols = [f"{f}__{r}" for f in FEATURES for r in ROLES]
    wide = pc.group_by("game_id", "k", "team_id").agg(
        *[pl.col(f).filter(pl.col("role") == r).sum().alias(f"{f}__{r}") for f in FEATURES for r in ROLES])
    a, b = wide.filter(pl.col("team_id") == 100).drop("team_id"), wide.filter(pl.col("team_id") == 200).drop("team_id")
    diff = a.join(b, on=["game_id", "k"], suffix="_o").select("game_id", "k", *[(pl.col(c) - pl.col(f"{c}_o")).alias(c) for c in cols])
    return d.select("game_id", "k", "swing", "match_id", "fold", *CONTROLS).join(diff, on=["game_id", "k"]), cols


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    k, alphas = cfg["g"]["crossfit_k"], cfg["allocation"]["ridge_alphas"]
    pc = player_clusters(cfg)
    p = load_player_games(cfg)
    minutes = p.group_by("role").agg(pl.col("play_min").sum())
    out = {}

    # 1. 피처 프로파일: 팀 합 중 포지션 몫 (%), 선수-클러스터 평균
    tot = pc.select(*[pl.col(f).sum() for f in FEATURES])
    prof = pc.group_by("role").agg(*[pl.col(f).sum().alias(f) for f in FEATURES], *[pl.col(f).mean().alias(f"mean_{f}") for f in FEATURES])
    out["feature_share_pct"] = {r["role"]: {f: 100 * r[f] / tot[f][0] for f in FEATURES} for r in prof.to_dicts()}
    out["feature_mean"] = {r["role"]: {f: r[f"mean_{f}"] for f in FEATURES} for r in prof.to_dicts()}

    # 2. 지분 구성, 3. 배분 파이의 피처별 몫 (분당)
    comp = pc.group_by("role").agg(pl.col("w").sum(), *[pl.col(f"w_{f}").sum() for f in FEATURES],
                                   pl.col("alloc").sum(), *[pl.col(f"a_{f}").sum() for f in FEATURES]).join(minutes, on="role")
    out["w_composition_pct"] = {r["role"]: {f: 100 * r[f"w_{f}"] / r["w"] for f in FEATURES} for r in comp.to_dicts()}
    out["pie_per_min"] = {r["role"]: {"total": r["alloc"] / r["play_min"], **{f: r[f"a_{f}"] / r["play_min"] for f in FEATURES}}
                          for r in comp.to_dicts()}
    lw = p.group_by("role").agg((pl.col("lwrc").sum() / pl.col("play_min").sum()).alias("lwrc_pm"))
    out["vision_share_of_lwrc_pct"] = {r["role"]: 100 * sum(out["pie_per_min"][r["role"]][f] for f in VISION) / r["lwrc_pm"]
                                       for r in lw.to_dicts()}
    # 선수별 시야 몫 (분당, 10게임 이상)
    pv = (pc.join(p.select("game_id", "participant_id", "puuid", "display_name", "team_code", "play_min"), on=["game_id", "participant_id"])
          .group_by("puuid").agg(pl.col("display_name").last().alias("name"), pl.col("team_code").mode().first().alias("team"),
                                 pl.col("role").mode().first(), pl.col("a_vision").sum(),
                                 pl.col("game_id").n_unique().alias("g")))
    pmin = p.group_by("puuid").agg(pl.col("play_min").sum())
    pv = (pv.join(pmin, on="puuid").filter(pl.col("g") >= cfg["season"]["min_games_for_ranking"])
          .with_columns((pl.col("a_vision") / pl.col("play_min")).alias("vision_pm")))
    out["vision_pm_by_role_players"] = {r: {"median": float(d["vision_pm"].median()), "max": float(d["vision_pm"].max())}
                                        for (r,), d in pv.partition_by("role", as_dict=True).items()}
    out["vision_top10"] = pv.sort("vision_pm", descending=True).head(10).select("name", "team", "role", "g", pl.col("vision_pm").round(1)).to_dicts()

    # 4. 모델 비교
    d = load_design()
    ab = {"full": FEATURES, "controls_only": []}
    for drop in (["cc"], ["protect"], VISION):
        ab["drop_" + "+".join(drop)] = [f for f in FEATURES if f not in drop]
    for only in (["cc"], ["protect"], VISION):
        ab["only_" + "+".join(only)] = only
    out["oos_r2"] = {name: oos_r2(d, cols, k, alphas) for name, cols in ab.items()}
    rd, rcols = role_design(pc, d)
    out["oos_r2"]["role_specific_beta"] = oos_r2(rd, rcols, k, alphas)
    alpha = choose_alpha(rd.select(rcols).to_numpy(), rd.select(CONTROLS).to_numpy(), rd["swing"].to_numpy(),
                         rd["match_id"].to_numpy(), k, alphas)
    m = NNLSRidge().fit(rd.select(rcols).to_numpy(), rd.select(CONTROLS).to_numpy(), rd["swing"].to_numpy(), alpha)
    sd = {f: pc.group_by("role").agg(pl.col(f).std()).to_dict(as_series=False) for f in FEATURES}
    sd_of = {f: dict(zip(sd[f]["role"], sd[f][f])) for f in FEATURES}
    out["role_specific_beta_x_sd"] = {r: {f: float(m.coef_[rcols.index(f"{f}__{r}")] * sd_of[f][r]) for f in FEATURES} for r in ROLES}
    pooled = json.loads((L2 / "alloc" / "model" / "validation.json").read_text())["beta_ci"]
    b_all = {b["feature"]: b["beta"] for b in pooled}
    out["pooled_beta_x_sd_by_role"] = {r: {f: float(b_all[f] * sd_of[f][r]) for f in FEATURES} for r in ROLES}

    # 5. CC: 팀 차이 피처 간 상관
    cm = np.corrcoef(d.select(FEATURES).to_numpy(), rowvar=False)
    out["team_diff_corr"] = {f: {g: float(cm[i, j]) for j, g in enumerate(FEATURES)} for i, f in enumerate(FEATURES) if f in ("cc", "protect", *VISION)}

    # 6. 서폿 안: 선수별 피처율(클러스터 평균) ~ LwRC/분, 골드 통제
    rates = pc.join(p.select("game_id", "participant_id", "puuid"), on=["game_id", "participant_id"]).group_by("puuid").agg(
        *[pl.col(f).mean() for f in FEATURES])
    sp = (p.group_by("puuid").agg(pl.col("role").mode().first(), pl.len().alias("g"),
                                  (pl.col("lwrc").sum() / pl.col("play_min").sum()).alias("lwrc_r"),
                                  (pl.col("c_gold").sum() / pl.col("play_min").sum()).alias("gold_r"))
          .filter(pl.col("g") >= cfg["season"]["min_games_for_ranking"]).join(rates, on="puuid"))
    out["within_role_feature_vs_lwrc"] = {}
    for (r,), dd in sorted(sp.partition_by("role", as_dict=True).items()):
        lwr, gd = dd["lwrc_r"].to_numpy(), dd["gold_r"].to_numpy()
        out["within_role_feature_vs_lwrc"][r] = {"n": dd.height, **{f: {"raw": partial_corr(dd[f].to_numpy(), lwr, []),
                                                                        "given_gold": partial_corr(dd[f].to_numpy(), lwr, [gd])}
                                                                    for f in FEATURES}}

    # 7. 시야 데이터
    we = pl.read_parquet(L1 / "ward_events" / "**" / "*.parquet", hive_partitioning=False)
    roles = pc.select("game_id", "participant_id", "role").unique()
    wr = we.join(roles, left_on=["game_id", "placer"], right_on=["game_id", "participant_id"], how="left")
    out["wards"] = {
        "placed": we.height, "unknown_pct": 100 * float((we["ward_type"] == "unknown").mean()),
        "vision_eligible_pct": 100 * float(we["vision_eligible"].mean()),
        "eligible_by_role_pct": {r["role"]: 100 * r["n"] / int(wr["vision_eligible"].sum())
                                 for r in wr.filter(pl.col("vision_eligible")).group_by("role").agg(pl.len().alias("n")).to_dicts()},
        "type_by_role": {r["role"]: {k_: v for k_, v in r.items() if k_ != "role"}
                         for r in wr.filter(pl.col("vision_eligible")).group_by("role").agg(
                             *[(pl.col("ward_type") == t).sum().alias(t) for t in cfg["allocation"]["vision_ward_types"]]).to_dicts()},
        "player_clusters_with_vision_pct": {r["role"]: r["pct"] for r in pc.group_by("role").agg(
            (100 * (pl.col("vision_provided") > 0).mean()).alias("pct")).to_dicts()},
    }
    (OUTPUTS / "support_vision_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k_: out[k_] for k_ in ("oos_r2", "vision_share_of_lwrc_pct")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
