"""게임 내 시야 점수(VISION_SCORE)와 LwRC 시야 몫 비교 — 정글 시야 몫이 작은 이유 (2026-11-01 Jun 질문).

raw `stats_update.participants[].stats`의 누적값 VISION_SCORE · WARD_PLACED · WARD_KILLED를 1Hz로 읽어
  1. 선수-게임 최종 시야 점수 (포지션별 분당, 팀 내 몫)
  2. 시야 점수가 교전 구간(클러스터 [t_s, t_e])에서 쌓인 몫 vs 교전 밖에서 쌓인 몫
  3. 시야 점수 ~ LwRC 시야 몫(배분된 파이 중 시야 피처 몫)의 관계
를 본다. L0는 읽기만 한다.

결과: data/l2/vision_score/player_games.parquet, outputs/vision_score_check.json
사용: PYTHONPATH=src uv run python -m report.vision_score_check [--workers 6]
"""

import argparse
import csv
import json
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.paths import L1, L2, MANIFEST, OUTPUTS
from ingest.leaguepedia import LWRC_YAML
from ingest.raw_reader import iter_events
from metrics.season import load_player_games, partial_corr
from report.support_vision_check import player_clusters

OUT = L2 / "vision_score" / "player_games.parquet"
STATS = ("VISION_SCORE", "WARD_PLACED", "WARD_KILLED")


def one_game(args: tuple[dict, list[tuple[int, int]], int]) -> list[dict]:
    """한 게임: 선수별 최종 누적값 + 교전 구간 안에서 늘어난 시야 점수."""
    meta, spans, last_ms = args
    ticks: list[tuple[int, dict[int, dict]]] = []
    for schema, _, p in iter_events(meta):
        if schema != "stats_update" or p.get("gameTime") is None or p["gameTime"] > last_ms:
            continue
        vals = {}
        for pp in p.get("participants", []):
            st = {s["name"]: s["value"] for s in pp.get("stats", []) if s.get("name") in STATS}
            vals[pp["participantID"]] = st
        ticks.append((p["gameTime"], vals))
    ticks.sort(key=lambda x: x[0])
    t = np.array([x[0] for x in ticks])
    in_cl = np.zeros(t.size, bool)
    for s, e in spans:
        in_cl |= (t > s) & (t <= e)                       # 틱 구간 (t_{i-1}, t_i]의 끝점 기준 — 타임라인 분할과 같은 규칙
    rows = []
    for pid in range(1, 11):
        vs = np.array([x[1].get(pid, {}).get("VISION_SCORE", np.nan) for x in ticks], float)
        vs = np.fmax.accumulate(np.nan_to_num(vs, nan=0.0))   # 누적값 — 결측·되감김은 직전 최대값 유지
        d = np.diff(vs, prepend=0.0)
        last = ticks[-1][1].get(pid, {})
        rows.append({"game_id": meta["game_id"], "participant_id": pid, "vision_score": float(vs[-1]),
                     "vs_in_cluster": float(d[in_cl].sum()), "ward_placed": last.get("WARD_PLACED"),
                     "ward_killed": last.get("WARD_KILLED")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    received = {r["game_id"]: r for r in csv.DictReader(open(MANIFEST / "received.csv")) if not r.get("superseded_by")}
    games = pl.read_parquet(L1 / "games" / "**" / "*.parquet").filter(~pl.col("excluded")).select("game_id", "last_tick_ms")
    cl = pl.read_parquet(L2 / "timeline" / "clusters" / "**" / "*.parquet", hive_partitioning=False).filter(
        pl.col("side") == 100).select("game_id", "t_s_ms", "t_e_ms")
    spans = {g: list(zip(d["t_s_ms"], d["t_e_ms"])) for (g,), d in cl.partition_by(["game_id"], as_dict=True).items()}
    jobs = [(received[g], spans.get(g, []), int(last)) for g, last in games.iter_rows()]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(one_game, jobs, chunksize=4) for r in part]
    vs = pl.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vs.write_parquet(OUT)

    p = load_player_games(cfg).join(vs, on=["game_id", "participant_id"])
    pc = player_clusters(cfg)
    vgold = pc.group_by("game_id", "participant_id").agg(
        pl.col("a_vision").sum().alias("vision_gold"),
        pl.col("vision_provided").sum().alias("vp_sum"), pl.col("vision_denied").sum().alias("vd_sum"))
    p = p.join(vgold, on=["game_id", "participant_id"], how="left").with_columns(pl.col("vision_gold", "vp_sum", "vd_sum").fill_null(0.0))
    out = {}

    role = p.group_by("role").agg(
        (pl.col("vision_score").sum() / pl.col("play_min").sum()).alias("vs_pm"),
        (pl.col("vs_in_cluster").sum() / pl.col("vision_score").sum()).alias("vs_in_cluster_share"),
        (pl.col("vision_gold").sum() / pl.col("play_min").sum()).alias("vision_gold_pm"),
        pl.col("vision_score").sum().alias("vs"), pl.col("vision_gold").sum().alias("vg"),
        pl.col("ward_placed").sum(), pl.col("ward_killed").sum())
    tot_vs, tot_vg = role["vs"].sum(), role["vg"].sum()
    tot_wk, tot_wp = role["ward_killed"].sum(), role["ward_placed"].sum()
    out["by_role"] = {r["role"]: {"vision_score_per_min": r["vs_pm"], "vision_score_team_share_pct": 100 * r["vs"] / tot_vs,
                                  "lwrc_vision_gold_share_pct": 100 * r["vg"] / tot_vg, "vision_gold_per_min": r["vision_gold_pm"],
                                  "vs_earned_in_fights_pct": 100 * r["vs_in_cluster_share"],
                                  "ward_placed_share_pct": 100 * r["ward_placed"] / tot_wp,
                                  "ward_killed_share_pct": 100 * r["ward_killed"] / tot_wk,
                                  "vision_gold_per_vision_score": r["vg"] / r["vs"]}
                     for r in role.sort("role").to_dicts()}
    allp = p.select(pl.col("vs_in_cluster").sum() / pl.col("vision_score").sum()).item()
    cover = pl.read_parquet(L2 / "timeline" / "summary" / "**" / "*.parquet", hive_partitioning=False)["coverage"].mean()
    out["vs_in_fights_all_pct"], out["fight_time_coverage_pct"] = 100 * allp, 100 * float(cover)

    # 선수-게임 / 선수 단위 관계 (포지션 안)
    out["player_game_corr"] = {}
    for (r,), d in sorted(p.partition_by("role", as_dict=True).items()):
        out["player_game_corr"][r] = {
            "vs_vs_vision_gold": float(np.corrcoef(d["vision_score"], d["vision_gold"])[0, 1]),
            "vs_in_fight_vs_vision_gold": float(np.corrcoef(d["vs_in_cluster"], d["vision_gold"])[0, 1]),
            "vs_vs_vp_sum": float(np.corrcoef(d["vision_score"], d["vp_sum"])[0, 1])}
    sp = (p.group_by("puuid").agg(pl.col("role").mode().first(), pl.len().alias("g"), pl.col("display_name").last().alias("name"),
                                  pl.col("team_code").mode().first().alias("team"),
                                  *[(pl.col(c).sum() / pl.col("play_min").sum()).alias(c) for c in ("vision_score", "vs_in_cluster", "vision_gold", "lwrc", "c_gold")])
          .filter(pl.col("g") >= cfg["season"]["min_games_for_ranking"]))
    out["player_corr"] = {}
    for (r,), d in sorted(sp.partition_by("role", as_dict=True).items()):
        a = {c: d[c].to_numpy() for c in ("vision_score", "vs_in_cluster", "vision_gold", "lwrc", "c_gold")}
        out["player_corr"][r] = {"n": d.height,
                                 "vs_pm_vs_vision_gold_pm": partial_corr(a["vision_score"], a["vision_gold"], []),
                                 "vs_pm_vs_lwrc_pm_given_gold": partial_corr(a["vision_score"], a["lwrc"], [a["c_gold"]]),
                                 "out_of_fight_vs_pm_vs_lwrc_pm_given_gold": partial_corr(a["vision_score"] - a["vs_in_cluster"], a["lwrc"], [a["c_gold"]])}
    out["jg_top_vs"] = sp.filter(pl.col("role") == "JG").sort("vision_score", descending=True).head(5).select(
        "name", "team", "g", pl.col("vision_score").round(2), pl.col("vision_gold").round(1)).to_dicts()
    (OUTPUTS / "vision_score_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["by_role"], ensure_ascii=False, indent=1))
    print("교전 중 적립 비율(전체)", round(out["vs_in_fights_all_pct"], 1), "% / 교전 시간 비율", round(out["fight_time_coverage_pct"], 1), "%")


if __name__ == "__main__":
    main()
