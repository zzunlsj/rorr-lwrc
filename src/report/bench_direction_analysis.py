"""#10 벤치 방향성 — 서폿·미드 역방향 원인 분석 (2026-11-01 Jun 요청).

이전 #10 지표: 포지션 안에서 rank_gain = 실현 골드 순위 − LwRC 순위, corr(교전 존재 비율, rank_gain).
여기서는 그 지표가 무엇에 흔들리는지 나눠 본다. 이 분석으로 #10은 골드 통제 부분상관으로 바뀌었다 (결정 31, season.bench_direction).
  1. 존재 비율이 실현 골드·승률 자체와 얼마나 묶여 있나 (대리 지표 혼입)
  2. 순위 천장: 골드 순위 1위는 오를 수 없다 → corr(rank_gain, 골드 순위)
  3. 골드(·승률)를 통제한 부분상관 corr(존재, LwRC/분 | 골드/분 [, 승률]) + 부트스트랩 95% 구간
  4. 성분별(C 파이·D·Risk) 부분상관 — "데스가 많아서" 가설 확인
  5. 포지션별 존재·골드 퍼짐

결과: outputs/bench_direction_analysis.json (리포트 specs/bench_direction_analysis_2026-11-01.md의 수치 원본)
사용: PYTHONPATH=src uv run python -m report.bench_direction_analysis
"""

import json

import numpy as np
import polars as pl
import yaml

from common.paths import OUTPUTS
from features.alloc_features import OUT as FEAT_DIR
from ingest.leaguepedia import LWRC_YAML
from metrics.season import load_player_games, partial_corr as pcorr

REPS = 2000


def player_table(min_games: int) -> pl.DataFrame:
    """bench_direction과 같은 선수 표 (전 게임, 최다 포지션, min_games 이상) + 성분·승률."""
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    p = load_player_games(cfg).with_columns((pl.col("team_id") == pl.col("winning_team")).alias("won"))
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=False).select("game_id", "participant_id", "presence")
    pres = p.select("game_id", "participant_id", "puuid").join(feat, on=["game_id", "participant_id"]).group_by("puuid").agg(
        pl.col("presence").mean())
    rate = [(pl.col(c).sum() / pl.col("play_min").sum()).alias(c) for c in ("lwrc", "c_gold", "c_pie", "d", "risk")]
    return (p.group_by("puuid").agg(pl.col("role").mode().first(), pl.len().alias("g"), pl.col("display_name").last().alias("name"),
                                    pl.col("team_code").mode().first().alias("team"), *rate, pl.col("won").mean().alias("wr"))
            .filter(pl.col("g") >= min_games).join(pres, on="puuid")
            .with_columns(pl.col("lwrc").rank(descending=True).over("role").alias("rk_l"),
                          pl.col("c_gold").rank(descending=True).over("role").alias("rk_g"))
            .with_columns((pl.col("rk_g") - pl.col("rk_l")).alias("gain")))


def boot_ci(a: np.ndarray, b: np.ndarray, ctrl: list[np.ndarray], rng) -> list[float]:
    n, v = len(a), []
    for _ in range(REPS):
        i = rng.integers(0, n, n)
        v.append(pcorr(a[i], b[i], [c[i] for c in ctrl]))
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    s = player_table(cfg["season"]["min_games_for_ranking"])
    rng, out = np.random.default_rng(0), {}
    for (role,), d in sorted(s.partition_by("role", as_dict=True).items()):
        a = {c: d[c].to_numpy().astype(float) for c in ("presence", "lwrc", "c_gold", "c_pie", "d", "risk", "wr", "gain", "rk_g")}
        pr, g, wr = a["presence"], a["c_gold"], a["wr"]
        out[role] = {
            "n": d.height,
            "current_metric": {"r": pcorr(pr, a["gain"], []), "ci": boot_ci(pr, a["gain"], [], rng)},
            "presence_vs": {c: pcorr(pr, a[c], []) for c in ("c_gold", "wr", "lwrc")},
            "gain_vs_gold_rank": pcorr(a["gain"], a["rk_g"], []),
            "partial_lwrc_given_gold": {"r": pcorr(pr, a["lwrc"], [g]), "ci": boot_ci(pr, a["lwrc"], [g], rng)},
            "partial_lwrc_given_gold_wr": {"r": pcorr(pr, a["lwrc"], [g, wr]), "ci": boot_ci(pr, a["lwrc"], [g, wr], rng)},
            "partial_components_given_gold": {c: pcorr(pr, a[c], [g]) for c in ("c_pie", "d", "risk")},
            "spread": {"presence_mean": float(pr.mean()), "presence_sd": float(pr.std(ddof=1)),
                       "gold_rate_sd": float(g.std(ddof=1)), "lwrc_rate_sd": float(a["lwrc"].std(ddof=1))},
            "top_presence": d.sort("presence", descending=True).head(6).select(
                "name", "team", "g", pl.col("presence").round(3), pl.col("wr").round(2), "rk_g", "rk_l", "gain").to_dicts(),
        }
    (OUTPUTS / "bench_direction_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for role, r in out.items():
        print(f"{role} n={r['n']} 이전지표 {r['current_metric']['r']:+.2f} | 존재~골드 {r['presence_vs']['c_gold']:+.2f} 존재~승률 {r['presence_vs']['wr']:+.2f}"
              f" | 부분(골드) {r['partial_lwrc_given_gold']['r']:+.2f} {[round(x, 2) for x in r['partial_lwrc_given_gold']['ci']]}"
              f" | 부분(골드·승률) {r['partial_lwrc_given_gold_wr']['r']:+.2f} {[round(x, 2) for x in r['partial_lwrc_given_gold_wr']['ci']]}")


if __name__ == "__main__":
    main()
