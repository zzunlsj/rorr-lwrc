"""QC 게이트 (CLAUDE.md §6). 경기 × 팀 단위 판정 + 수령 범위 대조."""

import csv

import numpy as np
import polars as pl

from common.paths import MANIFEST
from features.clusters import DRIFT, Cluster

QC_CSV = MANIFEST / "qc_results.csv"


def identity_check(ledger: pl.DataFrame, G: dict[int, np.ndarray], team_gold: dict[int, np.ndarray], eps: float) -> list[dict]:
    """Σ C + Σ Risk + 팀 버킷 = 실제 최종 팀 골드 − G(초기). D는 넣지 않는다."""
    out = []
    for side in G:
        own = ledger.filter((pl.col("side") == side) & pl.col("channel").is_in(["C", "Risk", "team_bucket"]))
        lhs = float(own["amount"].sum())
        rhs = float(team_gold[side][-1] - G[side][0])
        out.append({
            "side": side, "identity_lhs": lhs, "identity_rhs": rhs, "identity_residual": lhs - rhs,
            "g_end_equals_gold": bool(abs(G[side][-1] - team_gold[side][-1]) < 1e-6),
            "identity_ok": bool(abs(lhs - rhs) < eps),
        })
    return out


def partition_check(t: np.ndarray, bucket: np.ndarray, hits: np.ndarray, clusters: list[Cluster], warn_len_sec: float) -> dict:
    dur = np.diff(t)
    in_cluster = bucket != DRIFT
    lens = np.array([(c.end_ms - c.start_ms) / 1000 for c in clusters]) if clusters else np.zeros(1)
    return {
        "n_intervals": int(bucket.size),
        "n_overlap_intervals": int((hits > 1).sum()),
        "n_unassigned_intervals": int(((bucket == DRIFT) & (hits > 0)).sum()),
        "partition_ok": bool((hits <= 1).all()),
        "n_clusters": len(clusters),
        "coverage": float(dur[in_cluster].sum() / dur.sum()),
        "cluster_len_mean_s": float(lens.mean()),
        "cluster_len_median_s": float(np.median(lens)),
        "cluster_len_max_s": float(lens.max()),
        "n_clusters_over_warn": int((lens > warn_len_sec).sum()),
    }


def expected_vs_received() -> pl.DataFrame:
    """CLAUDE.md §6-1: 대회·스테이지별 기대 게임 수 대비 수령 수, 중복."""
    exp = pl.read_csv(MANIFEST / "expected_games.csv")
    rec = pl.read_csv(MANIFEST / "received.csv", schema_overrides={"match_id": pl.Utf8, "game_id": pl.Utf8})
    dup = rec.group_by("game_id").len().filter(pl.col("len") > 1).height
    got = rec.group_by("tournament", "stage").agg(pl.len().alias("received"), pl.col("game_id").n_unique().alias("unique"))
    return (exp.join(got, on=["tournament", "stage"], how="left")
            .with_columns(pl.col("received").fill_null(0), pl.col("unique").fill_null(0))
            .with_columns((pl.col("games_min") - pl.col("unique")).clip(lower_bound=0).alias("missing_min"),
                          pl.lit(dup).alias("duplicate_game_ids")))


def write_qc(rows: list[dict], spec_version: str) -> None:
    """같은 spec_version의 이전 행을 대체하고 나머지는 유지."""
    keep = []
    if QC_CSV.exists():
        with open(QC_CSV, newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["spec_version"] != spec_version]
    fields = list(rows[0].keys())
    for r in keep:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(QC_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep + rows)
