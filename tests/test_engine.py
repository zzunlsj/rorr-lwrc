"""G·클러스터·귀속 엔진 테스트. 합성 입력 단위 테스트 + 리허설 L3 통합 테스트."""

import numpy as np
import pytest

from common.db import connect
from common.paths import L3, MANIFEST
from features.clusters import DRIFT, Seed, assign_intervals, build_clusters
from features.g_models import RmTableG


def _seeds(*times):
    return [Seed(t, "kill", 100, 0.0, 0.0, 1, []) for t in times]


def test_window_gap_merges_seeds_up_to_30s_apart():
    cl = build_clusters(_seeds(100_000, 125_000, 200_000), 0, 1_000_000, 10, 10, "window_gap")
    assert [(c.start_ms, c.end_ms) for c in cl] == [(90_000, 135_000), (190_000, 210_000)]


def test_clusters_clipped_at_game_bounds():
    cl = build_clusters(_seeds(3_000, 995_000), 0, 1_000_000, 10, 10, "window_gap")
    assert cl[0].start_ms == 0 and cl[-1].end_ms == 1_000_000


def test_interval_partition_is_exact_with_window_gap():
    t = np.arange(0, 300_001, 1000, dtype=float)
    cl = build_clusters(_seeds(50_000, 75_000, 140_000, 151_000), 0, 300_000, 10, 10, "window_gap")
    bucket, hits = assign_intervals(t, cl)
    assert (hits <= 1).all()
    assert bucket.size == t.size - 1
    # 각 클러스터 길이 = 중점 기준 구간 수 × 1초
    for c in cl:
        assert (bucket == c.k).sum() == pytest.approx((c.end_ms - c.start_ms) / 1000, abs=1)
    assert ((bucket == DRIFT) | (bucket >= 0)).all()


def test_seed_gap_rule_produces_overlap_that_qc_detects():
    t = np.arange(0, 100_001, 1000, dtype=float)
    cl = build_clusters(_seeds(40_000, 55_000), 0, 100_000, 10, 10, "seed_gap")  # 15초 간격 → 두 클러스터, 창 겹침
    _, hits = assign_intervals(t, cl)
    assert (hits > 1).any()


def test_rm_table_end_collapse_and_interpolation():
    g = RmTableG([1000.0, 500.0, 0.0])
    assert g.residual(np.array([30_000.0]))[0] == pytest.approx(750.0)

    class S:  # 최소 상태
        t = np.array([0.0, 30_000.0, 60_000.0])
        team_gold = {100: np.array([500.0, 700.0, 900.0])}

    G = g.evaluate(S, 100)
    assert G[-1] == 900.0  # 마지막 틱: G = 실제 팀 골드


L3_DIR = L3 / "lwrc"


@pytest.mark.skipif(not (MANIFEST / "qc_results.csv").exists() or not L3_DIR.exists(), reason="L3 없음")
@pytest.mark.parametrize("spec", ["v0.1-coldstart", "v1.0-draft"])
def test_rehearsal_identity_and_partition(spec):
    rows = connect().execute(f"""
        SELECT count(*), max(abs(identity_residual)), bool_and(partition_ok), bool_and(g_end_equals_gold)
        FROM read_csv('{(MANIFEST / 'qc_results.csv').as_posix()}') WHERE spec_version = '{spec}'""").fetchone()
    n, resid, part, gend = rows
    assert n == 20 and resid < 1.0 and part and gend
