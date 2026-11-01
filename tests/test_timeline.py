"""Step 3 타임라인 — 클러스터 병합(결정 25), 틱 분할, 결정적 감쇠(결정 28), 항등식·D 교차검증 (spec/02, spec/03)."""

import numpy as np
import polars as pl
import pytest

from common.paths import L2
from features.timeline import build_clusters, decay_contrib, drift_spans, tick_spans


def test_window_gap_merges_seeds_up_to_30s_apart():
    k, spans = build_clusters(np.array([100_000, 130_000, 160_001.0]), 0, 1e7, 10, 10, "window_gap")
    assert list(k) == [0, 0, 1]                       # 창 간격 10초 = 씨앗 간격 30초까지 병합
    assert spans == [(90_000, 140_000), (150_001, 170_001)]


def test_clusters_are_clipped_at_game_bounds():
    _, spans = build_clusters(np.array([3_000.0, 995_000.0]), 1_000, 1_000_000, 10, 10, "window_gap")
    assert spans == [(1_000, 13_000), (985_000, 1_000_000)]


def test_unknown_merge_rule_stops():
    with pytest.raises(ValueError):
        build_clusters(np.array([1.0]), 0, 10, 10, 10, "seed_gap")


def test_clusters_and_drift_partition_all_tick_intervals():
    t = np.arange(0, 600_001, 1000.0)
    _, spans = build_clusters(np.array([5_000, 100_000, 125_000, 590_000.0]), t[0], t[-1], 10, 10, "window_gap")
    cl = tick_spans(t, spans)
    covered = np.zeros(t.size - 1, dtype=int)
    for a, b in cl + drift_spans(t.size, cl):
        covered[a:b] += 1
    assert (covered == 1).all()                        # 모든 틱 구간이 정확히 한 버킷


def test_decay_counts_only_countdown_and_respawn():
    x = pl.DataFrame({"respawn_sum_us": [0.0, 20, 19, 18, 0], "alive_us": [5.0, 4, 4, 4, 5]})
    beta = {"respawn_sum_us": -20.0, "alive_us": 40.0}
    d = decay_contrib(x, beta, {"decreasing": ["respawn_sum_us"], "increasing": ["alive_us"]})
    # 데스 순간의 점프(+20초, 생존 −1)는 Risk 몫이라 빠지고, 이후 감소 1+1+18초와 부활 +1만 남는다
    assert d.tolist() == [0, 0, 20, 40, 400 + 40]


TL = L2 / "timeline"


@pytest.mark.skipif(not (TL / "summary").exists(), reason="타임라인 산출 없음")
def test_identity_and_d_cross_check_hold_for_all_games():
    s = pl.read_parquet(TL / "summary" / "**" / "*.parquet", hive_partitioning=False)
    assert s["identity_resid"].abs().max() < 1e-6
    assert s["d_cross_resid"].abs().max() < 1e-6
    assert (s["n_deaths_outside"] == 0).all()


@pytest.mark.skipif(not (TL / "deaths").exists(), reason="타임라인 산출 없음")
def test_risk_is_never_positive():
    d = pl.read_parquet(TL / "deaths" / "**" / "*.parquet", hive_partitioning=False)
    assert (d["risk"] <= 0).all()
