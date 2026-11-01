"""Step 2 상태 테이블 — 레이블(spec/01 §1.1), 버프 잔여, 관점 대칭 (contracts/input.md §2)."""

import numpy as np
import polars as pl
import pytest
import yaml

from common.paths import L2
from features.state_table import FEATURES, MIRROR_PAIRS, _buff_left, labels
from ingest.leaguepedia import LWRC_YAML


def test_label_inside_horizon_is_observed_gold():
    t = np.arange(0, 1_801_000, 1000.0)            # 30분 게임, 1Hz
    gold = 2500 + t / 1000 * 20                       # 초당 20골드
    y, rate = labels(t, gold, end_ms=1_800_000, horizon_min=10, rate_window_min=3)
    assert y[0] == gold[600]                          # t=0 → 10분 뒤 관측 골드
    assert rate == pytest.approx(20 * 60)             # 분당 수입


def test_label_after_end_extrapolates_with_end_rate():
    t = np.arange(0, 1_801_000, 1000.0)
    gold = 2500 + t / 1000 * 20
    y, rate = labels(t, gold, end_ms=1_800_000, horizon_min=10, rate_window_min=3)
    # 종료 5분 전: 5분은 관측, 5분은 직전 수입으로 외삽 → 직선이 그대로 이어진다
    i = 1_500
    assert y[i] == pytest.approx(gold[-1] + rate * 5)
    assert y[-1] == pytest.approx(gold[-1] + rate * 10)  # G(END) := y(END)


def test_buff_left_counts_down_and_resets_on_new_kill():
    t = np.array([0, 10_000, 100_000, 200_000, 250_000], dtype=float)
    left = _buff_left(np.array([10_000.0, 200_000.0]), t, 180)
    assert list(left) == [0, 180, 90, 180, 130]


def test_mirror_pairs_cover_all_side_features():
    paired = {f for pair in MIRROR_PAIRS for f in pair}
    assert paired == set(FEATURES) - {"t_min", "t_min2", "level_diff", "side_blue"}


STATE = L2 / "state"


@pytest.mark.skipif(not STATE.exists(), reason="상태 테이블 없음")
def test_two_sides_are_mirror_images():
    path = next(STATE.glob("**/*.parquet"))
    d = pl.read_parquet(path, hive_partitioning=False)
    a = d.filter(pl.col("side") == 100).sort("t_ms")
    b = d.filter(pl.col("side") == 200).sort("t_ms")
    for x, y in MIRROR_PAIRS:
        assert (a[x] == b[y]).all(), (x, y)
    assert (a["level_diff"] == -b["level_diff"]).all()
    assert (a["side_blue"] + b["side_blue"] == 1).all()
