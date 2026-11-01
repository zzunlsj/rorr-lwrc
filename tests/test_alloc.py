"""Step 4 배분 — 흡수 측정, NNLS 리지(β ≥ 0, 통제항 투영), prior 가중 폴백 (spec/04)."""

import numpy as np
import polars as pl
import pytest

from common.paths import L2
from features.alloc_features import absorb_steps, near_any
from features.alloc_train import NNLSRidge, prior_weights


def test_absorb_skips_max_health_change_and_respawn():
    health = np.array([500, 400, 450, 300, 0, 600.0])
    hmax = np.array([600, 600, 650, 650, 650, 650.0])
    alive = np.array([True, True, True, True, False, True])
    # 500→400 흡수 100 / 400→450 레벨업(최대 체력 변화) 제외 / 450→300 흡수 150 / 300→0 데스 흡수 300 / 부활 제외
    assert absorb_steps(health, hmax, alive).tolist() == [100, 0, 150, 300, 0]


def test_near_any_is_or_over_seeds():
    hit = near_any(np.array([0, 5000, 9000.0]), np.zeros(3), np.array([0, 9100.0]), np.zeros(2), 2300)
    assert hit.tolist() == [True, False, True]


def test_nnls_ridge_recovers_nonnegative_betas_and_zeroes_negative():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3)) * [100, 1, 10]
    c = rng.normal(size=(4000, 1))
    y = 0.5 * x[:, 0] + 30 * x[:, 1] - 2 * x[:, 2] + 50 * c[:, 0] + rng.normal(size=4000)
    m = NNLSRidge().fit(x, c, y, alpha=0.01)
    assert m.coef_[0] == pytest.approx(0.5, rel=0.02) and m.coef_[1] == pytest.approx(30, rel=0.02)
    assert m.coef_[2] == 0.0                         # 음수 효과는 0으로 (관여했는데 지분이 깎이지 않게)


def test_prior_weights_use_team_shares_and_kill_fallback():
    feat = pl.DataFrame({
        "game_id": "g", "k": 0, "team_id": [100, 100, 200, 200],
        "dmg": [300.0, 100, 0, 0], "assist": [0.0, 1, 0, 0], "absorb": [0.0, 0, 0, 0], "presence": [0.0, 0, 0, 0],
        "kill_killer": [1, 0, 0, 1],
    })
    w = prior_weights(feat, {"dmg": 1.0, "assist": 0.5, "absorb": 0.3, "presence": 0.2}).to_list()
    assert w[:2] == pytest.approx([0.75, 0.25 + 0.5])   # 딜 점유율 0.75/0.25 + 어시스트 점유율 0/1 × 0.5
    assert w[2:] == pytest.approx([0.0, 1.0])           # 200팀: 관여 피처가 전부 0인데 킬 → 폴백 killer 1.0


W = L2 / "alloc" / "weights"


@pytest.mark.skipif(not W.exists(), reason="배분 가중치 없음")
def test_weights_are_nonnegative_and_cover_every_player_cluster():
    w = pl.read_parquet(W / "**" / "*.parquet", hive_partitioning=False)
    assert (w["w_reg"] >= 0).all() and (w["w_prior"] >= 0).all()
    assert w.group_by("game_id", "k").len()["len"].unique().to_list() == [10]
