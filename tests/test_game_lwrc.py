"""Step 5 배분·항등식 — 지분 정규화, Σw = 0 처리 (spec/03 §3.3), 전 게임 항등식 (spec/05 §5.1)."""

import polars as pl
import pytest

from common.paths import L3
from metrics.game_lwrc import allocate


def test_allocate_normalizes_within_team_and_routes_zero_weight():
    clusters = pl.DataFrame({"k": [0, 0, 1, 1], "side": [100, 200, 100, 200],
                             "c_pie": [100.0, 0, 50, 0], "d_pie": [0.0, 30, 0, 20]})
    weights = pl.DataFrame({"k": [0, 0, 0, 1, 1, 1], "participant_id": [1, 2, 6, 1, 2, 6], "team_id": [100, 100, 200, 100, 100, 200],
                            "w": [3.0, 1.0, 0.0, 0.0, 0.0, 2.0]})
    player, team = allocate(clusters, weights, "w", "c_pie", "d_pie")
    p = dict(zip(player["participant_id"], zip(player["c_pie"], player["d"])))
    assert p[1] == pytest.approx((75, 0)) and p[2] == pytest.approx((25, 0))   # k0: 3:1 지분
    assert p[6] == pytest.approx((0, 20))                                       # k1: 200팀 혼자 관여 → D 20
    t = dict(zip(team["team_id"], zip(team["c_unalloc"], team["d_unattributed"])))
    assert t[100] == pytest.approx((50, 0))     # k1 100팀 Σw = 0 → C 파이는 팀 계정으로
    assert t[200] == pytest.approx((0, 30))     # k0 200팀 Σw = 0 → D 파이는 D_unattributed로 (팀 계정과 섞지 않음)


T = L3 / "game_team"


@pytest.mark.skipif(not T.exists(), reason="Step 5 산출 없음")
def test_identity_holds_for_every_game_and_variant():
    t = pl.read_parquet(T / "**" / "*.parquet", hive_partitioning=False)
    assert t["identity_pass"].all()
    assert set(t["variant"].unique()) == {"main", "prior_w", "legacy", "prior_g"}
