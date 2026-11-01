"""실제 골드 비교 표 — 게임 내 totalGold만으로 합의 비율 · 포지션 분모 (metrics.gold)."""

import polars as pl
import pytest

from metrics.gold import scope_players, scope_teams


def _games() -> pl.DataFrame:
    # 2게임, 각 팀 MID 1명만 (팀 골드는 따로 관측값)
    return pl.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"], "puuid": ["a", "b", "a", "b"], "role": ["MID"] * 4,
        "team_code": ["AAA", "BBB", "AAA", "BBB"], "league": ["LCK"] * 4, "team_id": [100, 200, 100, 200],
        "winning_team": [100, 100, 200, 200], "play_min": [20.0, 20.0, 40.0, 40.0],
        "gold": [10000.0, 8000.0, 18000.0, 20000.0], "team_gold": [50000.0, 42000.0, 90000.0, 95000.0],
        "opp_team_gold": [42000.0, 50000.0, 95000.0, 90000.0], "gd_lane": [500.0, -500.0, None, None]})


def test_gold_plus_is_ratio_of_sums_against_role_pool():
    r = {d["puuid"]: d for d in scope_players(_games()).to_dicts()}
    den = 56000 / 120
    assert r["a"]["gpm"] == pytest.approx(28000 / 60)            # 게임별 GPM 평균이 아니라 합의 비율
    assert r["a"]["gold_plus"] == pytest.approx(100 * (28000 / 60) / den)
    assert r["b"]["gold_pg"] == pytest.approx((28000 / 60 - den) * 60 / 2)
    assert r["a"]["gold_share"] == pytest.approx(100 * 28000 / 140000)
    assert r["a"]["gd15"] == 500 and r["a"]["g15"] == 1          # 15분 전 종료 게임은 결측 → 평균에서 빠짐


def test_team_gold_plus_and_end_diff():
    x = _games()
    t = {d["team_code"]: d for d in scope_teams(x, scope_players(x)).to_dicts()}
    assert t["AAA"]["gpm"] == pytest.approx(140000 / 60)
    assert t["AAA"]["gold_plus"] == pytest.approx(100 * (140000 / 60) / (277000 / 120))
    assert t["AAA"]["gd_end"] == pytest.approx((8000 - 5000) / 2)
    assert t["AAA"]["wr"] == pytest.approx(0.5)
