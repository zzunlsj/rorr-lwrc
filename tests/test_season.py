"""Step 6 집계 — 합의 비율, 포지션 분모, 윈저라이즈, 웹 출력 계약 (spec/05 §5.3, contracts/output-web.md)."""

import json

import polars as pl
import pytest

from common.paths import OUTPUTS, ROOT
from metrics.season import lwrc_plus


def test_rate_is_ratio_of_sums_and_denominator_is_role_pool():
    x = pl.DataFrame({"puuid": ["a", "a", "b"], "role": ["MID"] * 3,
                      "play_min": [20.0, 40.0, 30.0], "lwrc": [400.0, 1400.0, 300.0]})
    r = {d["puuid"]: d for d in lwrc_plus(x, "lwrc", sigma=99).to_dicts()}
    assert r["a"]["rate"] == pytest.approx(1800 / 60)            # 게임별 rate 평균(25)이 아니라 합의 비율 30
    assert r["a"]["denom"] == pytest.approx(2100 / 90)
    assert r["a"]["ix_f"] == pytest.approx(100 * 30 / (2100 / 90))
    assert r["b"]["pg_f"] == pytest.approx((10 - 2100 / 90) * 30 / 1)


def test_winsorize_clips_extreme_player_games_within_role():
    x = pl.DataFrame({"puuid": [f"p{i}" for i in range(21)], "role": ["TOP"] * 21,
                      "play_min": [30.0] * 21, "lwrc": [300.0] * 20 + [30000.0]})
    raw = lwrc_plus(x, "lwrc", sigma=99).filter(pl.col("puuid") == "p20")["rate"][0]
    cut = lwrc_plus(x, "lwrc", sigma=3).filter(pl.col("puuid") == "p20")["rate"][0]
    assert cut < raw


WEB = OUTPUTS / "web"


@pytest.mark.skipif(not (WEB / "players.json").exists(), reason="웹 출력 없음")
def test_web_outputs_follow_schema_and_role_average_is_100():
    import jsonschema
    for name in ("players", "teams"):
        body = json.loads((WEB / f"{name}.json").read_text())
        jsonschema.validate(body, json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text()))
    p = pl.DataFrame(json.loads((WEB / "players.json").read_text())["players"])
    avg = p.group_by("scope", "role").agg(((pl.col("ix") * pl.col("minutes")).sum() / pl.col("minutes").sum()).alias("a"))
    assert avg["a"].is_between(99, 101).all()      # 분 가중 평균 = 100 (정수 반올림·윈저라이즈 오차만)


def test_leaguepedia_winner_is_blue_red_side():
    """MatchScheduleGame.Winner(1/2)는 블루/레드다 — raw 승자와 전 게임 일치해야 한다 (team1/team2로 읽으면 약 49%)."""
    from common.paths import L1, MANIFEST
    lp = pl.read_csv(MANIFEST / "leaguepedia_games_2026.csv").filter(pl.col("winner").is_not_null())
    g = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("platform_game_id", "winning_team")
    j = g.join(lp, on="platform_game_id")
    assert j.height > 0
    assert ((j["winner"] == 1) == (j["winning_team"] == 100)).all()


def test_match_results_count_only_matches_complete_in_raw():
    from metrics.season import match_results
    x = pl.DataFrame({  # m1: A 2-0 (완결), m2: 1-1이지만 실제로는 3게임 (raw 미완)
        "game_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
        "match_id": ["m1"] * 4 + ["m2"] * 4,
        "team_code": ["A", "B"] * 4, "team_id": [100, 200] * 4,
        "winning_team": [100, 100, 200, 200, 100, 100, 200, 200],
        "game_start_utc": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-02-01", "2026-02-01", "2026-02-02", "2026-02-02"]})
    x = x.with_columns(pl.when(pl.col("game_id") == "g2").then(pl.when(pl.col("team_code") == "A").then(200).otherwise(100))
                       .otherwise(pl.col("team_id")).alias("team_id"))   # g2: 진영 교체, A가 레드에서 승
    played = pl.DataFrame({"match_id": ["m1", "m2"], "lp_match_id": ["L1", "L2"], "played": [2, 3]})
    res, inc = match_results(x, played)
    r = {d["team"]: d for d in res.to_dicts()}
    assert (r["A"]["W"], r["A"]["L"], r["A"]["form"]) == (1, 0, "W")
    assert (r["B"]["W"], r["B"]["L"], r["B"]["form"]) == (0, 1, "L")
    assert (r["A"]["sw"], r["A"]["sl"]) == (3, 1)          # 게임 전적은 미완 경기의 raw 게임도 센다
    assert [m["lp_match_id"] for m in inc] == ["L2"] and inc[0]["raw_games"] == 2


def test_partial_corr_removes_shared_driver():
    """#10 (결정 31): 골드가 존재·LwRC를 함께 끌어올리는 경우, 골드를 통제하면 그 몫이 빠진다."""
    import numpy as np
    from metrics.season import partial_corr
    rng = np.random.default_rng(1)
    gold = rng.normal(size=500)
    presence = gold + rng.normal(scale=0.5, size=500)                    # 존재 = 골드에 묶인 대리 지표
    lwrc = gold + rng.normal(scale=0.5, size=500)                        # 존재와 무관, 골드만 공유
    assert partial_corr(presence, lwrc, []) > 0.7
    assert abs(partial_corr(presence, lwrc, [gold])) < 0.1
