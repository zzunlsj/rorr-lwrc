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
