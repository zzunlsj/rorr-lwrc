"""10게임 리허설 L1 통합 테스트. data/l1이 없으면 건너뛴다 (data/는 커밋하지 않음).

기대값 출처: Notion "콜드스타트 설정값 v0.1" §0, §7 (같은 CSV의 실측).
L1에는 이제 678게임이 있으므로, 리허설 10게임(CSV 추출본으로 처음 받은 게임)으로 범위를 좁혀 같은 기대값을 확인한다.
"""

import csv
import re

import pytest

from common.db import connect
from common.paths import L1, MANIFEST

pytestmark = pytest.mark.skipif(not (L1 / "games").exists(), reason="L1 데이터 없음")


def rehearsal_game_ids() -> list[str]:
    with open(MANIFEST / "received.csv", newline="") as f:
        return sorted(r["game_id"] for r in csv.DictReader(f) if (r.get("raw_format") or "csv_extract") == "csv_extract")


def q(sql: str):
    ids = ",".join(f"'{g}'" for g in rehearsal_game_ids())
    sql = re.sub(r"read_parquet\('\{L1\}/([a-z_]+)/\*\*/\*\.parquet'\)",
                 lambda m: f"(SELECT * FROM read_parquet('{{L1}}/{m[1]}/**/*.parquet') WHERE game_id IN ({ids}))", sql)
    return connect().execute(sql.replace("{L1}", L1.as_posix())).fetchall()


def test_ten_games_all_ticks_ok_and_ms():
    rows = q("SELECT count(*), bool_and(tick_ok), bool_and(game_time_is_ms) FROM read_parquet('{L1}/games/**/*.parquet')")
    assert rows == [(10, True, True)]


def test_rule2_epic_counts_match_coldstart_doc():
    rows = dict(q("SELECT monster_type, count(*) FROM read_parquet('{L1}/monsters/**/*.parquet') WHERE is_epic GROUP BY 1"))
    assert rows == {"dragon": 46, "VoidGrub": 30, "baron": 15, "riftHerald": 10}


def test_rule3_elder_present_and_flagged():
    (n,), = q("SELECT count(*) FROM read_parquet('{L1}/monsters/**/*.parquet') WHERE is_elder")
    assert n == 1


def test_rule4_kill_damage_not_collapsed_to_single_caster():
    # 잘못 파싱하면 킬당 caster가 1명으로 몰린다 — 평균 caster 수가 1보다 충분히 커야 한다
    (avg_casters,), = q("""SELECT avg(n) FROM (SELECT game_id, kill_seq, count(*) n
                           FROM read_parquet('{L1}/kill_damage/**/*.parquet') GROUP BY ALL)""")
    assert avg_casters > 1.5


def test_rule5_bounty_range_matches_doc():
    rows = q("SELECT count(*), min(bounty), max(bounty), round(avg(bounty)) FROM read_parquet('{L1}/kills/**/*.parquet')")
    assert rows == [(253, 109.0, 420.0, 332.0)]


def test_rule6_team_gold_close_to_participant_sum():
    (max_diff,), = q("""
        SELECT max(abs(t.total_gold - p.s)) FROM read_parquet('{L1}/frames_team/**/*.parquet') t
        JOIN (SELECT game_id, game_time_ms, team_id, sum(total_gold) s
              FROM read_parquet('{L1}/frames_player/**/*.parquet') GROUP BY ALL) p USING (game_id, game_time_ms, team_id)""")
    # 관측 팀 골드를 그대로 쓰되, 개인 합과 크게 어긋나면 파싱 이상
    assert max_diff < 1000


def test_rule7_tick_stats_match_doc():
    rows = q("SELECT round(max(tick_p95_s),3), round(max(tick_max_s),2) FROM read_parquet('{L1}/games/**/*.parquet')")
    assert rows == [(1.003, 1.03)]


def test_rule8_ward_and_item_events_present():
    rows = dict(q("""SELECT schema, sum(n) FROM read_parquet('{L1}/event_counts/**/*.parquet')
                     WHERE schema IN ('ward_placed','ward_killed','item_purchased') GROUP BY 1"""))
    assert rows == {"ward_placed": 4202, "ward_killed": 1180, "item_purchased": 3163}
