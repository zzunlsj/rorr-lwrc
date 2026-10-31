"""CLAUDE.md §5 전처리 규칙 8개 — 규칙마다 단위 테스트(합성 입력) + 실데이터 통합 테스트."""

import pytest

from normalize import rules


# 규칙 1 — gameTime ms, 비시간순 행 정렬
def test_rule1_sorts_unordered_rows_by_game_time():
    rows = [{"gameTime": 3000, "sequenceIndex": 9}, {"gameTime": 1000, "sequenceIndex": 2}, {"gameTime": 2000, "sequenceIndex": 5}]
    assert [r["gameTime"] for r in rules.sort_by_game_time(rows)] == [1000, 2000, 3000]


def test_rule1_ties_broken_by_sequence_index():
    rows = [{"gameTime": 1000, "sequenceIndex": 7}, {"gameTime": 1000, "sequenceIndex": 3}]
    assert [r["sequenceIndex"] for r in rules.sort_by_game_time(rows)] == [3, 7]


def test_rule1_game_time_unit_is_ms():
    # 30분 경기 = 1800틱(1Hz) = 1,800,000ms
    assert rules.game_time_is_ms(1_800_000, 1800)
    assert not rules.game_time_is_ms(1800, 1800)  # 초 단위였다면 실패해야 한다


# 규칙 2 — 에픽 판정 필터
@pytest.mark.parametrize("camp", sorted(rules.JUNGLE_CAMP_TYPES))
def test_rule2_jungle_camps_are_not_epic(camp):
    assert not rules.is_epic(camp)


@pytest.mark.parametrize("epic", ["dragon", "baron", "riftHerald", "VoidGrub"])
def test_rule2_epic_monsters(epic):
    assert rules.is_epic(epic)


@pytest.mark.parametrize("wrong", ["horde", "voidgrub", "riftherald", "Atakhan"])
def test_rule2_misspelled_names_are_not_epic_and_are_rejected(wrong):
    # 공허 유충은 원본 표기 VoidGrub만 인정 (대소문자 구분). 다른 표기는 조용히 빠지지 않고 오류
    assert not rules.is_epic(wrong)
    with pytest.raises(ValueError, match=wrong):
        rules.check_monster_types(["dragon", wrong])


def test_rule2_known_names_pass_check():
    rules.check_monster_types(sorted(rules.KNOWN_MONSTER_TYPES))


# 규칙 3 — 장로 판정과 영혼 스택
def test_rule3_elder_is_dragon_with_elder_type():
    assert rules.is_elder("dragon", "elder")
    assert not rules.is_elder("dragon", "water")  # 바다 드래곤 raw 값은 'water'
    assert not rules.is_elder("baron", None)


def test_rule3_soul_stacks_exclude_elder():
    # teams.dragonKills=5 (원소 4 + 장로 1) → 영혼 스택 4
    assert rules.soul_dragon_stacks(5, 1) == 4


# 규칙 4 — deathRecap list 파싱
RECAP = [
    {"source": "Champion", "casterId": 3, "breakdown": [
        {"physicalDmg": 100, "magicDmg": 50, "trueDmg": 10},
        {"physicalDmg": 0, "magicDmg": 40, "trueDmg": 0},
    ]},
    {"source": "Minion", "casterId": 0, "breakdown": [{"physicalDmg": 999, "magicDmg": 0, "trueDmg": 0}]},
    {"source": "Turret", "casterId": 0, "breakdown": [{"physicalDmg": 500, "magicDmg": 0, "trueDmg": 0}]},
    {"source": "Champion", "casterId": 4, "breakdown": [{"physicalDmg": 300, "magicDmg": 0, "trueDmg": 25}]},
    {"source": "Champion", "casterId": 3, "breakdown": [{"physicalDmg": 0, "magicDmg": 0, "trueDmg": 5}]},
]


def test_rule4_death_recap_champion_only_summed_per_caster():
    assert rules.parse_death_recap(RECAP) == {3: 205.0, 4: 325.0}


def test_rule4_death_recap_dict_is_rejected():
    with pytest.raises(TypeError):
        rules.parse_death_recap({"source": "Champion", "casterId": 1})


def test_rule4_death_recap_missing_is_empty():
    assert rules.parse_death_recap(None) == {}
    assert rules.parse_death_recap([]) == {}


# 규칙 5 — bounty 가변, shutdownBounty 별도
def test_rule5_bounty_and_shutdown_are_separate_fields():
    assert rules.parse_kill_gold({"bounty": 109, "shutdownBounty": 150}) == (109.0, 150.0)
    assert rules.parse_kill_gold({"bounty": 420}) == (420.0, 0.0)


def test_rule5_bounty_range():
    assert rules.bounty_in_range(109) and rules.bounty_in_range(420)
    assert not rules.bounty_in_range(300 + 500)


# 규칙 6 — 팀 골드는 teams[].totalGold 관측값
def test_rule6_team_gold_uses_observed_team_total_not_participant_sum():
    payload = {
        "teams": [{"teamID": 100, "totalGold": 50_000}, {"teamID": 200, "totalGold": 48_000}],
        "participants": [{"teamID": 100, "totalGold": 1}] * 5 + [{"teamID": 200, "totalGold": 1}] * 5,
    }
    assert rules.team_gold(payload) == {100: 50_000.0, 200: 48_000.0}


# 규칙 7 — 1Hz 연속성
def test_rule7_regular_1hz_passes():
    s = rules.tick_interval_stats([i * 1000 for i in range(600)])
    assert s["ok"] and s["mean_s"] == pytest.approx(1.0) and s["n_gaps"] == 0


def test_rule7_gap_fails():
    times = [i * 1000 for i in range(300)] + [300_000 + 5000 + i * 1000 for i in range(300)]
    s = rules.tick_interval_stats(times)
    assert not s["ok"] and s["n_gaps"] == 1 and s["max_s"] == pytest.approx(6.0)


def test_rule7_unsorted_input_is_sorted_first():
    assert rules.tick_interval_stats([2000, 0, 1000])["max_s"] == pytest.approx(1.0)


# 규칙 8 — ward/item 이벤트는 피드에 존재
def test_rule8_feed_event_list():
    assert set(rules.FEED_EVENTS_PRESENT) == {"ward_placed", "ward_killed", "item_purchased"}
