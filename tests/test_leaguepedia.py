"""결정 18 — Leaguepedia 게임 → (tournament, stage) 분류 규칙 (config/lwrc.yaml stage_map)."""

import pytest
import yaml

from ingest.leaguepedia import LWRC_YAML, classify

CFG = yaml.safe_load(LWRC_YAML.read_text())["stage_map"]
GROUPS = {("LCK/2026 Season/Rounds 3-4", "T1"): "Legend Group",
          ("LCK/2026 Season/Rounds 3-4", "Gen.G"): "Legend Group",
          ("LCK/2026 Season/Rounds 3-4", "BNK FEARX"): "Rise Group"}


def game(overview, tab, team1="A", team2="B"):
    return {"overview_page": overview, "tab": tab, "team1": team1, "team2": team2, "lp_match_id": f"{overview}_{tab}"}


def run(g):
    return classify(g, CFG["rules"], GROUPS, CFG["group_stage"])


@pytest.mark.parametrize("overview,tab,expected", [
    ("LCK/2026 Season/Cup", "Week 3", ("LCK_Cup", "Group")),
    ("LCK/2026 Season/Cup", "Play-In Round 2", ("LCK_Cup", "PlayIn")),
    ("LCK/2026 Season/Cup", "Finals", ("LCK_Cup", "Playoffs")),
    ("2026 First Stand", "Groups Day 5", ("FirstStand", "Group")),
    ("2026 First Stand", "Semifinals", ("FirstStand", "Knockout")),
    ("2026 Mid-Season Invitational", "Play-In Day 1", ("MSI", "PlayIn")),
    ("2026 Mid-Season Invitational", "Bracket Round 4", ("MSI", "Bracket")),
    ("2026 Season World Championship/Main Event", "Round 5", ("Worlds", "Swiss")),
    ("2026 Season World Championship/Main Event", "Quarterfinals", ("Worlds", "Knockout")),
    ("LCK/2026 Season/Season Playoffs", "Round 1", ("LCK_PlayIn_PO", "Playoffs")),
])
def test_tab_rules(overview, tab, expected):
    assert run(game(overview, tab)) == expected


def test_r34_stage_from_team_group():
    assert run(game("LCK/2026 Season/Rounds 3-4", "Week 11", "T1", "Gen.G")) == ("LCK_R3-4", "Legend")


def test_r34_mixed_groups_stops():
    with pytest.raises(ValueError, match="그룹 판정 불가"):
        run(game("LCK/2026 Season/Rounds 3-4", "Week 11", "T1", "BNK FEARX"))


def test_unknown_page_stops():
    with pytest.raises(ValueError, match="규칙에 없는"):
        run(game("2026 Asian Games", "Day 1"))
