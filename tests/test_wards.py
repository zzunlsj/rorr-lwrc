"""spec/06 §2 와드 이벤트 테이블 — 매칭·만료·교체 규칙 (결정 16·17)."""

import yaml

from ingest.leaguepedia import LWRC_YAML
from normalize.wards import build_ward_tables

CFG = yaml.safe_load(LWRC_YAML.read_text())["allocation"]
RULES, VISION = CFG["ward_rules"], set(CFG["vision_ward_types"])
TEAM = {5: 100, 10: 200}


def ward(seq, t_s, wtype, pid=5, x=1000, z=1000):
    return {"seq": seq, "game_time_ms": int(t_s * 1000), "ward_type": wtype, "pid": pid, "x": x, "z": z}


def run(placed, killed=(), level=1.0, end_s=3000):
    wards, kills = build_ward_tables(list(placed), list(killed), lambda t: level, TEAM, end_s * 1000, RULES, VISION)
    return {w["ward_id"]: w for w in wards}, kills


def test_kill_matches_latest_same_type_nearby_placement():
    w, k = run([ward(1, 10, "control"), ward(2, 20, "control", x=5000), ward(3, 30, "control", x=1020)],
               [ward(9, 40, "control", pid=10)])
    assert k[0]["ward_id"] == 3
    assert (w[3]["removal_kind"], w[3]["t_removed_ms"], w[3]["remover"]) == ("killed", 40_000, 10)


def test_trinket_expires_by_average_level():
    w, _ = run([ward(1, 100, "yellowTrinket")], level=18)
    assert (w[1]["removal_kind"], w[1]["t_removed_ms"]) == ("expired", 220_000)  # 120초
    w, _ = run([ward(1, 100, "yellowTrinket")], level=1)
    assert w[1]["t_removed_ms"] == 190_000  # 90초


def test_fourth_stealth_ward_replaces_oldest_alive():
    placed = [ward(1, 0, "sight"), ward(2, 10, "sight"), ward(3, 20, "yellowTrinket"), ward(4, 30, "sight")]
    w, _ = run(placed)
    assert (w[1]["removal_kind"], w[1]["t_removed_ms"]) == ("replaced", 30_000)
    assert w[2]["removal_kind"] == "expired"


def test_expired_ward_does_not_count_toward_limit():
    placed = [ward(1, 0, "sight"), ward(2, 10, "sight"), ward(3, 20, "sight"), ward(4, 200, "sight")]
    w, _ = run(placed)  # 1번은 150초에 이미 만료 → 교체 없음
    assert all(w[i]["removal_kind"] == "expired" for i in (1, 2, 3))


def test_second_control_ward_replaces_first():
    w, _ = run([ward(1, 0, "control"), ward(2, 60, "control", x=3000)])
    assert (w[1]["removal_kind"], w[1]["t_removed_ms"]) == ("replaced", 60_000)
    assert w[2]["removal_kind"] == "game_end"


def test_unknown_and_unmapped_are_not_vision_eligible():
    w, _ = run([ward(1, 0, "unknown"), ward(2, 0, "control", pid=0)])
    assert not w[1]["vision_eligible"] and not w[2]["vision_eligible"]
    assert w[2]["placer_mapped"] is False and w[2]["removal_kind"] == "game_end"  # 미매핑은 교체 재현에서 빠진다
