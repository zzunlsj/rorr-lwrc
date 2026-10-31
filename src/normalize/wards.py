"""와드 이벤트 테이블 (spec/06 §2, spec/04 §3, 결정 16·17).

ward_events: 설치 1건 = 1행. 생존 구간 끝 = min(제거, 만료, 교체, 게임 종료)
  removal_kind: killed | expired(지속시간 추정) | replaced(설치 한도 초과 추정) | game_end
ward_kills:  제거 이벤트 1건 = 1행 (매칭된 설치 ward_id, 없으면 null) — 시야 제거 피처의 원천

설치↔제거 연결 키가 raw에 없다 → 같은 종류·위치 ±match_xz·설치가 먼저인 미사용 설치 중 가장 최근 것과 매칭.
"""

from bisect import bisect_right


def _duration(ward_type: str, avg_level: float | None, rules: dict) -> float | None:
    d = rules["duration_sec"].get(ward_type)
    if isinstance(d, dict):  # 장신구: 평균 챔피언 레벨 비례 (선형 가정, specs/ward_duration_2026-10-31.md)
        lv = 1.0 if avg_level is None else avg_level
        return d["min"] + (d["max"] - d["min"]) * (min(max(lv, 1), 18) - 1) / 17
    return d


def build_ward_tables(placed: list[dict], killed: list[dict], level_at, team_of: dict[int, int],
                      end_ms: int, rules: dict, vision_types: set[str]) -> tuple[list[dict], list[dict]]:
    """placed/killed: {seq, game_time_ms, ward_type, pid, x, z}. level_at(t_ms) → 그 시각 평균 챔피언 레벨."""
    placed = sorted(placed, key=lambda w: (w["game_time_ms"], w["seq"]))
    wards = []
    for w in placed:
        dur = _duration(w["ward_type"], level_at(w["game_time_ms"]), rules)
        wards.append({
            "ward_id": w["seq"], "ward_type": w["ward_type"], "placer": w["pid"],
            "side": team_of.get(w["pid"]), "placer_mapped": w["pid"] in team_of,
            "vision_eligible": w["ward_type"] in vision_types and w["pid"] in team_of,
            "t_placed_ms": w["game_time_ms"], "x": w["x"], "z": w["z"],
            "duration_s": dur, "t_killed_ms": None, "killer": None, "t_replaced_ms": None,
        })

    # 1) 제거 이벤트 매칭
    tol = rules["match_xz"]
    kills_out = []
    used = set()
    for k in sorted(killed, key=lambda k: (k["game_time_ms"], k["seq"])):
        best = None
        for i, w in enumerate(wards):
            if w["t_placed_ms"] > k["game_time_ms"]:
                break
            if (i in used or w["ward_type"] != k["ward_type"] or w["x"] is None or k["x"] is None
                    or abs(w["x"] - k["x"]) > tol or abs(w["z"] - k["z"]) > tol):
                continue
            best = i
        if best is not None:
            used.add(best)
            wards[best]["t_killed_ms"] = k["game_time_ms"]
            wards[best]["killer"] = k["pid"]
        kills_out.append({
            "seq": k["seq"], "game_time_ms": k["game_time_ms"], "ward_type": k["ward_type"], "killer": k["pid"],
            "killer_side": team_of.get(k["pid"]), "x": k["x"], "z": k["z"],
            "ward_id": wards[best]["ward_id"] if best is not None else None,
        })

    # 2) 설치 한도 교체 — 선수·그룹별로 시간순 재현. 살아 있는 와드 = 아직 제거·만료·교체 전
    def end_before_replace(w):
        ends = [e for e in (w["t_killed_ms"],
                            w["t_placed_ms"] + w["duration_s"] * 1000 if w["duration_s"] is not None else None) if e is not None]
        return min(ends) if ends else None

    group_of, limit = rules["limit_group"], rules["limit"]
    active: dict[tuple, list[dict]] = {}
    for w in wards:
        grp = group_of.get(w["ward_type"])
        if grp is None or not w["placer_mapped"]:
            continue
        key = (w["placer"], grp)
        t = w["t_placed_ms"]
        alive = [a for a in active.get(key, []) if (e := end_before_replace(a)) is None or e > t]
        if len(alive) >= limit[grp]:
            oldest = alive.pop(0)
            oldest["t_replaced_ms"] = t
        alive.append(w)
        active[key] = alive

    # 3) 생존 구간 끝 확정
    for w in wards:
        cands = []
        if w["t_killed_ms"] is not None:
            cands.append((w["t_killed_ms"], "killed"))
        if w["t_replaced_ms"] is not None:
            cands.append((w["t_replaced_ms"], "replaced"))
        if w["duration_s"] is not None:
            cands.append((w["t_placed_ms"] + w["duration_s"] * 1000, "expired"))
        cands.append((end_ms, "game_end"))
        t_end, kind = min(cands, key=lambda c: c[0])
        w["t_removed_ms"] = int(min(t_end, end_ms)) if t_end is not None else end_ms
        w["removal_kind"] = kind
        w["remover"] = w["killer"] if kind == "killed" else None
        w["kill_after_replace"] = w["t_killed_ms"] is not None and w["t_replaced_ms"] is not None and w["t_killed_ms"] > w["t_replaced_ms"]
    return wards, kills_out


def level_lookup(times_ms: list[int], avg_levels: list[float]):
    """정렬된 틱 시각 → 그 시각 이전 마지막 틱의 평균 레벨."""
    def at(t: int) -> float | None:
        i = bisect_right(times_ms, t) - 1
        return avg_levels[i] if i >= 0 else (avg_levels[0] if avg_levels else None)
    return at
