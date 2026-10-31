"""확정 전처리 규칙 (CLAUDE.md §5). 규칙마다 tests/test_rules.py에 테스트가 있다.

L1 변환(build_l1)과 이후 단계는 이 모듈의 함수만 써서 규칙을 적용한다.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import yaml

from common.paths import ROOT

LWRC_YAML = ROOT / "config" / "lwrc.yaml"

# 규칙 1 — gameTime은 ms, 행은 시간순이 아니다
GAME_TIME_UNIT_MS = 1000


def sort_by_game_time(rows: Iterable[dict], key: str = "gameTime", tiebreak: str = "sequenceIndex") -> list[dict]:
    return sorted(rows, key=lambda r: (r[key], r.get(tiebreak, 0)))


def game_time_is_ms(end_game_time: float, n_ticks: int, tol: float = 0.1) -> bool:
    """1Hz 틱 수와 종료 gameTime/1000이 맞으면 ms 단위다."""
    if n_ticks <= 0:
        return False
    return abs(end_game_time / GAME_TIME_UNIT_MS / n_ticks - 1.0) <= tol


# 규칙 2 — epic_monster_kill에는 일반 정글 캠프가 섞여 있다.
# 이름 목록의 정본은 config/lwrc.yaml (원본 monsterType 표기, 대소문자 구분. 공허 유충 = VoidGrub)
def _load_monster_names() -> tuple[frozenset[str], frozenset[str]]:
    with open(LWRC_YAML, encoding="utf-8") as f:
        cluster = yaml.safe_load(f)["cluster"]
    return frozenset(cluster["seed_monsters"]), frozenset(cluster["jungle_camps"])


EPIC_MONSTER_TYPES, JUNGLE_CAMP_TYPES = _load_monster_names()
KNOWN_MONSTER_TYPES = EPIC_MONSTER_TYPES | JUNGLE_CAMP_TYPES


def is_epic(monster_type: str | None) -> bool:
    return monster_type in EPIC_MONSTER_TYPES


def check_monster_types(monster_types: Iterable[str | None], where: str = "") -> None:
    """설정에 없는 monsterType이 있으면 오류. 표기가 다른 에픽(예: 'horde', 'Atakhan')이 조용히 빠지는 것을 막는다."""
    unknown = sorted({str(m) for m in monster_types} - KNOWN_MONSTER_TYPES)
    if unknown:
        raise ValueError(f"알 수 없는 monsterType {unknown} {where} — config/lwrc.yaml cluster.seed_monsters/jungle_camps에 분류 추가 필요")


# 규칙 3 — 장로 = dragon + dragonType='elder'. 영혼 판정 시 장로 제외. 바다 드래곤은 'water'
def is_elder(monster_type: str | None, dragon_type: str | None) -> bool:
    return monster_type == "dragon" and dragon_type == "elder"


def soul_dragon_stacks(team_dragon_kills: int, team_elder_kills: int) -> int:
    """teams.dragonKills는 장로를 포함한다 → 영혼 스택은 장로를 뺀 값."""
    return team_dragon_kills - team_elder_kills


# 규칙 4 — deathRecap은 list. source='Champion'만, casterId별 physical+magic+true 합산
def parse_death_recap(recap) -> dict[int, float]:
    if recap is None:
        return {}
    if not isinstance(recap, list):
        raise TypeError(f"deathRecap은 list여야 합니다: {type(recap).__name__}")
    out: dict[int, float] = defaultdict(float)
    for entry in recap:
        if entry.get("source") != "Champion":
            continue
        dmg = sum(
            b.get("physicalDmg", 0) + b.get("magicDmg", 0) + b.get("trueDmg", 0)
            for b in entry.get("breakdown") or []
        )
        out[int(entry["casterId"])] += dmg
    return dict(out)


# 규칙 5 — champion_kill.bounty는 109~420 가변, 셧다운 골드는 shutdownBounty 별도 필드
BOUNTY_RANGE = (109, 420)


def parse_kill_gold(payload: dict) -> tuple[float, float]:
    return float(payload.get("bounty") or 0), float(payload.get("shutdownBounty") or 0)


def bounty_in_range(bounty: float) -> bool:
    lo, hi = BOUNTY_RANGE
    return lo <= bounty <= hi


# 규칙 6 — 팀 골드는 stats_update.teams[].totalGold 관측값 (근사 금지)
def team_gold(stats_payload: dict) -> dict[int, float]:
    return {int(t["teamID"]): float(t["totalGold"]) for t in stats_payload["teams"]}


# 규칙 7 — stats_update 1Hz. 벗어나면 QC 경고
TICK_MEAN_RANGE_S = (0.99, 1.01)
TICK_P95_MAX_S = 1.05
TICK_GAP_WARN_S = 2.0


def tick_interval_stats(times_ms: Sequence[float]) -> dict:
    t = np.sort(np.asarray(times_ms, dtype=float))
    d = np.diff(t) / GAME_TIME_UNIT_MS
    if d.size == 0:
        return {"n_ticks": int(t.size), "mean_s": np.nan, "p95_s": np.nan, "max_s": np.nan, "n_gaps": 0, "ok": False}
    stats = {
        "n_ticks": int(t.size),
        "mean_s": float(d.mean()),
        "p95_s": float(np.percentile(d, 95)),
        "max_s": float(d.max()),
        "n_gaps": int((d > TICK_GAP_WARN_S).sum()),
    }
    lo, hi = TICK_MEAN_RANGE_S
    stats["ok"] = lo <= stats["mean_s"] <= hi and stats["p95_s"] <= TICK_P95_MAX_S and stats["n_gaps"] == 0
    return stats


# 규칙 8 — ward_placed / ward_killed / item_purchased는 피드에 존재한다 (수집 결손 아님)
FEED_EVENTS_PRESENT = ("ward_placed", "ward_killed", "item_purchased")
