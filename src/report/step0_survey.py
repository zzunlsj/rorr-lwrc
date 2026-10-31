"""Step 0 데이터 조사 (docs/workflow.md Step 0, docs/contracts/input.md §1).

L0 Raw를 게임마다 한 번 훑어 필드 적재율·프레임 간격·품질 플래그·선수 목록을 게임 단위 행으로 만든다.
결과: data/l2/step0/games.parquet, players.parquet (git 제외) → 리포트는 사람이 읽는 표로 따로 정리한다.

사용: uv run python -m report.step0_survey [--workers 6]
"""

import argparse
import csv
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.paths import L2, MANIFEST
from ingest.leaguepedia import LWRC_YAML
from ingest.raw_reader import iter_events

OUT = L2 / "step0"
DMG = "TOTAL_DAMAGE_DEALT_TO_CHAMPIONS"
CC = "TOTAL_TIME_CROWD_CONTROL_DEALT_TO_CHAMPIONS"
SHIELD = "TOTAL_DAMAGE_SHIELDED_ON_TEAMMATES"
HEAL = "TOTAL_HEAL_ON_TEAMMATES"
CUMULATIVE = (DMG, CC, SHIELD, HEAL)
FRAME_FIELDS = ("participantID", "teamID", "totalGold", "level", "alive", "respawnTimer", "health", "healthMax")
WARD_MATCH_XZ = 50


def _has_xz(p: dict) -> bool:
    pos = p.get("position") or {}
    return pos.get("x") is not None and pos.get("z") is not None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def survey_game(meta: dict) -> tuple[dict, list[dict]]:
    stats_rows = []                     # (seq, gameTime, payload) — stats_update만
    counts: Counter = Counter()
    kills = Counter()
    epic = Counter()
    bld = Counter()
    monster_types: Counter = Counter()
    wards_placed, wards_killed = [], []
    game_end = game_info = None
    restart_ids = set()
    read: dict = {}

    for schema, seq, p in iter_events(meta, read):
        counts[schema] += 1
        gen, parent = p.get("generationID"), p.get("parentGameID")
        if gen not in (None, 0, 1) or parent not in (None, 0):
            restart_ids.add((gen, parent))
        if schema == "stats_update":
            stats_rows.append((seq, int(p["gameTime"]), p))
        elif schema == "champion_kill":
            kills["n"] += 1
            kills["killer"] += p.get("killer") is not None
            kills["victim"] += p.get("victim") is not None
            kills["assistants"] += "assistants" in p
            kills["bounty"] += p.get("bounty") is not None
            kills["xz"] += _has_xz(p)
            recap = p.get("deathRecap")
            kills["recap_list"] += isinstance(recap, list)
            kills["recap_champion"] += isinstance(recap, list) and any(r.get("source") == "Champion" for r in recap)
        elif schema == "epic_monster_kill":
            mt = p.get("monsterType")
            monster_types[mt] += 1
            epic["n"] += 1
            epic["team"] += p.get("killerTeamID") is not None
            epic["type"] += mt is not None
            epic["xz"] += _has_xz(p)
            if mt == "dragon":
                epic["dragon"] += 1
                epic["dragon_type"] += p.get("dragonType") is not None
        elif schema == "building_destroyed":
            bld["n"] += 1
            bld["team"] += p.get("teamID") is not None
            bld["type"] += p.get("buildingType") is not None
            bld["xz"] += _has_xz(p)
            if p.get("buildingType") == "turret":
                bld["turret"] += 1
                bld["tier"] += p.get("turretTier") is not None
        elif schema == "ward_placed":
            pos = p.get("position") or {}
            wards_placed.append((int(p["gameTime"]), p.get("wardType"), p.get("placer"), pos.get("x"), pos.get("z")))
        elif schema == "ward_killed":
            pos = p.get("position") or {}
            wards_killed.append((int(p["gameTime"]), p.get("wardType"), p.get("killer"), pos.get("x"), pos.get("z")))
        elif schema == "game_end":
            game_end = p
        elif schema == "game_info":
            game_info = p

    # 프레임: 시퀀스 순서에서 게임 시각이 되감기는지(재시작), 시각 정렬 후 간격
    by_seq = sorted(stats_rows, key=lambda r: r[0])
    rewinds = sum(1 for a, b in zip(by_seq, by_seq[1:]) if b[1] < a[1] - 1000)
    times = np.array(sorted({t for _, t, _ in stats_rows}), dtype=float) / 1000
    gaps = np.diff(times) if len(times) > 1 else np.array([])

    frame = Counter()
    pid_identity: dict[int, set] = {}
    players: dict[int, dict] = {}
    last_cum: dict[tuple, float] = {}
    for _, t, p in sorted(stats_rows, key=lambda r: (r[1], r[0])):
        for q in p.get("participants", []):
            frame["n"] += 1
            frame["fields"] += all(q.get(f) is not None for f in FRAME_FIELDS)
            frame["xz"] += _has_xz(q)
            st = {s["name"]: s["value"] for s in q.get("stats") or [] if "name" in s}
            pid = q.get("participantID")
            for name in CUMULATIVE:
                if name in st:
                    frame[name] += 1
                    prev = last_cum.get((pid, name))
                    if prev is not None and st[name] < prev:
                        frame[name + "_decrease"] += 1
                    last_cum[(pid, name)] = st[name]
            pid_identity.setdefault(pid, set()).add(q.get("puuid"))
            if pid not in players:
                players[pid] = {"participant_id": pid, "team_id": q.get("teamID"), "puuid": q.get("puuid"),
                                "player_name": q.get("playerName"), "role": q.get("role"), "champion": q.get("championName")}

    # 와드: 제거 ↔ 설치 위치·시각 매칭 (같은 종류, ±50, 설치가 먼저, 미사용 중 가장 최근)
    used = set()
    matched = 0
    for kt, kw, _, kx, kz in sorted(wards_killed):
        best = None
        for i, (pt, pw, _, px, pz) in enumerate(wards_placed):
            if i in used or pw != kw or pt > kt or px is None or kx is None:
                continue
            if abs(px - kx) <= WARD_MATCH_XZ and abs(pz - kz) <= WARD_MATCH_XZ and (best is None or pt > wards_placed[best][0]):
                best = i
        if best is not None:
            used.add(best)
            matched += 1
    n_wp = len(wards_placed)
    version = (game_info or {}).get("gameVersion", "")
    row = {
        "game_id": meta["game_id"], "match_id": meta["match_id"], "game_n": int(meta["game_n"]),
        "platform_game_id": meta["platform_game_id"], "tournament": meta["tournament"], "stage": meta["stage"],
        "month": meta["source_file"][:7], "patch": ".".join(version.split(".")[:2]), "game_version": version,
        "n_events": read["events"], "n_events_expected": int(meta["rows"]), "n_duplicate_events": read["duplicates"],
        "n_stats_update": counts["stats_update"],
        # 프레임·참가자
        "frame_fields_rate": _rate(frame["fields"], frame["n"]), "frame_xz_rate": _rate(frame["xz"], frame["n"]),
        "dmg_rate": _rate(frame[DMG], frame["n"]), "dmg_decreases": frame[DMG + "_decrease"],
        "cc_rate": _rate(frame[CC], frame["n"]), "cc_decreases": frame[CC + "_decrease"],
        "shield_rate": _rate(frame[SHIELD], frame["n"]), "shield_decreases": frame[SHIELD + "_decrease"],
        "heal_rate": _rate(frame[HEAL], frame["n"]), "heal_decreases": frame[HEAL + "_decrease"],
        "n_participants": len(players),
        # 이벤트
        "n_kills": kills["n"],
        "kill_fields_rate": _rate(min(kills["killer"], kills["victim"], kills["assistants"], kills["bounty"]), kills["n"]),
        "kill_xz_rate": _rate(kills["xz"], kills["n"]), "kill_recap_list_rate": _rate(kills["recap_list"], kills["n"]),
        "kill_recap_champion_rate": _rate(kills["recap_champion"], kills["n"]),
        "n_epic_events": epic["n"], "epic_fields_rate": _rate(min(epic["team"], epic["type"]), epic["n"]),
        "epic_xz_rate": _rate(epic["xz"], epic["n"]), "dragon_type_rate": _rate(epic["dragon_type"], epic["dragon"]),
        "monster_types": ",".join(sorted(f"{k}:{v}" for k, v in monster_types.items() if k)),
        "n_buildings": bld["n"], "building_fields_rate": _rate(min(bld["team"], bld["type"]), bld["n"]),
        "building_xz_rate": _rate(bld["xz"], bld["n"]), "turret_tier_rate": _rate(bld["tier"], bld["turret"]),
        "n_ward_placed": n_wp, "n_ward_killed": len(wards_killed),
        "ward_placed_xz_rate": _rate(sum(w[3] is not None for w in wards_placed), n_wp),
        "ward_unknown": sum(w[1] == "unknown" for w in wards_placed),
        "ward_placer_unmapped": sum(w[2] not in players for w in wards_placed),
        "ward_kill_match_rate": _rate(matched, len(wards_killed)),
        # 프레임 해상도
        "n_ticks": len(times), "tick_mean_s": float(gaps.mean()) if len(gaps) else None,
        "tick_p95_s": float(np.percentile(gaps, 95)) if len(gaps) else None,
        "tick_max_s": float(gaps.max()) if len(gaps) else None, "tick_gaps_over_2s": int((gaps > 2).sum()),
        # 품질 플래그
        "restart_flag": bool(restart_ids) or rewinds > 0, "time_rewinds": rewinds,
        "pid_identity_mismatch": sum(len(v) > 1 for v in pid_identity.values()),
        "winner": (game_end or {}).get("winningTeam"), "winner_missing": not (game_end or {}).get("winningTeam"),
        "had_pause": counts["pause_started"] > 0,
    }
    player_rows = [{**pl_, "game_id": meta["game_id"], "tournament": meta["tournament"], "month": row["month"]}
                   for pl_ in players.values()]
    return row, player_rows


def active_metas() -> list[dict]:
    with open(MANIFEST / "received.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if not r.get("superseded_by")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    metas = active_metas()
    split_of = {t: s for s, ts in yaml.safe_load(LWRC_YAML.read_text())["season"]["scope_tournaments"].items()
                if s.startswith("lck_s") for t in ts}
    t0 = time.time()
    games, players = [], []
    with Pool(args.workers) as pool:
        for i, (row, prow) in enumerate(pool.imap_unordered(survey_game, metas, chunksize=4), 1):
            games.append(row)
            players += prow
            if i % 100 == 0:
                print(f"{i}/{len(metas)} {time.time() - t0:.0f}초", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    g = pl.DataFrame(games, infer_schema_length=None).with_columns(
        pl.col("tournament").replace_strict(split_of, default=None).alias("lck_split"))
    g.sort("month", "tournament", "match_id", "game_n").write_parquet(OUT / "games.parquet")
    pl.DataFrame(players, infer_schema_length=None).write_parquet(OUT / "players.parquet")
    print(f"{len(games)}게임, {time.time() - t0:.0f}초 → {OUT}")


if __name__ == "__main__":
    main()
