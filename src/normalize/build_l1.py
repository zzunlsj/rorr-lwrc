"""L1 정규화: 경기별 Raw(csv_extract | rawstats_ndjson) → 표준 테이블 Parquet (CLAUDE.md §4).

테이블 (data/l1/<table>/tournament=<t>/stage=<s>/<game_id>.parquet):
  games, participants, frames_team, frames_player,
  kills, kill_damage, monsters, buildings, ward_events, ward_kills, event_counts
게임 단위 밖 (Step 1, spec/06 §1):
  data/l1/players.parquet  선수 ID 매핑 (puuid 기준, 이름 변경·이적 포함)
  manifest/excluded_games.csv  재시작·식별 불일치·승자 미확정 게임과 사유

게임마다 한 프로세스가 한 경기만 메모리에 올린다.
사용: uv run python -m normalize.build_l1 [--game-id ID ...] [--workers 6]
"""

import argparse
import csv
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import yaml

from common.paths import L1, MANIFEST
from ingest.leaguepedia import LWRC_YAML, load_lp
from ingest.raw_reader import iter_events, read_session
from normalize import players, rules
from normalize.wards import build_ward_tables, level_lookup

TABLES = (
    "games", "participants", "frames_team", "frames_player",
    "kills", "kill_damage", "monsters", "buildings", "ward_events", "ward_kills", "event_counts",
)
PARSED_SCHEMAS = frozenset({
    "stats_update", "champion_kill", "epic_monster_kill", "building_destroyed", "building_respawned",
    "ward_placed", "ward_killed", "game_end", "game_info",
})
# 관여 피처용 누적 스탯 (spec/04 §1·§3b, 결정 19) — stats[] {name, value}
CUMULATIVE_STATS = {
    "TOTAL_DAMAGE_DEALT_TO_CHAMPIONS": "dmg_to_champions",
    "TOTAL_TIME_CROWD_CONTROL_DEALT_TO_CHAMPIONS": "cc_to_champions",
    "TOTAL_DAMAGE_SHIELDED_ON_TEAMMATES": "shielded_teammates",
    "TOTAL_HEAL_ON_TEAMMATES": "healed_teammates",
}


def _xz(payload: dict) -> tuple[float | None, float | None]:
    pos = payload.get("position") or {}
    return pos.get("x"), pos.get("z")


def parse_game(meta: dict, lp_winner: dict | None = None, cfg: dict | None = None) -> dict[str, pl.DataFrame]:
    cfg = cfg or yaml.safe_load(LWRC_YAML.read_text())
    game_id = meta["game_id"]
    counts: Counter = Counter()
    stats: dict[int, tuple[int, dict]] = {}  # gameTime → (sequenceIndex, payload); 같은 시각은 마지막 시퀀스
    kills, kill_dmg, monsters, buildings = [], [], [], []
    wards_placed, wards_killed = [], []
    game_end = game_info = None
    restart_markers = set()     # spec/06 §1: 리메이크·크로노브레이크
    stats_seq_time = []         # (seq, gameTime) — 시퀀스 순 게임 시각 되감기 탐지

    read_stats: dict = {}
    for schema, seq, p in iter_events(meta, read_stats):
        counts[schema] += 1
        if schema not in PARSED_SCHEMAS:
            continue
        gen, parent = p.get("generationID"), p.get("parentGameID")
        if gen not in (None, 0, 1) or parent not in (None, 0):
            restart_markers.add((gen, parent))
        if schema == "stats_update":
            stats_seq_time.append((seq, int(p["gameTime"])))
            t = int(p["gameTime"])
            if t not in stats or seq > stats[t][0]:
                stats[t] = (seq, p)
        elif schema == "champion_kill":
            bounty, shutdown = rules.parse_kill_gold(p)
            recap = rules.parse_death_recap(p.get("deathRecap"))
            x, z = _xz(p)
            kills.append({
                "seq": seq, "game_time_ms": int(p["gameTime"]),
                "killer": int(p["killer"]), "victim": int(p["victim"]),
                "killer_team": int(p["killerTeamID"]), "victim_team": int(p["victimTeamID"]),
                "assistants": [int(a) for a in p.get("assistants") or []],
                "bounty": bounty, "shutdown_bounty": shutdown,
                "has_champion_recap": bool(recap), "x": x, "z": z,
            })
            kill_dmg.extend({"kill_seq": seq, "caster_id": c, "dmg": d} for c, d in recap.items())
        elif schema == "epic_monster_kill":
            mt, dt = p.get("monsterType"), p.get("dragonType")
            x, z = _xz(p)
            monsters.append({
                "seq": seq, "game_time_ms": int(p["gameTime"]), "monster_type": mt, "dragon_type": dt,
                "killer": int(p["killer"]) if p.get("killer") is not None else None,
                "killer_team": int(p["killerTeamID"]),
                "assistants": [int(a) for a in p.get("assistants") or []],
                "kill_type": p.get("killType"),
                "is_epic": rules.is_epic(mt), "is_elder": rules.is_elder(mt, dt), "x": x, "z": z,
            })
        elif schema in ("building_destroyed", "building_respawned"):
            x, z = _xz(p)
            buildings.append({
                "seq": seq, "game_time_ms": int(p["gameTime"]),
                "event": "destroyed" if schema == "building_destroyed" else "respawned",
                "team_id": int(p["teamID"]),
                "lane": p.get("lane"), "building_type": p.get("buildingType"), "turret_tier": p.get("turretTier"),
                "nexus_turret_name": p.get("nexusTurretName"),
                "last_hitter": int(p["lastHitter"]) if p.get("lastHitter") is not None else None,
                "assistants": [int(a) for a in p.get("assistants") or []], "x": x, "z": z,
            })
        elif schema in ("ward_placed", "ward_killed"):
            x, z = _xz(p)
            who = p.get("placer") if schema == "ward_placed" else p.get("killer")
            (wards_placed if schema == "ward_placed" else wards_killed).append({
                "seq": seq, "game_time_ms": int(p["gameTime"]), "ward_type": p.get("wardType"),
                "pid": int(who) if who is not None else None, "x": x, "z": z,
            })
        elif schema == "game_end":
            game_end = p
        elif schema == "game_info":
            game_info = p

    # 규칙 1: gameTime 정렬
    ticks = sorted(stats.items())
    frames_team, frames_player = [], []
    participants: dict[int, dict] = {}
    pid_puuids: dict[int, set] = {}
    tick_times, tick_avg_level = [], []
    for t, (seq, p) in ticks:
        levels = [q.get("level") for q in p["participants"] if q.get("level") is not None]
        tick_times.append(t)
        tick_avg_level.append(sum(levels) / len(levels) if levels else None)
        gold = rules.team_gold(p)  # 규칙 6
        for team in p["teams"]:
            tid = int(team["teamID"])
            frames_team.append({
                "game_time_ms": t, "seq": seq, "team_id": tid, "total_gold": gold[tid],
                "tower_kills": team.get("towerKills"), "inhib_kills": team.get("inhibKills"),
                "baron_kills": team.get("baronKills"), "dragon_kills": team.get("dragonKills"),
                "champion_kills": team.get("championsKills"), "deaths": team.get("deaths"),
            })
        for q in p["participants"]:
            pid = int(q["participantID"])
            pos = q.get("position") or {}
            cum = {s["name"]: s.get("value") for s in q.get("stats") or [] if s.get("name") in CUMULATIVE_STATS}
            pid_puuids.setdefault(pid, set()).add(q.get("puuid"))
            frames_player.append({
                "game_time_ms": t, "seq": seq, "participant_id": pid, "team_id": int(q["teamID"]),
                "total_gold": float(q["totalGold"]), "current_gold": float(q.get("currentGold") or 0),
                "alive": bool(q["alive"]), "respawn_timer": float(q.get("respawnTimer") or 0),
                "health": float(q.get("health") or 0), "health_max": float(q.get("healthMax") or 0),
                "x": pos.get("x"), "z": pos.get("z"), "level": q.get("level"), "xp": q.get("XP"),
                # 누적값 — 결측은 null 그대로 (0으로 채우지 않는다)
                **{col: (float(cum[name]) if cum.get(name) is not None else None) for name, col in CUMULATIVE_STATS.items()},
            })
            if pid not in participants:
                participants[pid] = {
                    "participant_id": pid, "team_id": int(q["teamID"]), "role": q.get("role"),
                    "player_name": q.get("playerName"), "champion": q.get("championName"), "puuid": q.get("puuid"),
                }

    version = (game_info or {}).get("gameVersion", "")
    end_ms = int(game_end["gameTime"]) if game_end else (ticks[-1][0] if ticks else None)
    tick_stats = rules.tick_interval_stats([t for t, _ in ticks])  # 규칙 7
    # 결정 22: game_end·game_info가 없으면 Kazel 메타로 대체 (승자는 Leaguepedia 교차 확인)
    session = read_session(meta) or {}
    winner, winner_source = rules.resolve_winner(
        game_end.get("winningTeam") if game_end else None,
        session.get("winning_side"), (lp_winner or {}).get(meta["platform_game_id"]))
    patch, patch_source = rules.resolve_patch(version, session.get("patch"))

    # 선수·팀 (spec/06 §1): 진영 → Kazel 팀 ID. 이름은 팀 접두어를 뗀 표시 이름도 함께 둔다
    side_team = {100: session.get("blue_team") or {}, 200: session.get("red_team") or {}}
    for q in participants.values():
        team = side_team.get(q["team_id"], {})
        q.update({"esports_team_id": team.get("esports_team_id"), "team_code": team.get("code"),
                  "team_name": team.get("name"), "display_name": players.strip_team_prefix(q["player_name"])})

    # 품질 플래그와 제외 사유 (데이터는 고치지 않고 탐지·플래그·제외)
    by_seq = sorted(stats_seq_time)
    rewinds = sum(1 for a, b in zip(by_seq, by_seq[1:]) if b[1] < a[1] - rules.REWIND_TOLERANCE_MS)
    id_mismatch = sum(len(v) > 1 or None in v for v in pid_puuids.values())
    exclude = [reason for reason, hit in (
        ("restart", bool(restart_markers) or rewinds > 0),
        ("player_id_mismatch", id_mismatch > 0 or len(participants) != 10),
        ("winner_unresolved", winner is None),
    ) if hit]

    # 와드 (spec/06 §2): 생존 구간과 제거 종류
    ward_events, ward_kills = build_ward_tables(
        wards_placed, wards_killed, level_lookup(tick_times, tick_avg_level),
        {pid: q["team_id"] for pid, q in participants.items()}, end_ms or 0,
        cfg["allocation"]["ward_rules"], set(cfg["allocation"]["vision_ward_types"]))

    games = [{
        "match_id": meta["match_id"], "game_n": int(meta["game_n"]), "platform_game_id": meta["platform_game_id"],
        "game_version": version, "patch": patch, "patch_source": patch_source,
        "winning_team": winner, "winner_source": winner_source, "has_game_end": game_end is not None,
        "end_game_time_ms": end_ms, "first_tick_ms": ticks[0][0] if ticks else None,
        "last_tick_ms": ticks[-1][0] if ticks else None, "n_ticks": len(ticks),
        "game_time_is_ms": rules.game_time_is_ms(end_ms or 0, len(ticks)),
        "tick_mean_s": tick_stats["mean_s"], "tick_p95_s": tick_stats["p95_s"],
        "tick_max_s": tick_stats["max_s"], "tick_n_gaps": tick_stats["n_gaps"], "tick_ok": tick_stats["ok"],
        "raw_sha256_gz": meta["sha256_gz"], "raw_format": meta.get("raw_format") or "csv_extract",
        "n_events": read_stats["events"], "n_duplicate_events": read_stats["duplicates"],
        "game_start_utc": session.get("game_start_time"), "league_label": session.get("league"),
        "tournament_label": session.get("tournament"), "stage_label": session.get("stage"),
        "blue_team_id": side_team[100].get("esports_team_id"), "blue_team_code": side_team[100].get("code"),
        "red_team_id": side_team[200].get("esports_team_id"), "red_team_code": side_team[200].get("code"),
        "restart_markers": len(restart_markers), "time_rewinds": rewinds, "player_id_mismatch": id_mismatch,
        "n_ward_kills_unmatched": sum(k["ward_id"] is None for k in ward_kills),
        "excluded": bool(exclude), "exclude_reason": ";".join(exclude),
    }]

    rules.check_monster_types((m["monster_type"] for m in monsters), where=f"(game {game_id})")

    def df(rows: list[dict], sort: tuple[str, ...] = ()) -> pl.DataFrame:
        out = pl.DataFrame(rows, infer_schema_length=None)
        if sort and out.height:
            out = out.sort(list(sort))
        return out.with_columns(pl.lit(game_id).alias("game_id")).select(["game_id", *out.columns])

    return {
        "games": df(games),
        "participants": df(list(participants.values()), ("participant_id",)),
        "frames_team": df(frames_team, ("game_time_ms", "team_id")),
        "frames_player": df(frames_player, ("game_time_ms", "participant_id")),
        "kills": df(kills, ("game_time_ms", "seq")),
        "kill_damage": df(kill_dmg, ("kill_seq", "caster_id")),
        "monsters": df(monsters, ("game_time_ms", "seq")),
        "buildings": df(buildings, ("game_time_ms", "seq")),
        "ward_events": df(ward_events, ("t_placed_ms", "ward_id")),
        "ward_kills": df(ward_kills, ("game_time_ms", "seq")),
        "event_counts": df([{"schema": k, "n": v} for k, v in sorted(counts.items())]),
    }


def leaguepedia_winner_sides() -> dict[str, str]:
    """platform_game_id → 승리 진영(blue|red). Leaguepedia Winner 1 = 블루, 2 = 레드."""
    _, lp, _ = load_lp()
    return {pgid: {"1": "blue", "2": "red"}.get(g["winner"]) for pgid, g in lp.items()}


def l1_path(table: str, meta: dict) -> Path:
    return L1 / table / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"


def build_one(meta: dict, lp_winner: dict, cfg: dict) -> dict:
    tables = parse_game(meta, lp_winner, cfg)
    for name, frame in tables.items():
        path = l1_path(name, meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression="zstd")
    g = tables["games"].row(0, named=True)
    return {"game_id": meta["game_id"], "platform_game_id": meta["platform_game_id"], "tournament": meta["tournament"],
            "stage": meta["stage"], "excluded": g["excluded"], "exclude_reason": g["exclude_reason"]}


def build(game_ids: set[str] | None = None, workers: int = 1) -> list[dict]:
    with open(MANIFEST / "received.csv", newline="") as f:
        # superseded_by가 비어 있는 행만 (같은 게임을 두 형식으로 받았으면 tar 원본이 우선)
        metas = [r for r in csv.DictReader(f)
                 if not r.get("superseded_by") and (not game_ids or r["game_id"] in game_ids)]
    job = partial(build_one, lp_winner=leaguepedia_winner_sides(), cfg=yaml.safe_load(LWRC_YAML.read_text()))
    if workers > 1:
        with Pool(workers) as pool:
            return list(pool.imap_unordered(job, metas, chunksize=4))
    return [job(m) for m in metas]


def write_excluded(results: list[dict]) -> None:
    """manifest/excluded_games.csv — Step 1 제외분. 다시 만들 때마다 전체를 갱신한다."""
    rows = sorted((r for r in results if r["excluded"]), key=lambda r: (r["tournament"], r["game_id"]))
    with open(MANIFEST / "excluded_games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "platform_game_id", "tournament", "stage", "step", "reason"])
        w.writeheader()
        w.writerows({**{k: r[k] for k in ("game_id", "platform_game_id", "tournament", "stage")},
                     "step": "1_normalize", "reason": r["exclude_reason"]} for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    results = build(set(args.game_id) if args.game_id else None, args.workers)
    if not args.game_id:  # 전체 빌드일 때만 게임 밖 테이블 갱신
        write_excluded(results)
        players.build_players()
    print(f"L1 완료: {len(results)}경기, 제외 {sum(r['excluded'] for r in results)}, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
