"""L1 정규화: 경기별 Raw(csv.gz) → 표준 테이블 Parquet (CLAUDE.md §4).

테이블 (data/l1/<table>/tournament=<t>/stage=<s>/<game_id>.parquet):
  games, participants, frames_team, frames_player,
  kills, kill_damage, monsters, buildings, event_counts

한 번에 한 경기만 메모리에 올린다.
사용: uv run python -m normalize.build_l1 [--game-id ID ...]
"""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from common.paths import L1, MANIFEST, ROOT
from normalize import rules

csv.field_size_limit(sys.maxsize)

TABLES = (
    "games", "participants", "frames_team", "frames_player",
    "kills", "kill_damage", "monsters", "buildings", "event_counts",
)
PARSED_SCHEMAS = frozenset({
    "stats_update", "champion_kill", "epic_monster_kill", "building_destroyed", "game_end", "game_info",
})


def _xz(payload: dict) -> tuple[float | None, float | None]:
    pos = payload.get("position") or {}
    return pos.get("x"), pos.get("z")


def parse_game(raw_path: Path, meta: dict) -> dict[str, pl.DataFrame]:
    game_id = meta["game_id"]
    counts: Counter = Counter()
    stats: dict[int, tuple[int, dict]] = {}  # gameTime → (sequenceIndex, payload); 같은 시각은 마지막 시퀀스
    kills, kill_dmg, monsters, buildings = [], [], [], []
    game_end = game_info = None

    with gzip.open(raw_path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            schema = row["rfc461_schema"]
            counts[schema] += 1
            if schema not in PARSED_SCHEMAS:
                continue
            p = json.loads(row["payload"])
            seq = int(row["sequence_index"])
            if schema == "stats_update":
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
            elif schema == "building_destroyed":
                x, z = _xz(p)
                buildings.append({
                    "seq": seq, "game_time_ms": int(p["gameTime"]), "team_id": int(p["teamID"]),
                    "lane": p.get("lane"), "building_type": p.get("buildingType"), "turret_tier": p.get("turretTier"),
                    "last_hitter": int(p["lastHitter"]) if p.get("lastHitter") is not None else None,
                    "assistants": [int(a) for a in p.get("assistants") or []], "x": x, "z": z,
                })
            elif schema == "game_end":
                game_end = p
            elif schema == "game_info":
                game_info = p

    # 규칙 1: gameTime 정렬
    ticks = sorted(stats.items())
    frames_team, frames_player = [], []
    participants: dict[int, dict] = {}
    for t, (seq, p) in ticks:
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
            frames_player.append({
                "game_time_ms": t, "seq": seq, "participant_id": pid, "team_id": int(q["teamID"]),
                "total_gold": float(q["totalGold"]), "current_gold": float(q.get("currentGold") or 0),
                "alive": bool(q["alive"]), "respawn_timer": float(q.get("respawnTimer") or 0),
                "health": float(q.get("health") or 0), "health_max": float(q.get("healthMax") or 0),
                "x": pos.get("x"), "z": pos.get("z"), "level": q.get("level"), "xp": q.get("XP"),
            })
            if pid not in participants:
                participants[pid] = {
                    "participant_id": pid, "team_id": int(q["teamID"]), "role": q.get("role"),
                    "player_name": q.get("playerName"), "champion": q.get("championName"), "puuid": q.get("puuid"),
                }

    version = (game_info or {}).get("gameVersion", "")
    end_ms = int(game_end["gameTime"]) if game_end else (ticks[-1][0] if ticks else None)
    tick_stats = rules.tick_interval_stats([t for t, _ in ticks])  # 규칙 7
    games = [{
        "match_id": meta["match_id"], "game_n": int(meta["game_n"]), "platform_game_id": meta["platform_game_id"],
        "game_version": version, "patch": ".".join(version.split(".")[:2]),
        "winning_team": int(game_end["winningTeam"]) if game_end else None,
        "end_game_time_ms": end_ms, "first_tick_ms": ticks[0][0] if ticks else None,
        "last_tick_ms": ticks[-1][0] if ticks else None, "n_ticks": len(ticks),
        "game_time_is_ms": rules.game_time_is_ms(end_ms or 0, len(ticks)),
        "tick_mean_s": tick_stats["mean_s"], "tick_p95_s": tick_stats["p95_s"],
        "tick_max_s": tick_stats["max_s"], "tick_n_gaps": tick_stats["n_gaps"], "tick_ok": tick_stats["ok"],
        "raw_sha256_gz": meta["sha256_gz"],
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
        "event_counts": df([{"schema": k, "n": v} for k, v in sorted(counts.items())]),
    }


def l1_path(table: str, meta: dict) -> Path:
    return L1 / table / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"


def build(game_ids: set[str] | None = None) -> list[str]:
    with open(MANIFEST / "received.csv", newline="") as f:
        metas = [r for r in csv.DictReader(f) if not game_ids or r["game_id"] in game_ids]
    done = []
    for meta in metas:
        tables = parse_game(ROOT / meta["raw_path"], meta)
        for name, frame in tables.items():
            path = l1_path(name, meta)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path, compression="zstd")
        done.append(meta["game_id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", nargs="*")
    args = ap.parse_args()
    done = build(set(args.game_id) if args.game_id else None)
    print(f"L1 완료: {len(done)}경기")


if __name__ == "__main__":
    main()
