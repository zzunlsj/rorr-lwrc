"""교전 클러스터와 틱 구간 완전 분할 (CLAUDE.md §6-4, 정본 v1.0 §3).

틱 구간 i = (t_{i-1}, t_i], i=1..n-1. 구간 중점이 클러스터 [s, e] 안이면 그 클러스터,
아니면 드리프트(-1). 한 구간은 정확히 하나의 버킷에만 속한다.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from features.state import GameState, opp

DRIFT = -1


@dataclass
class Seed:
    time_ms: int
    kind: str              # kill / epic / building
    credited_team: int     # 이 씨앗으로 이득을 본 팀
    x: float | None
    z: float | None
    killer: int | None
    assistants: list[int]
    kill_seq: int | None = None
    victim: int | None = None
    victim_team: int | None = None


def seeds_of(state: GameState) -> list[Seed]:
    out = []
    for r in state.kills.iter_rows(named=True):
        out.append(Seed(r["game_time_ms"], "kill", r["killer_team"], r["x"], r["z"], r["killer"],
                        r["assistants"] or [], kill_seq=r["seq"], victim=r["victim"], victim_team=r["victim_team"]))
    for r in state.epics.iter_rows(named=True):
        out.append(Seed(r["game_time_ms"], "epic", r["killer_team"], r["x"], r["z"], r["killer"], r["assistants"] or []))
    for r in state.buildings.iter_rows(named=True):
        out.append(Seed(r["game_time_ms"], "building", opp(r["team_id"]), r["x"], r["z"], r["last_hitter"], r["assistants"] or []))
    return sorted(out, key=lambda s: s.time_ms)


@dataclass
class Cluster:
    k: int
    start_ms: float
    end_ms: float
    seeds: list[Seed]


def build_clusters(seeds: list[Seed], t_start: float, t_end: float, window_sec: float,
                   merge_sec: float, merge_rule: str = "window_gap") -> list[Cluster]:
    w, gap = window_sec * 1000, merge_sec * 1000
    if merge_rule == "window_gap":
        max_seed_gap = 2 * w + gap        # 창 사이 간격 ≤ merge
    elif merge_rule == "seed_gap":
        max_seed_gap = gap                # 문구 그대로 (겹침 발생 가능 — QC가 잡는다)
    else:
        raise ValueError(merge_rule)
    groups: list[list[Seed]] = []
    for s in seeds:
        if groups and s.time_ms - groups[-1][-1].time_ms <= max_seed_gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    return [Cluster(k, max(t_start, g[0].time_ms - w), min(t_end, g[-1].time_ms + w), g) for k, g in enumerate(groups)]


def assign_intervals(t: np.ndarray, clusters: list[Cluster]) -> tuple[np.ndarray, np.ndarray]:
    """틱 구간별 버킷(클러스터 k 또는 DRIFT)과, 구간 중점을 포함하는 클러스터 수(분할 검증용)."""
    mid = (t[1:] + t[:-1]) / 2.0
    bucket = np.full(mid.size, DRIFT, dtype=int)
    hits = np.zeros(mid.size, dtype=int)
    for c in clusters:
        inside = (mid > c.start_ms) & (mid <= c.end_ms)
        hits += inside
        bucket[inside & (bucket == DRIFT)] = c.k
    return bucket, hits


def seeds_frame(clusters: list[Cluster]) -> pl.DataFrame:
    return pl.DataFrame([
        {"k": c.k, "time_ms": s.time_ms, "kind": s.kind, "credited_team": s.credited_team}
        for c in clusters for s in c.seeds
    ])
