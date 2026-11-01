"""Step 2 — 상태 테이블 x(t, side) + 학습 레이블 y(t, side) (contracts/input.md §2, spec/01 §1.1·§1.2).

한 게임 = 1Hz 틱 × 양 팀 관점 2행. 산출(ΔG)은 1Hz 전체를 쓰고, 학습은 g.train_sample_sec 간격으로 뽑는다.
결과: data/l2/state/tournament=<t>/stage=<s>/<game_id>.parquet

해석 메모 (사양 문구가 열려 있는 곳 — 리포트에 적는다):
- 타워 6–11: turret_tier outer / inner / base(억제기 타워). 외곽·2차·억제기 타워는 리스폰이 없다.
- 억제기 12–13: 파괴 − 리스폰 = 현재 파괴 상태 수 (사양 그대로).
- 넥서스 타워 14–15: 넥서스 타워는 리스폰 이벤트가 있다(678게임 38건) → 억제기와 같이 현재 파괴 상태 수로 둔다.
- 드래곤 스택 16–17: 장로 제외 처치 수. 영혼 18–19: 스택 ≥ 4.
- 버프 20–23: 처치 시각부터 config buffs.baron_sec / elder_sec 동안 남은 초 (팀 단위).
- 처치 팀이 100/200이 아닌 몬스터(중립 처치, killer_team 300)는 어느 쪽에도 세지 않는다.
"""

import argparse
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.paths import L1, L2
from ingest.leaguepedia import LWRC_YAML

OUT = L2 / "state"
SIDES = (100, 200)
FEATURES = [
    "t_min", "t_min2",
    "gold_us", "gold_opp",
    "level_diff",
    "tower_outer_taken", "tower_inner_taken", "tower_base_taken",
    "tower_outer_lost", "tower_inner_lost", "tower_base_lost",
    "inhib_down_opp", "inhib_down_us",
    "nexus_tower_down_opp", "nexus_tower_down_us",
    "dragon_stack_us", "dragon_stack_opp", "soul_us", "soul_opp",
    "baron_buff_us", "baron_buff_opp", "elder_buff_us", "elder_buff_opp",
    "alive_us", "alive_opp",
    "respawn_sum_us", "respawn_sum_opp",
    "voidgrub_us", "voidgrub_opp",
    "herald_or_atakhan_us", "herald_or_atakhan_opp",
    "side_blue",
]
TIME_FEATURES = ("t_min", "t_min2")
# 관점 반전 쌍 (contracts/input.md §2): 우리↔상대 교환, 레벨차 부호 반전, 진영 반전
MIRROR_PAIRS = [(f, f.replace("_us", "_opp")) for f in FEATURES if f.endswith("_us")] + [
    ("tower_outer_taken", "tower_outer_lost"), ("tower_inner_taken", "tower_inner_lost"),
    ("tower_base_taken", "tower_base_lost"), ("inhib_down_opp", "inhib_down_us"),
    ("nexus_tower_down_opp", "nexus_tower_down_us"),
]
assert len(FEATURES) == 32


def _read(table: str, game_id: str) -> pl.DataFrame:
    return pl.read_parquet(L1 / table / "**" / f"{game_id}.parquet", hive_partitioning=True)


def _count_upto(event_ms: np.ndarray, t: np.ndarray) -> np.ndarray:
    """각 틱 t에서 시각 ≤ t인 이벤트 수."""
    return np.searchsorted(np.sort(event_ms), t, side="right").astype(float)


def _buff_left(kill_ms: np.ndarray, t: np.ndarray, dur_s: float) -> np.ndarray:
    """마지막 처치 이후 남은 버프 초 (없으면 0)."""
    if kill_ms.size == 0:
        return np.zeros(t.size)
    k = np.sort(kill_ms)
    i = np.searchsorted(k, t, side="right") - 1
    last = np.where(i >= 0, k[np.clip(i, 0, None)], -np.inf)
    return np.clip(dur_s - (t - last) / 1000.0, 0, dur_s)


def _gold_at(t_ticks: np.ndarray, gold: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """t_query 시점의 관측 팀 골드 (그 시각 이전 마지막 틱)."""
    i = np.clip(np.searchsorted(t_ticks, t_query, side="right") - 1, 0, t_ticks.size - 1)
    return gold[i]


def labels(t: np.ndarray, gold: np.ndarray, end_ms: float, horizon_min: float, rate_window_min: float) -> tuple[np.ndarray, float]:
    """spec/01 §1.1: y = gold(t+H), 종료를 넘으면 gold(END) + rate × (t+H−END). rate = 종료 직전 창의 분당 수입."""
    h_ms = horizon_min * 60_000
    g_end = _gold_at(t, gold, np.array([end_ms]))[0]
    g_before = _gold_at(t, gold, np.array([end_ms - rate_window_min * 60_000]))[0]
    rate = (g_end - g_before) / rate_window_min  # 골드/분
    target = t + h_ms
    inside = target <= end_ms
    y = np.where(inside, _gold_at(t, gold, np.minimum(target, end_ms)),
                 g_end + rate * (target - end_ms) / 60_000)
    return y, rate


def build_state(game_id: str, cfg: dict) -> pl.DataFrame:
    games = _read("games", game_id).row(0, named=True)
    ft = _read("frames_team", game_id).sort("game_time_ms")
    fp = _read("frames_player", game_id).sort("game_time_ms", "participant_id")
    b = _read("buildings", game_id)
    m = _read("monsters", game_id).filter(pl.col("is_epic"))

    t = ft.filter(pl.col("team_id") == 100)["game_time_ms"].to_numpy().astype(float)
    gold = {s: ft.filter(pl.col("team_id") == s)["total_gold"].to_numpy().astype(float) for s in SIDES}
    assert all(g.size == t.size for g in gold.values()), "팀 프레임 시각 불일치"

    per_team = (fp.group_by("game_time_ms", "team_id").agg(
                    pl.col("level").sum().alias("lv"), pl.col("alive").sum().alias("alive"),
                    pl.when(~pl.col("alive")).then(pl.col("respawn_timer")).otherwise(0).sum().alias("resp"))
                .sort("game_time_ms"))
    team_arr = {s: per_team.filter(pl.col("team_id") == s) for s in SIDES}

    def ev(df: pl.DataFrame) -> np.ndarray:
        return df["game_time_ms"].to_numpy().astype(float)

    def building_state(owner: int, btype: str, tier: str | None = None, net: bool = False) -> np.ndarray:
        q = (pl.col("team_id") == owner) & (pl.col("building_type") == btype)
        if tier:
            q &= pl.col("turret_tier") == tier
        down = _count_upto(ev(b.filter(q & (pl.col("event") == "destroyed"))), t)
        if net:
            down -= _count_upto(ev(b.filter(q & (pl.col("event") == "respawned"))), t)
        return down

    def mon(side: int, mtype: str | tuple, extra=None) -> pl.DataFrame:
        types = (mtype,) if isinstance(mtype, str) else mtype
        q = (pl.col("killer_team") == side) & pl.col("monster_type").is_in(types)
        if extra is not None:
            q &= extra
        return m.filter(q)

    baron_s, elder_s = cfg["buffs"]["baron_sec"], cfg["buffs"]["elder_sec"]
    end_ms = float(games["end_game_time_ms"])
    rows = []
    for s in SIDES:
        o = 200 if s == 100 else 100
        stack = {x: _count_upto(ev(mon(x, "dragon", ~pl.col("is_elder"))), t) for x in SIDES}
        y, rate = labels(t, gold[s], end_ms, cfg["g"]["label_horizon_min"], cfg["g"]["censor_rate_window_min"])
        feat = {
            "t_min": t / 60_000, "t_min2": (t / 60_000) ** 2,
            "gold_us": gold[s], "gold_opp": gold[o],
            "level_diff": team_arr[s]["lv"].to_numpy() - team_arr[o]["lv"].to_numpy(),
            **{f"tower_{k}_taken": building_state(o, "turret", k) for k in ("outer", "inner", "base")},
            **{f"tower_{k}_lost": building_state(s, "turret", k) for k in ("outer", "inner", "base")},
            "inhib_down_opp": building_state(o, "inhibitor", net=True),
            "inhib_down_us": building_state(s, "inhibitor", net=True),
            "nexus_tower_down_opp": building_state(o, "turret", "nexus", net=True),
            "nexus_tower_down_us": building_state(s, "turret", "nexus", net=True),
            "dragon_stack_us": stack[s], "dragon_stack_opp": stack[o],
            "soul_us": (stack[s] >= 4).astype(float), "soul_opp": (stack[o] >= 4).astype(float),
            "baron_buff_us": _buff_left(ev(mon(s, "baron")), t, baron_s),
            "baron_buff_opp": _buff_left(ev(mon(o, "baron")), t, baron_s),
            "elder_buff_us": _buff_left(ev(mon(s, "dragon", pl.col("is_elder"))), t, elder_s),
            "elder_buff_opp": _buff_left(ev(mon(o, "dragon", pl.col("is_elder"))), t, elder_s),
            "alive_us": team_arr[s]["alive"].to_numpy().astype(float),
            "alive_opp": team_arr[o]["alive"].to_numpy().astype(float),
            "respawn_sum_us": team_arr[s]["resp"].to_numpy(), "respawn_sum_opp": team_arr[o]["resp"].to_numpy(),
            "voidgrub_us": _count_upto(ev(mon(s, "VoidGrub")), t), "voidgrub_opp": _count_upto(ev(mon(o, "VoidGrub")), t),
            "herald_or_atakhan_us": (_count_upto(ev(mon(s, ("riftHerald", "atakhan"))), t) > 0).astype(float),
            "herald_or_atakhan_opp": (_count_upto(ev(mon(o, ("riftHerald", "atakhan"))), t) > 0).astype(float),
            "side_blue": np.full(t.size, 1.0 if s == 100 else 0.0),
        }
        rows.append(pl.DataFrame({
            "game_id": game_id, "match_id": games["match_id"], "side": s, "t_ms": t.astype(np.int64),
            **{f: np.asarray(feat[f], dtype=float) for f in FEATURES},
            "y": y, "is_end": t == end_ms, "end_rate_per_min": rate,
        }))
    return pl.concat(rows)


def build_one(meta: dict, cfg: dict) -> int:
    df = build_state(meta["game_id"], cfg)
    path = OUT / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return df.height


def game_list() -> list[dict]:
    g = (pl.scan_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True)
         .filter(~pl.col("excluded")).select("game_id", "tournament", "stage").collect())
    return g.to_dicts()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    t0 = time.time()
    with Pool(args.workers) as pool:
        n = sum(pool.imap_unordered(partial(build_one, cfg=cfg), game_list(), chunksize=4))
    print(f"상태 테이블 {n}행, {time.time() - t0:.0f}초 → {OUT}")


if __name__ == "__main__":
    main()
