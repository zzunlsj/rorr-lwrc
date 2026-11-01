"""Step 4 — 클러스터 × 선수 관여 피처 8종 (spec/04 §4.1·§4.3·§4.3b).

입력: Step 3 클러스터·씨앗(data/l2/timeline), L1(frames_player, kills, buildings, monsters, ward_events, ward_kills).
결과: data/l2/alloc/features/tournament=<t>/stage=<s>/<game_id>.parquet — (k, participant_id) 한 행.

측정 규칙:
- dmg / cc / protect: 프레임 누적값의 클러스터 틱 [i_s, i_e] 증가분. protect = 실드 + 힐.
- assist: 클러스터 안 champion_kill의 assistants 등장 횟수. 처형(killer 0)은 배분 입력에서 제외 (결정 26).
- absorb: Σ max(0, health[t] − health[t+1]), 최대 체력이 바뀐 프레임 쌍과 부활(사망 → 생존) 프레임 쌍 제외.
- presence: 클러스터 틱 중 "생존 + 어떤 씨앗 위치든 반경 presence_radius 안"인 틱 비율 (OR 판정).
- vision_provided: 시야창 [s − 30초, e] ∩ 와드 생존 구간 (초), 본인 설치·시야 대상 종류·위치가 시야 구역(씨앗 반경 vision_radius 합집합) 안.
- vision_denied: 시야창 동안 시야 구역 안에서 본인이 제거한 상대 와드 수 (시야 대상 종류만, 결정 16).
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

TL = L2 / "timeline"
OUT = L2 / "alloc" / "features"
FEATURES = ["dmg", "assist", "absorb", "presence", "vision_provided", "vision_denied", "cc", "protect"]


def _read(base, table: str, meta: dict) -> pl.DataFrame:
    return pl.read_parquet(base / table / f"tournament={meta['tournament']}" / "**" / f"{meta['game_id']}.parquet",
                           hive_partitioning=False)


def seed_positions(meta: dict, cfg: dict) -> pl.DataFrame:
    """씨앗 이벤트의 (seq, x, z) — Step 3 씨앗과 seq로 연결."""
    k = _read(L1, "kills", meta).select("seq", "x", "z")
    b = _read(L1, "buildings", meta).filter(pl.col("event") == "destroyed").select("seq", "x", "z")
    m = (_read(L1, "monsters", meta).filter(pl.col("monster_type").is_in(cfg["cluster"]["seed_monsters"]))
         .select("seq", "x", "z"))
    return pl.concat([k, b, m], how="vertical_relaxed")


def near_any(px: np.ndarray, pz: np.ndarray, sx: np.ndarray, sz: np.ndarray, r: float) -> np.ndarray:
    """점들(px, pz) 각각이 씨앗 중 하나라도 반경 r 안인지."""
    if sx.size == 0:
        return np.zeros(px.size, dtype=bool)
    d2 = (px[:, None] - sx[None, :]) ** 2 + (pz[:, None] - sz[None, :]) ** 2
    return (d2 <= r * r).any(axis=1)


def absorb_steps(health: np.ndarray, hmax: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """틱 쌍 (t, t+1)별 흡수량. 최대 체력 변화·부활 쌍은 0."""
    drop = np.clip(health[:-1] - health[1:], 0, None)
    bad = (hmax[:-1] != hmax[1:]) | (~alive[:-1] & alive[1:])
    return np.where(bad, 0.0, drop)


def build_features(meta: dict, cfg: dict) -> pl.DataFrame:
    acfg = cfg["allocation"]
    clusters = _read(TL, "clusters", meta).filter(pl.col("side") == 100)
    seeds = _read(TL, "seeds", meta).join(seed_positions(meta, cfg), on="seq", how="left")
    assert seeds["x"].null_count() == 0, "씨앗 위치 결측"
    fp = _read(L1, "frames_player", meta).sort("participant_id", "game_time_ms")
    kills = _read(L1, "kills", meta).filter(pl.col("killer") > 0)
    wards = _read(L1, "ward_events", meta).filter(pl.col("vision_eligible"))
    vision_types = acfg["vision_ward_types"]
    wkills = (_read(L1, "ward_kills", meta).filter(pl.col("ward_type").is_in(vision_types))
              .join(_read(L1, "ward_events", meta).select("ward_id", pl.col("side").alias("ward_side")), on="ward_id", how="left"))

    pids = sorted(fp["participant_id"].unique().to_list())
    team_of = dict(fp.select("participant_id", "team_id").unique().iter_rows())
    per = {p: fp.filter(pl.col("participant_id") == p) for p in pids}
    t = per[pids[0]]["game_time_ms"].to_numpy()
    arr = {p: {c: d[c].to_numpy() for c in ("dmg_to_champions", "cc_to_champions", "shielded_teammates",
                                             "healed_teammates", "x", "z", "health", "health_max", "alive")}
           for p, d in per.items()}
    for p in pids:
        assert np.array_equal(per[p]["game_time_ms"].to_numpy(), t), "선수 프레임 시각 불일치"
        arr[p]["absorb"] = np.concatenate([[0.0], np.cumsum(absorb_steps(
            arr[p]["health"].astype(float), arr[p]["health_max"].astype(float), arr[p]["alive"]))])

    r_p, r_v, pre = acfg["presence_radius"], acfg["vision_radius"], acfg["vision_pre_window_sec"] * 1000
    rows = []
    for c in clusters.iter_rows(named=True):
        i_s, i_e = int(np.searchsorted(t, c["t_s_ms"])), int(np.searchsorted(t, c["t_e_ms"]))
        sj = seeds.filter(pl.col("k") == c["k"])
        sx, sz = sj["x"].to_numpy().astype(float), sj["z"].to_numpy().astype(float)
        kj = kills.filter(pl.col("game_time_ms").is_between(c["s_ms"], c["e_ms"]))
        assists = [a for lst in kj["assistants"].to_list() for a in (lst or [])]
        w0, w1 = c["s_ms"] - pre, c["e_ms"]
        wz = wards.filter(near_any(wards["x"].to_numpy().astype(float), wards["z"].to_numpy().astype(float), sx, sz, r_v))
        overlap = (pl.min_horizontal(pl.col("t_removed_ms"), pl.lit(w1)) - pl.max_horizontal(pl.col("t_placed_ms"), pl.lit(w0))).clip(lower_bound=0) / 1000
        provided = dict(wz.group_by("placer").agg(overlap.sum()).iter_rows())
        kz = wkills.filter(pl.col("game_time_ms").is_between(w0, w1) & (pl.col("ward_side") != pl.col("killer_side")))
        kz = kz.filter(near_any(kz["x"].to_numpy().astype(float), kz["z"].to_numpy().astype(float), sx, sz, r_v))
        denied = dict(kz.group_by("killer").len().iter_rows())
        span = slice(i_s, i_e + 1)
        for p in pids:
            a = arr[p]
            inside = a["alive"][span] & near_any(a["x"][span].astype(float), a["z"][span].astype(float), sx, sz, r_p)
            rows.append({
                "k": c["k"], "participant_id": p, "team_id": team_of[p],
                "dmg": float(a["dmg_to_champions"][i_e] - a["dmg_to_champions"][i_s]),
                "assist": float(assists.count(p)),
                "absorb": float(a["absorb"][i_e] - a["absorb"][i_s]),
                "presence": float(inside.mean()),
                "vision_provided": float(provided.get(p, 0.0)),
                "vision_denied": float(denied.get(p, 0)),
                "cc": float(a["cc_to_champions"][i_e] - a["cc_to_champions"][i_s]),
                "protect": float(a["shielded_teammates"][i_e] - a["shielded_teammates"][i_s]
                                 + a["healed_teammates"][i_e] - a["healed_teammates"][i_s]),
                "kill_killer": int((kj["killer"] == p).sum()),
            })
    return pl.DataFrame(rows).with_columns(pl.lit(meta["game_id"]).alias("game_id"))


def build_one(meta: dict, cfg: dict) -> int:
    df = build_features(meta, cfg)
    path = OUT / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return df.height


def game_list() -> list[dict]:
    return (pl.scan_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True)
            .filter(~pl.col("excluded")).select("game_id", "tournament", "stage").collect().to_dicts())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    t0 = time.time()
    with Pool(args.workers) as pool:
        n = sum(pool.imap_unordered(partial(build_one, cfg=cfg), game_list(), chunksize=4))
    print(f"관여 피처 {n}행, {time.time() - t0:.0f}초 → {OUT}")


if __name__ == "__main__":
    main()
