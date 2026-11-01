"""교전 구역의 와드 — 누가 설치했고 누가 지웠나 (2026-11-01 Jun 요청).

교전 구역·시간은 시야 피처(alloc_features)와 같다: 씨앗 반경 vision_radius(3,300) 합집합, 시야창 [s − 30초, e].
  설치: 구역 안 + 생존 구간이 시야창과 겹치는 와드 (시야 대상 종류, 결정 16) → 설치자 포지션·종류·겹친 초·창 시작 전 설치 여부
  제거: 시야창 동안 구역 안에서 상대 와드를 지운 기록 → 제거자 포지션·종류·창 안에서 없앤 시야(초)
        없앤 시야(초) = min(와드 예상 만료, 창 끝) − 제거 시각 (만료 없는 종류는 창 끝까지)
비교 기준: 게임 전체의 시야 대상 와드 설치·제거 (포지션 몫).

결과: outputs/fight_ward_check.json, data/l2/vision_score/fight_wards.parquet · fight_ward_kills.parquet
사용: PYTHONPATH=src uv run python -m report.fight_ward_check [--workers 6]
"""

import argparse
import json
from functools import partial
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml

from common.paths import L1, L2, OUTPUTS
from features.alloc_features import TL, _read, game_list, near_any, seed_positions
from ingest.leaguepedia import LWRC_YAML

OUT_DIR = L2 / "vision_score"


def one_game(meta: dict, cfg: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    acfg = cfg["allocation"]
    r_v, pre = acfg["vision_radius"], acfg["vision_pre_window_sec"] * 1000
    clusters = _read(TL, "clusters", meta).filter(pl.col("side") == 100)
    seeds = _read(TL, "seeds", meta).join(seed_positions(meta, cfg), on="seq", how="left")
    roles = _read(L1, "participants", meta).select("participant_id", pl.col("role").replace_strict(cfg["season"]["role_codes"]))
    we = _read(L1, "ward_events", meta)
    wards = we.filter(pl.col("vision_eligible"))
    wk = (_read(L1, "ward_kills", meta).filter(pl.col("ward_type").is_in(acfg["vision_ward_types"]))
          .join(we.select("ward_id", pl.col("side").alias("ward_side"), "t_placed_ms", "duration_s"), on="ward_id", how="left")
          .filter(pl.col("ward_side") != pl.col("killer_side")))
    pw, pk = [], []
    for c in clusters.iter_rows(named=True):
        sj = seeds.filter(pl.col("k") == c["k"])
        sx, sz = sj["x"].to_numpy().astype(float), sj["z"].to_numpy().astype(float)
        w0, w1 = c["s_ms"] - pre, c["e_ms"]
        wz = wards.filter(near_any(wards["x"].to_numpy().astype(float), wards["z"].to_numpy().astype(float), sx, sz, r_v))
        wz = wz.with_columns(((pl.min_horizontal(pl.col("t_removed_ms"), pl.lit(w1)) - pl.max_horizontal(pl.col("t_placed_ms"), pl.lit(w0)))
                              .clip(lower_bound=0) / 1000).alias("overlap_s")).filter(pl.col("overlap_s") > 0)
        pw.append(wz.select("ward_id", "ward_type", "placer", "side", "overlap_s", (pl.col("t_placed_ms") < w0).alias("pre_placed"),
                            (pl.col("t_placed_ms") >= c["s_ms"]).alias("placed_in_fight")).with_columns(pl.lit(c["k"]).alias("k")))
        kz = wk.filter(pl.col("game_time_ms").is_between(w0, w1))
        kz = kz.filter(near_any(kz["x"].to_numpy().astype(float), kz["z"].to_numpy().astype(float), sx, sz, r_v))
        expiry = pl.when(pl.col("duration_s").is_not_null()).then(pl.col("t_placed_ms") + pl.col("duration_s") * 1000).otherwise(pl.lit(w1))
        pk.append(kz.select("ward_id", "ward_type", "killer", "killer_side",
                            ((pl.min_horizontal(expiry, pl.lit(w1)) - pl.col("game_time_ms")).clip(lower_bound=0) / 1000).alias("denied_s"))
                  .with_columns(pl.lit(c["k"]).alias("k")))
    gid = pl.lit(meta["game_id"]).alias("game_id")
    fw = pl.concat(pw, how="vertical_relaxed").join(roles, left_on="placer", right_on="participant_id", how="left").with_columns(gid) if pw else pl.DataFrame()
    fk = pl.concat(pk, how="vertical_relaxed").join(roles, left_on="killer", right_on="participant_id", how="left").with_columns(gid) if pk else pl.DataFrame()
    allw = wards.join(roles, left_on="placer", right_on="participant_id", how="left").select("ward_type", "role").with_columns(gid)
    allk = wk.join(roles, left_on="killer", right_on="participant_id", how="left").select("ward_type", "role").with_columns(gid)
    return fw, fk, allw, allk


def share(df: pl.DataFrame, by: str, val: str | None = None) -> dict:
    agg = (pl.col(val).sum() if val else pl.len()).alias("v")
    g = df.group_by(by).agg(agg)
    tot = g["v"].sum()
    return {r[by]: 100 * r["v"] / tot for r in g.to_dicts()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    with Pool(args.workers) as pool:
        res = pool.map(partial(one_game, cfg=cfg), game_list(), chunksize=4)
    fw = pl.concat([r[0] for r in res if r[0].height], how="vertical_relaxed")
    fk = pl.concat([r[1] for r in res if r[1].height], how="vertical_relaxed")
    allw = pl.concat([r[2] for r in res], how="vertical_relaxed")
    allk = pl.concat([r[3] for r in res], how="vertical_relaxed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fw.write_parquet(OUT_DIR / "fight_wards.parquet")
    fk.write_parquet(OUT_DIR / "fight_ward_kills.parquet")
    n_games, n_cl = len(res), int(fw.select(pl.struct("game_id", "k").n_unique()).item())
    types = cfg["allocation"]["vision_ward_types"]
    out = {
        "n_games": n_games,
        "placed": {
            "fight_rows": fw.height, "fight_unique_wards": int(fw.select(pl.struct("game_id", "ward_id").n_unique()).item()),
            "share_count_pct": share(fw, "role"), "share_seconds_pct": share(fw, "role", "overlap_s"),
            "game_share_pct": share(allw, "role"),
            "per_game": {r["role"]: r["n"] / n_games for r in fw.group_by("role").agg(pl.len().alias("n")).to_dicts()},
            "pre_placed_pct": {r["role"]: 100 * r["p"] for r in fw.group_by("role").agg(pl.col("pre_placed").mean().alias("p")).to_dicts()},
            "mean_overlap_s": {r["role"]: r["s"] for r in fw.group_by("role").agg(pl.col("overlap_s").mean().alias("s")).to_dicts()},
            "type_by_role_seconds": {r["role"]: {t: r[t] for t in types} for r in fw.group_by("role").agg(
                *[pl.col("overlap_s").filter(pl.col("ward_type") == t).sum().alias(t) for t in types]).to_dicts()},
            "type_by_role_count": {r["role"]: {t: r[t] for t in types} for r in fw.group_by("role").agg(
                *[(pl.col("ward_type") == t).sum().alias(t) for t in types]).to_dicts()},
        },
        "killed": {
            "fight_kills": fk.height, "share_count_pct": share(fk, "role"), "share_denied_s_pct": share(fk, "role", "denied_s"),
            "game_share_pct": share(allk, "role"),
            "per_game": {r["role"]: r["n"] / n_games for r in fk.group_by("role").agg(pl.len().alias("n")).to_dicts()},
            "game_per_game": {r["role"]: r["n"] / n_games for r in allk.group_by("role").agg(pl.len().alias("n")).to_dicts()},
            "mean_denied_s": {r["role"]: r["s"] for r in fk.group_by("role").agg(pl.col("denied_s").mean().alias("s")).to_dicts()},
            "type_by_role_count": {r["role"]: {t: r[t] for t in types} for r in fk.group_by("role").agg(
                *[(pl.col("ward_type") == t).sum().alias(t) for t in types]).to_dicts()},
        },
        "game_placed_per_game": {r["role"]: r["n"] / n_games for r in allw.group_by("role").agg(pl.len().alias("n")).to_dicts()},
        "fight_capture_pct": {},
    }
    # 게임 전체 설치·제거 중 교전 구역에 들어간 비율 (와드 단위, 중복 교전 제거)
    fw_u = fw.unique(["game_id", "ward_id"]).group_by("role").agg(pl.len().alias("f"))
    aw = allw.group_by("role").agg(pl.len().alias("a"))
    fk_u = fk.unique(["game_id", "ward_id", "killer"]).group_by("role").agg(pl.len().alias("f"))
    ak = allk.group_by("role").agg(pl.len().alias("a"))
    out["fight_capture_pct"] = {"placed": {r["role"]: 100 * r["f"] / r["a"] for r in fw_u.join(aw, on="role").to_dicts()},
                                "killed": {r["role"]: 100 * r["f"] / r["a"] for r in fk_u.join(ak, on="role").to_dicts()}}
    out["n_cluster_with_wards"] = n_cl
    (OUTPUTS / "fight_ward_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
