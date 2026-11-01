"""LwRC / LwRC+ 산출 파이프라인 (경기 단위 스트리밍) → L3 + QC.

사용: uv run python -m metrics.run --config v1_0
출력: data/l3/lwrc/spec=<spec_version>/{players,ledger,clusters,denom,diagnostics}.parquet
      manifest/qc_results.csv (해당 spec_version 행 교체)
"""

import argparse
import json
import resource
import time

import numpy as np
import polars as pl

from common import config
from common.paths import L3
from features import g_models
from features.clusters import DRIFT, assign_intervals, build_clusters, seeds_of
from features.state import SIDES, list_games, load_state
from metrics import lwrc_plus, qc
from metrics.attribution import allocator_from_config, attribute

UNRESOLVED = "extra_metrics=미확정"


def run_game(game_id: str, cfg: dict, gmodel, allocator):
    state = load_state(game_id)
    G = {s: gmodel.evaluate(state, s) for s in SIDES}
    c = cfg["cluster"]
    clusters = build_clusters(seeds_of(state), state.t[0], state.t_end, c["window_sec"], c["merge_sec"], c["merge_rule"])
    bucket, hits = assign_intervals(state.t, clusters)
    ledger_rows, diag_rows = attribute(state, G, clusters, bucket, allocator)
    ledger = pl.DataFrame(ledger_rows, schema={"side": pl.Int64, "participant_id": pl.Int64, "channel": pl.Utf8,
                                                 "source": pl.Utf8, "amount": pl.Float64})

    part = qc.partition_check(state.t, bucket, hits, clusters, c["warn_len_sec"])
    ident = qc.identity_check(ledger, G, state.team_gold, cfg["identity"]["epsilon"])

    # 선수 행
    pivot = (ledger.filter(pl.col("participant_id") >= 0)
             .group_by("participant_id", "channel").agg(pl.col("amount").sum())
             .pivot(on="channel", index="participant_id", values="amount"))
    for ch in ("C", "D", "Risk"):
        if ch not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0.0).alias(ch))
    roster = pl.DataFrame({
        "participant_id": state.pids.astype(int), "side": state.p_team.astype(int), "role": state.p_role,
        "player_name": state.p_name, "champion": state.p_champ,
        "realized_gold": state.p_gold[-1] - state.p_gold[0],
    })
    players = (roster.join(pivot.select("participant_id", "C", "D", "Risk"), on="participant_id", how="left")
               .with_columns(pl.col("C", "D", "Risk").fill_null(0.0))
               .with_columns((pl.col("C") + pl.col("D") + pl.col("Risk")).alias("lwrc"),
                             pl.lit(state.minutes).alias("minutes"))
               .with_columns((pl.col("lwrc") / pl.col("minutes")).alias("rate")))

    # 진단: |ΔG|/초 분포 (클러스터 안/밖), D/C, 후반 10분 D
    dt = np.diff(state.t) / 1000.0
    diag = pl.DataFrame(diag_rows) if diag_rows else pl.DataFrame()
    late_cut = state.t_end - 10 * 60000
    qc_rows, dg_rates = [], []
    for side in SIDES:
        rate = np.abs(np.diff(G[side])) / dt
        dg_rates.append(pl.DataFrame({"side": side, "in_cluster": bucket != DRIFT, "abs_dg_per_s": rate}))
        own = ledger.filter(pl.col("side") == side)
        s = lambda ch: float(own.filter(pl.col("channel") == ch)["amount"].sum())
        d_side = diag.filter(pl.col("side") == side) if diag.height else diag
        late_d = early_d = 0.0
        if d_side.height:
            d_pie = (-d_side["dG_opp"]).clip(lower_bound=0) * (d_side["sum_w"] > 0)
            mid = (d_side["start_ms"] + d_side["end_ms"]) / 2
            late_d, early_d = float(d_pie.filter(mid >= late_cut).sum()), float(d_pie.filter(mid < late_cut).sum())
        sid = next(i for i in ident if i["side"] == side)
        qc_rows.append({
            "game_id": game_id, "tournament": state.tournament, "stage": state.stage, "patch": state.patch,
            "side": side, **{k: v for k, v in sid.items() if k != "side"}, **part,
            "sum_C": s("C"), "sum_D": s("D"), "sum_Risk": s("Risk"), "team_bucket": s("team_bucket"),
            "D_unattributed": s("D_unattributed"),
            "d_over_c": s("D") / s("C") if s("C") else float("nan"),
            "d_unattr_share": s("D_unattributed") / (s("D") + s("D_unattributed")) if (s("D") + s("D_unattributed")) else 0.0,
            "late10_D_per_min": late_d / 10.0,
            "early_D_per_min": early_d / max(state.minutes - 10, 1e-9),
            "minutes": state.minutes,
        })
    return players.with_columns(pl.lit(game_id).alias("game_id")), ledger.with_columns(pl.lit(game_id).alias("game_id")), \
        diag.with_columns(pl.lit(game_id).alias("game_id")) if diag.height else diag, \
        pl.concat(dg_rates).with_columns(pl.lit(game_id).alias("game_id")), qc_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v1_0")
    args = ap.parse_args()
    cfg = config.load(args.config)
    spec, snapshot = cfg["spec_version"], config.data_snapshot()
    gmodel, allocator = g_models.from_config(cfg), allocator_from_config(cfg)

    games = list_games()
    tick_ok = dict(zip(games["game_id"], games["tick_ok"]))
    all_players, all_ledger, all_diag, all_dg, all_qc, timings = [], [], [], [], [], []
    for game_id in games["game_id"]:
        t0 = time.perf_counter()
        players, ledger, diag, dg, qc_rows = run_game(game_id, cfg, gmodel, allocator)
        timings.append(time.perf_counter() - t0)
        game_pass = tick_ok[game_id] and all(r["identity_ok"] and r["partition_ok"] and r["g_end_equals_gold"] for r in qc_rows)
        for r in qc_rows:
            r.update({"tick_ok": tick_ok[game_id], "qc_pass": game_pass, "spec_version": spec, "data_snapshot": snapshot})
        all_qc += qc_rows
        all_players.append(players.with_columns(pl.lit(game_pass).alias("qc_pass")))
        all_ledger.append(ledger)
        all_diag.append(diag)
        all_dg.append(dg)

    players = pl.concat(all_players)
    failed = players.filter(~pl.col("qc_pass"))["game_id"].unique().to_list()
    pool = players.filter(pl.col("qc_pass"))           # QC 실패 경기는 L3에서 제외
    denom = lwrc_plus.denominators(pool, cfg)
    players = lwrc_plus.apply(pool, denom)
    tag = [pl.lit(spec).alias("spec_version"), pl.lit(snapshot).alias("data_snapshot")]
    players = players.with_columns(*tag, pl.lit(UNRESOLVED).alias("unresolved"))

    out = L3 / "lwrc" / f"spec={spec}"
    out.mkdir(parents=True, exist_ok=True)
    players.write_parquet(out / "players.parquet")
    pl.concat(all_ledger).with_columns(*tag).write_parquet(out / "ledger.parquet")
    pl.concat([d for d in all_diag if d.height]).with_columns(*tag).write_parquet(out / "clusters.parquet")
    denom.with_columns(*tag).write_parquet(out / "denom.parquet")
    pl.concat(all_dg).write_parquet(out / "dg_rates.parquet")
    qc.write_qc(all_qc, spec)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes
    summary = {"spec_version": spec, "games": len(timings), "qc_failed_games": failed,
               "sec_per_game_mean": float(np.mean(timings)), "sec_per_game_max": float(np.max(timings)),
               "peak_rss_mb": peak_mb}
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
