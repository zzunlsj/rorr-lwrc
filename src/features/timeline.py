"""Step 3 — 클러스터·드리프트 분할, 데스별 Risk, 클러스터 파이 (spec/02, spec/03 §3.1·§3.2·§3.4).

입력: L1(kills, buildings, monsters, frames_player, games), Step 2 산출 data/l2/g (1Hz G, fold), g_model/coefficients.
결과: data/l2/timeline/<table>/tournament=<t>/stage=<s>/<game_id>.parquet
  clusters  (k, side)   — 구간, 씨앗 수, ΔG, 실현 골드, ΔV, 시간항, Risk 합, 잔여, C·D 파이, 팀 계정 몫 (+ 기존 ΔG 파이 비교값)
  seeds     씨앗 이벤트 — 소속 클러스터 k, 종류, 이득 팀
  deaths    데스별 Risk — 피해자, 처형 여부, 부활 대기, 현상금
  drift     (side, 구간) — ΔG, 선수 실현 골드 합, 팀 계정 몫
  gold_player (participant) — 실현 골드 (클러스터 / 드리프트), 본인 귀속
  summary   (side) — G(0), G(END), 성분 합, 자팀 항등식 잔차, D 교차검증 잔차, 커버리지

구간 규칙:
- 클러스터 [s, e]는 s·e에 가장 가까운 틱 [i_s, i_e]로 옮긴다 (spec/02 §2.2). 드리프트는 그 보집합 틱 구간.
  틱 구간 (t_{i-1}, t_i]는 정확히 하나의 버킷에 속한다 (CLAUDE.md §6-4).
- G(END) := y(END) (spec/01 §1.4). 마지막 틱의 G는 관측 레이블로 바꿔 쓴다.
- 속도형 (결정 27): G = 현재 골드 + V(앞으로 10분 기대 수입). 실현 골드는 클러스터·드리프트 모두 번 선수 본인에게,
  클러스터 파이는 ΔV(= ΔG − 팀 실현 골드)에서 Risk·시간항·결정적 감쇠를 뺀 잔여로 만든다. C = 우리 가속, D = 상대 감속.
- 결정적 감쇠 (결정 28): 부활 대기·버프 잔여의 초당 감소와 부활(생존 +1)은 사건 없이 시간만으로 일어난다.
  사건 가치는 사건 순간에 매기므로(Risk·버프 처치), 이 몫은 시간항과 함께 팀 계정으로 보낸다.
- Risk는 선형 G이므로 Δx · β (게임의 fold 모델): 생존 −1, 부활 대기 +respawn_timer, 상대 골드 +현상금 (spec/03 §3.1).
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

G_DIR = L2 / "g"
MODEL_DIR = L2 / "g_model"
OUT = L2 / "timeline"
SIDES = (100, 200)


def _read(base, table: str, game_id: str) -> pl.DataFrame:
    return pl.read_parquet(base / table / "**" / f"{game_id}.parquet", hive_partitioning=False)


def opp(side: int) -> int:
    return 200 if side == 100 else 100


def seeds_of(kills: pl.DataFrame, buildings: pl.DataFrame, monsters: pl.DataFrame, seed_monsters: list[str]) -> pl.DataFrame:
    """씨앗 = champion_kill 전부 + building_destroyed + 에픽 몬스터 (spec/02 §2.1). credited_team = 이득 팀."""
    k = kills.select(pl.col("game_time_ms").alias("t_ms"), pl.lit("kill").alias("kind"), "seq",
                     pl.col("killer_team").alias("credited_team"))
    b = (buildings.filter(pl.col("event") == "destroyed")
         .select(pl.col("game_time_ms").alias("t_ms"), pl.lit("building").alias("kind"), "seq",
                 pl.when(pl.col("team_id") == 100).then(200).otherwise(100).alias("credited_team")))
    m = (monsters.filter(pl.col("monster_type").is_in(seed_monsters))
         .select(pl.col("game_time_ms").alias("t_ms"), pl.lit("epic").alias("kind"), "seq",
                 pl.col("killer_team").alias("credited_team")))
    return pl.concat([k, b, m], how="vertical_relaxed").sort("t_ms", "seq")


def build_clusters(seed_ms: np.ndarray, t0: float, t1: float, window_sec: float, merge_sec: float,
                   merge_rule: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """씨앗별 클러스터 번호와 클러스터 구간 [s, e] (게임 시작·종료에서 절단)."""
    w, gap = window_sec * 1000, merge_sec * 1000
    if merge_rule == "window_gap":
        max_seed_gap = 2 * w + gap
    else:
        raise ValueError(f"알 수 없는 cluster.merge_rule: {merge_rule}")
    if seed_ms.size == 0:
        return np.zeros(0, dtype=int), []
    k = np.concatenate([[0], np.cumsum(np.diff(seed_ms) > max_seed_gap)])
    spans = [(max(t0, seed_ms[k == j].min() - w), min(t1, seed_ms[k == j].max() + w)) for j in range(k[-1] + 1)]
    return k, spans


def tick_spans(t: np.ndarray, spans: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """클러스터 [s, e] → 가장 가까운 틱 인덱스 [i_s, i_e]. 겹치면 분할이 깨지므로 중단."""
    def nearest(x):
        i = int(np.clip(np.searchsorted(t, x), 1, t.size - 1))
        return i - 1 if x - t[i - 1] <= t[i] - x else i
    out = [(nearest(s), nearest(e)) for s, e in spans]
    for (a0, a1), (b0, b1) in zip(out, out[1:]):
        assert a1 <= b0, f"클러스터 틱 구간 겹침: {(a0, a1)} {(b0, b1)}"
    return out


def drift_spans(n: int, cl: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """클러스터 보집합의 틱 구간 [a, b] (a < b만)."""
    edges, prev = [], 0
    for i_s, i_e in cl:
        if i_s > prev:
            edges.append((prev, i_s))
        prev = max(prev, i_e)
    if prev < n - 1:
        edges.append((prev, n - 1))
    return edges


def death_risk(kills: pl.DataFrame, fp: pl.DataFrame, beta: dict[str, float]) -> pl.DataFrame:
    """데스별 Risk = β_alive·(−1) + β_respawn·respawn_timer + β_gold_opp·(bounty + shutdown) — 피해자 팀 G 기준."""
    rows = []
    for r in kills.sort("game_time_ms", "seq").iter_rows(named=True):
        v = fp.filter((pl.col("participant_id") == r["victim"]) & (pl.col("game_time_ms") >= r["game_time_ms"])
                      & ~pl.col("alive")).head(1)
        observed = v.height == 1
        rt = float(v["respawn_timer"][0]) if observed else None
        bounty = float(r["bounty"] or 0) + float(r["shutdown_bounty"] or 0)
        risk = -beta["alive_us"] + beta["gold_opp"] * bounty + (beta["respawn_sum_us"] * rt if observed else 0.0)
        rows.append({"kill_seq": r["seq"], "t_ms": r["game_time_ms"], "victim": r["victim"], "victim_team": r["victim_team"],
                     "killer": r["killer"], "is_execution": r["killer"] == 0, "respawn_timer": rt,
                     "respawn_observed": observed, "bounty_total": bounty, "risk": risk})
    schema = {"kill_seq": pl.Int64, "t_ms": pl.Int64, "victim": pl.Int64, "victim_team": pl.Int64, "killer": pl.Int64,
              "is_execution": pl.Boolean, "respawn_timer": pl.Float64, "respawn_observed": pl.Boolean,
              "bounty_total": pl.Float64, "risk": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def decay_contrib(x: pl.DataFrame, beta: dict[str, float], feats: dict[str, list[str]]) -> np.ndarray:
    """결정적 감쇠의 누적 G 기여 (결정 28). 틱 간 감소분(부활 대기·버프) / 증가분(부활) × β, 0부터 누적."""
    step = np.zeros(x.height - 1)
    for f in feats["decreasing"]:
        step += beta[f] * np.clip(np.diff(x[f].to_numpy()), None, 0)
    for f in feats["increasing"]:
        step += beta[f] * np.clip(np.diff(x[f].to_numpy()), 0, None)
    return np.concatenate([[0.0], np.cumsum(step)])


def build_timeline(meta: dict, cfg: dict, coefs: dict[str, dict[str, float]]) -> dict[str, pl.DataFrame]:
    gid = meta["game_id"]
    ccfg = cfg["cluster"]
    g = pl.read_parquet(G_DIR / f"tournament={meta['tournament']}" / f"{gid}.parquet")
    state = pl.read_parquet(L2 / "state" / f"tournament={meta['tournament']}" / "**" / f"{gid}.parquet", hive_partitioning=False)
    beta = coefs[str(g["fold"][0])]
    kills = _read(L1, "kills", gid)
    fp = _read(L1, "frames_player", gid).sort("game_time_ms", "participant_id")

    G, TT, DEC = {}, {}, {}
    for s in SIDES:
        gs = g.filter(pl.col("side") == s).sort("t_ms")
        t = gs["t_ms"].to_numpy().astype(float)
        G[s] = gs["G"].to_numpy().copy()
        G[s][-1] = gs["y"][-1]                       # G(END) := y(END)
        assert gs["is_end"][-1], "마지막 틱이 종료 틱이 아님"
        TT[s] = gs["G_time_term"].to_numpy()
        DEC[s] = decay_contrib(state.filter(pl.col("side") == s).sort("t_ms"), beta, ccfg["decay_features"])
        assert DEC[s].size == t.size, "상태 테이블과 G 틱 수 불일치"
    n = t.size

    seeds = seeds_of(kills, _read(L1, "buildings", gid), _read(L1, "monsters", gid), ccfg["seed_monsters"])
    seeds = seeds.filter(pl.col("t_ms") <= t[-1])
    k, spans = build_clusters(seeds["t_ms"].to_numpy().astype(float), t[0], t[-1],
                              ccfg["window_sec"], ccfg["merge_sec"], ccfg["merge_rule"])
    seeds = seeds.with_columns(pl.Series("k", k, dtype=pl.Int64))
    cl = tick_spans(t, spans)

    deaths = death_risk(kills, fp, beta)
    ks = seeds.filter(pl.col("kind") == "kill")
    kill_k = dict(zip(ks["seq"].to_list(), ks["k"].to_list()))
    deaths = deaths.with_columns(pl.col("kill_seq").replace_strict(kill_k, default=None, return_dtype=pl.Int64).alias("k"))

    gold = {pid: d.sort("game_time_ms")["total_gold"].to_numpy().astype(float)
            for (pid,), d in fp.group_by(["participant_id"])}
    team_of = dict(fp.select("participant_id", "team_id").unique().iter_rows())
    assert all(v.size == n for v in gold.values()), "선수 프레임 틱 수 불일치"
    pids = {x: [p for p in gold if team_of[p] == x] for x in SIDES}
    realized_player = {p: {"cluster": 0.0, "drift": 0.0} for p in gold}

    def realize(a: int, b: int, where: str) -> dict[int, float]:
        """구간 [a, b]의 선수 실현 골드를 본인에게 적립하고 팀 합을 돌려준다."""
        for p in gold:
            realized_player[p][where] += gold[p][b] - gold[p][a]
        return {x: sum(gold[p][b] - gold[p][a] for p in pids[x]) for x in SIDES}

    # 클러스터 파이 — 속도형 (결정 27, spec/03 §3.2): ΔV = ΔG − 실현 골드
    cl_rows = []
    for j, ((s_ms, e_ms), (i_s, i_e)) in enumerate(zip(spans, cl)):
        sj = seeds.filter(pl.col("k") == j)
        real = realize(i_s, i_e, "cluster")
        resid, legacy, base = {}, {}, {}
        for s in SIDES:
            dG = G[s][i_e] - G[s][i_s]
            tt = TT[s][i_e] - TT[s][i_s]
            dec = DEC[s][i_e] - DEC[s][i_s]
            rk = float(deaths.filter((pl.col("k") == j) & (pl.col("victim_team") == s))["risk"].sum())
            resid[s] = dG - real[s] - rk - tt - dec
            legacy[s] = dG - rk - tt                        # 기존 ΔG 파이 (비교 기준선)
            base[s] = {"dG": dG, "gold_sum": real[s], "dV": dG - real[s], "time_term": tt, "decay_term": dec, "risk_sum": rk}
        for s in SIDES:
            cl_rows.append({
                "k": j, "side": s, "s_ms": s_ms, "e_ms": e_ms, "t_s_ms": int(t[i_s]), "t_e_ms": int(t[i_e]),
                "len_s": (e_ms - s_ms) / 1000, "n_seeds": sj.height,
                "n_kills": int((sj["kind"] == "kill").sum()), "n_buildings": int((sj["kind"] == "building").sum()),
                "n_epics": int((sj["kind"] == "epic").sum()),
                **base[s], "resid": resid[s], "c_pie": max(resid[s], 0.0), "d_pie": max(-resid[opp(s)], 0.0),
                "team_acct": min(resid[s], 0.0) + base[s]["time_term"] + base[s]["decay_term"],
                "c_pie_legacy": max(legacy[s], 0.0), "d_pie_legacy": max(-legacy[opp(s)], 0.0),
            })
    clusters = pl.DataFrame(cl_rows)

    # 드리프트 (spec/03 §3.4): 실현 골드는 본인, 나머지는 팀 계정
    drift_rows = []
    for a, b in drift_spans(n, cl):
        real = realize(a, b, "drift")
        for s in SIDES:
            dG = G[s][b] - G[s][a]
            drift_rows.append({"side": s, "t_a_ms": int(t[a]), "t_b_ms": int(t[b]), "dG": dG,
                               "gold_sum": real[s], "team_acct": dG - real[s]})
    drift = pl.DataFrame(drift_rows, schema={"side": pl.Int64, "t_a_ms": pl.Int64, "t_b_ms": pl.Int64, "dG": pl.Float64,
                                             "gold_sum": pl.Float64, "team_acct": pl.Float64})
    gold_player = pl.DataFrame({"participant_id": list(realized_player), "team_id": [team_of[p] for p in realized_player],
                                "gold_cluster": [v["cluster"] for v in realized_player.values()],
                                "gold_drift": [v["drift"] for v in realized_player.values()]})

    # 자팀 항등식: 실현 골드 + ΣC파이 + ΣRisk + 팀 계정 = G(END) − G(0)
    # D 교차검증: 우리 D 파이 총량 = 상대 클러스터 잔여의 음수 부분 총량
    summ = []
    for s in SIDES:
        c = clusters.filter(pl.col("side") == s)
        co = clusters.filter(pl.col("side") == opp(s))
        d = drift.filter(pl.col("side") == s)
        total = G[s][-1] - G[s][0]
        gold_total = float(c["gold_sum"].sum() + d["gold_sum"].sum())
        parts = gold_total + float(c["c_pie"].sum() + c["risk_sum"].sum() + c["team_acct"].sum() + d["team_acct"].sum())
        summ.append({"side": s, "G0": G[s][0], "G_end": G[s][-1], "dG_total": total,
                     "gold_cluster": float(c["gold_sum"].sum()), "gold_drift": float(d["gold_sum"].sum()),
                     "c_pie": float(c["c_pie"].sum()), "d_pie": float(c["d_pie"].sum()), "risk": float(c["risk_sum"].sum()),
                     "team_acct_cluster": float(c["team_acct"].sum()), "time_term_cluster": float(c["time_term"].sum()),
                     "decay_term_cluster": float(c["decay_term"].sum()),
                     "team_acct_drift": float(d["team_acct"].sum()),
                     "identity_resid": parts - total,
                     "d_cross_resid": float(c["d_pie"].sum() + co["resid"].clip(upper_bound=0).sum()),
                     "c_pie_legacy": float(c["c_pie_legacy"].sum()), "d_pie_legacy": float(c["d_pie_legacy"].sum())})
    cluster_ms = sum(t[i_e] - t[i_s] for i_s, i_e in cl)
    lens = np.array([(e - s) / 1000 for s, e in spans])
    summary = pl.DataFrame(summ).with_columns(
        pl.lit(len(spans)).alias("n_clusters"), pl.lit(cluster_ms / (t[-1] - t[0])).alias("coverage"),
        pl.lit(float(lens.max()) if lens.size else 0.0).alias("max_len_s"),
        pl.lit(int((lens > ccfg["warn_len_sec"]).sum())).alias("n_long"),
        pl.lit(int((~deaths["respawn_observed"]).sum())).alias("n_respawn_unobserved"),
        pl.lit(int(deaths["k"].is_null().sum())).alias("n_deaths_outside"),
        pl.lit((t[-1] - t[0]) / 60_000).alias("game_min"))

    tables = {"clusters": clusters, "seeds": seeds, "deaths": deaths, "drift": drift,
              "gold_player": gold_player, "summary": summary}
    return {name: df.with_columns(pl.lit(gid).alias("game_id")) for name, df in tables.items()}


def build_one(meta: dict, cfg: dict, coefs: dict) -> dict:
    tables = build_timeline(meta, cfg, coefs)
    for name, df in tables.items():
        path = OUT / name / f"tournament={meta['tournament']}" / f"stage={meta['stage']}" / f"{meta['game_id']}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")
    return tables["summary"].to_dicts()[0]


def load_coefs() -> dict[str, dict[str, float]]:
    c = pl.read_parquet(MODEL_DIR / "coefficients.parquet")
    return {r["fold"]: r for r in c.to_dicts()}


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
        n = len(list(pool.imap_unordered(partial(build_one, cfg=cfg, coefs=load_coefs()), game_list(), chunksize=4)))
    print(f"타임라인 {n}게임, {time.time() - t0:.0f}초 → {OUT}")


if __name__ == "__main__":
    main()
