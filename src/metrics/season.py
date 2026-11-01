"""Step 6 — scope별 집계 · 포지션 분모 · LwRC+ · 웹 출력 (spec/05 §5.3–5.5, contracts/output-web.md, 검증 spec/07 #9–#11).

rate_i(scope)     = Σ LwRC_i / Σ 플레이 분_i                       (합의 비율, 게임별 LwRC+ 평균 아님)
denom(scope, pos) = Σ_(scope·pos 전원) LwRC / Σ 플레이 분
LwRC+_i           = 100 × rate_i / denom(scope, pos_i),  pg_i = (rate_i − denom) × 분_i / 게임 수_i

해석 (사양 문구가 열린 곳 — 리포트에 적는다):
- 윈저라이즈 ±σ (`season.winsorize_sigma`): scope·포지션 안에서 선수-게임 분당 LwRC를 평균 ± σ·표준편차로 자른 뒤 rate·분모를 계산.
  출력 `lwrc`·`c`·`d`·`risk`는 자르지 않은 합.
- 분모 prior 수축: 적용하지 않는다. scope·포지션별 분모 표본이 최소 90 선수-게임(First Stand)이라 "표본 부족 구간"이 없다.
- 선수 포지션 = 그 scope에서 가장 많이 뛴 포지션, 팀 = 가장 많이 뛴 팀. LCK scope는 LCK 팀 소속 선수-게임만.
- 시리즈 결과(W·L·sw·sl·form)는 Leaguepedia 공식 결과 (raw 누락 게임 포함). `g`는 LwRC가 있는 게임 수.

가중 방식: #9 split-half 신뢰도가 높은 쪽(회귀 main / prior_w)을 웹 출력 정본으로 쓴다 (spec/04 §4.5).
결과: data/l3/season_player_lwrc_plus.parquet, outputs/web/{players,teams}.json, outputs/validation_report.json
"""

import json
import time
from datetime import datetime, timezone

import jsonschema
import numpy as np
import polars as pl
import yaml

from common.config import data_snapshot
from common.paths import L1, L2, L3, MANIFEST, OUTPUTS, ROOT
from features.alloc_features import OUT as FEAT_DIR
from features.g_train import SPEC_VERSION
from ingest.leaguepedia import LWRC_YAML

WEB = OUTPUTS / "web"
VARIANT_COL = {"main": "lwrc", "prior_w": "lwrc_prior_w", "legacy": "lwrc_legacy", "prior_g": "lwrc_prior_g"}
METHOD_OF = {"main": "regression", "prior_w": "prior"}


def load_player_games(cfg: dict) -> pl.DataFrame:
    p = pl.read_parquet(L3 / "game_player_lwrc" / "**" / "*.parquet", hive_partitioning=False).filter(pl.col("identity_pass"))
    g = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("game_id", "game_start_utc", "winning_team")
    lg = pl.read_csv(MANIFEST / "team_leagues_2026.csv", schema_overrides={"esports_team_id": pl.Utf8}).select("esports_team_id", "league")
    p = p.join(g, on="game_id").join(lg, on="esports_team_id", how="left")
    assert p["league"].null_count() == 0, "리그 미매핑 팀"
    return p.with_columns(pl.col("role").replace_strict(cfg["season"]["role_codes"]).alias("role"))


def scope_rows(p: pl.DataFrame, scope: str, cfg: dict) -> pl.DataFrame:
    st = cfg["season"]["scope_tournaments"]
    if scope == "all":
        return p
    x = p.filter(pl.col("tournament").is_in(st[scope]))
    return x.filter(pl.col("league") == "LCK") if scope.startswith("lck") else x


def winsorized(x: pl.DataFrame, col: str, sigma: float) -> pl.Expr:
    r = pl.col(col) / pl.col("play_min")
    mu, sd = r.mean().over("role"), r.std().over("role")
    return r.clip(mu - sigma * sd, mu + sigma * sd) * pl.col("play_min")


def lwrc_plus(x: pl.DataFrame, col: str, sigma: float) -> pl.DataFrame:
    """선수 × (scope 내) LwRC+. x는 한 scope의 선수-게임."""
    x = x.with_columns(winsorized(x, col, sigma).alias("_w"))
    den = x.group_by("role").agg((pl.col("_w").sum() / pl.col("play_min").sum()).alias("denom"))
    main_role = x.group_by("puuid", "role").len().sort("len", descending=True).unique("puuid", keep="first").select("puuid", pl.col("role").alias("main_role"))
    r = (x.join(main_role, on="puuid").group_by("puuid").agg(
            pl.col("main_role").first().alias("role"), pl.len().alias("g"), pl.col("play_min").sum().alias("minutes"),
            pl.col(col).sum().alias("lwrc"), pl.col("_w").sum().alias("_wsum")))
    return (r.join(den, on="role").with_columns((pl.col("_wsum") / pl.col("minutes")).alias("rate"))
            .with_columns((100 * pl.col("rate") / pl.col("denom")).alias("ix_f"),
                          ((pl.col("rate") - pl.col("denom")) * pl.col("minutes") / pl.col("g")).alias("pg_f")).drop("_wsum"))


def split_half(p: pl.DataFrame, col: str, sigma: float, min_half: int, mode: str, rng=None) -> tuple[float, int]:
    p = p.sort("game_start_utc").with_columns(pl.int_range(pl.len()).over("puuid").alias("_i"), pl.len().over("puuid").alias("_n"))
    a, b = ((p.filter(pl.col("_i") < pl.col("_n") // 2), p.filter(pl.col("_i") >= pl.col("_n") // 2)) if mode == "half"
            else (p.filter(pl.col("_i") % 2 == 0), p.filter(pl.col("_i") % 2 == 1)))
    j = lwrc_plus(a, col, sigma).join(lwrc_plus(b, col, sigma), on="puuid", suffix="_b").filter(
        (pl.col("g") >= min_half) & (pl.col("g_b") >= min_half))
    return float(np.corrcoef(j["ix_f"], j["ix_f_b"])[0, 1]), j.height


def split_half_diff_ci(p: pl.DataFrame, sigma: float, min_half: int, reps: int, seed: int) -> dict:
    """회귀 − prior 신뢰도 차이의 선수 재표집 부트스트랩 (시간 전반/후반)."""
    p = p.sort("game_start_utc").with_columns(pl.int_range(pl.len()).over("puuid").alias("_i"), pl.len().over("puuid").alias("_n"))
    a, b = p.filter(pl.col("_i") < pl.col("_n") // 2), p.filter(pl.col("_i") >= pl.col("_n") // 2)
    cols = {}
    for v in ("main", "prior_w"):
        c = VARIANT_COL[v]
        j = lwrc_plus(a, c, sigma).join(lwrc_plus(b, c, sigma), on="puuid", suffix="_b").filter(
            (pl.col("g") >= min_half) & (pl.col("g_b") >= min_half)).sort("puuid")
        cols[v] = (j["ix_f"].to_numpy(), j["ix_f_b"].to_numpy())
    rng = np.random.default_rng(seed)
    n = cols["main"][0].size
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        diffs.append(np.corrcoef(cols["main"][0][i], cols["main"][1][i])[0, 1] - np.corrcoef(cols["prior_w"][0][i], cols["prior_w"][1][i])[0, 1])
    return {"diff": float(np.mean(diffs)), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
            "p_regression_higher": float((np.array(diffs) > 0).mean()), "n_players": int(n)}


def bench_direction(p: pl.DataFrame, min_games: int) -> dict:
    """#10: 포지션 안에서 (실현 골드 순위 − LwRC 순위) 상승폭과 교전 존재 비율의 상관. 양수면 로밍형이 오른다."""
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=False).select("game_id", "participant_id", "presence")
    pres = p.select("game_id", "participant_id", "puuid").join(feat, on=["game_id", "participant_id"]).group_by("puuid").agg(
        pl.col("presence").mean())
    s = (p.group_by("puuid").agg(pl.col("role").mode().first(), pl.len().alias("g"),
                                 (pl.col("lwrc").sum() / pl.col("play_min").sum()).alias("lwrc_r"),
                                 (pl.col("c_gold").sum() / pl.col("play_min").sum()).alias("gold_r"))
         .filter(pl.col("g") >= min_games).join(pres, on="puuid")
         .with_columns(pl.col("lwrc_r").rank(descending=True).over("role").alias("rk_l"),
                       pl.col("gold_r").rank(descending=True).over("role").alias("rk_g"))
         .with_columns((pl.col("rk_g") - pl.col("rk_l")).alias("rank_gain")))
    out = {}
    for role, d in sorted(s.partition_by("role", as_dict=True).items()):
        out[role[0]] = {"n": d.height, "corr_presence_rank_gain": float(np.corrcoef(d["presence"], d["rank_gain"])[0, 1]),
                        "corr_lwrc_gold_rate": float(np.corrcoef(d["lwrc_r"], d["gold_r"])[0, 1])}
    return out


def series_results(scope_games: pl.DataFrame) -> pl.DataFrame:
    """Leaguepedia 공식 결과로 팀별 시리즈·게임 결과. scope_games: game_id 목록 (우리 데이터)."""
    lp = pl.read_csv(MANIFEST / "leaguepedia_games_2026.csv")
    tl = pl.read_csv(MANIFEST / "team_leagues_2026.csv", schema_overrides={"esports_team_id": pl.Utf8})
    name_to_code = {n: c for c, names in tl.select("team_code", "lp_team").iter_rows() for n in names.split(" / ")}
    g = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("game_id", "platform_game_id")
    matches = scope_games.join(g, on="game_id").join(lp, on="platform_game_id")["lp_match_id"].unique()
    games = lp.filter(pl.col("lp_match_id").is_in(matches.implode()) & pl.col("winner").is_not_null())
    # MatchScheduleGame.Winner: 1 = 블루, 2 = 레드 (시리즈 team1/team2가 아님 — normalize.build_l1과 같은 해석)
    games = games.with_columns(pl.when(pl.col("winner") == 1).then(pl.col("blue")).otherwise(pl.col("red")).alias("w"),
                               pl.when(pl.col("winner") == 1).then(pl.col("red")).otherwise(pl.col("blue")).alias("l"))
    rows = []
    for (mid,), m in games.partition_by(["lp_match_id"], as_dict=True).items():
        wins = m.group_by("w").len()
        top = wins.sort("len", descending=True)
        t1, t2 = m["team1"][0], m["team2"][0]
        winner = top["w"][0]
        when = m["datetime_utc"].min()
        for team in (t1, t2):
            rows.append({"team": name_to_code[team], "match": mid, "when": when, "won": team == winner,
                         "sw": int((m["w"] == team).sum()), "sl": int((m["l"] == team).sum())})
    r = pl.DataFrame(rows).sort("when")
    return r.group_by("team").agg(pl.col("won").sum().alias("W"), (~pl.col("won")).sum().alias("L"),
                                  pl.col("sw").sum(), pl.col("sl").sum(),
                                  pl.when(pl.col("won")).then(pl.lit("W")).otherwise(pl.lit("L")).tail(5).str.join("").alias("form"))


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    sc = cfg["season"]
    sigma, min_rank = sc["winsorize_sigma"], sc["min_games_for_ranking"]
    t0 = time.time()
    p = load_player_games(cfg)

    # 검증 #9 (정본 가중 선택), #10, #11
    rel = {mode: {v: dict(zip(("r", "n"), split_half(p, c, sigma, min_rank, mode))) for v, c in
                  {**VARIANT_COL, "gold_only": "c_gold"}.items()} for mode in ("half", "oddeven")}
    for mode in rel:
        for v in rel[mode]:
            r = rel[mode][v]["r"]
            rel[mode][v]["spearman_brown"] = 2 * r / (1 + r)
    diff = split_half_diff_ci(p, sigma, min_rank, cfg["allocation"]["bootstrap_reps"], 0)
    chosen = max(METHOD_OF, key=lambda v: rel["half"][v]["r"])
    length = {v: float(np.corrcoef(p["play_min"], p[c] / p["play_min"])[0, 1]) for v, c in VARIANT_COL.items()}
    stratify = any(abs(x) > sc["length_corr_threshold"] for x in length.values())
    assert not stratify, "게임 길이 상관 임계 초과 — 분모 층화 필요 (spec/05 §5.5), 구현 전 보고"

    # scope별 LwRC+ (모든 변형 — parquet, 정본만 웹)
    players_all, players_web, teams_web = [], [], []
    who = (p.sort("game_start_utc").group_by("puuid").agg(pl.col("display_name").last()))
    for scope in sc["output_scopes"]:
        x = scope_rows(p, scope, cfg)
        team_main = (x.group_by("puuid", "team_code", "league").len().sort("len", descending=True)
                     .unique("puuid", keep="first").select("puuid", pl.col("team_code").alias("team"), pl.col("league").alias("lg")))
        for v, col in VARIANT_COL.items():
            r = lwrc_plus(x, col, sigma).with_columns(pl.lit(scope).alias("scope"), pl.lit(v).alias("variant"))
            players_all.append(r)
        r = lwrc_plus(x, VARIANT_COL[chosen], sigma)
        sfx = "" if chosen == "main" else "_prior_w"
        ch = x.group_by("puuid").agg(pl.col(f"c{sfx}").sum().alias("c"), pl.col(f"d{sfx}").sum().alias("d"),
                                     pl.col(f"risk{sfx}").sum().alias("risk"), pl.col("c_gold").sum(), pl.col("c_pie").sum())
        r = (r.join(ch, on="puuid").join(team_main, on="puuid").join(who, on="puuid")
             .with_columns(pl.col("ix_f").round(0).cast(pl.Int64).alias("ix"), pl.col("pg_f").round(0).cast(pl.Int64).alias("pg")))
        ranked = r.filter(pl.col("g") >= min_rank).with_columns(
            pl.col("ix_f").rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("role_rank"),
            pl.len().over("role").cast(pl.Int64).alias("role_n")).select("puuid", "role_rank", "role_n")
        r = r.join(ranked, on="puuid", how="left").with_columns(
            (100 * pl.col("lwrc").clip(lower_bound=0) / pl.col("lwrc").clip(lower_bound=0).sum().over("team")).alias("team_share"))
        players_web.append(r.with_columns(pl.lit(scope).alias("scope")))

        # 팀: 선수 ix의 플레이 분 가중 평균 (팀 소속 분 기준), pg = Σ (rate − denom) × 팀 소속 분 / 팀 게임 수
        tm = (x.group_by("puuid", "team_code").agg(pl.col("play_min").sum().alias("tmin"))
              .join(r.select("puuid", "role", "ix_f", "rate", "denom"), on="puuid"))
        tg = x.group_by("team_code").agg(pl.col("game_id").n_unique().alias("g"), pl.col("league").first().alias("lg"))
        team = tm.group_by("team_code").agg(((pl.col("ix_f") * pl.col("tmin")).sum() / pl.col("tmin").sum()).alias("ix_f"),
                                            ((pl.col("rate") - pl.col("denom")) * pl.col("tmin")).sum().alias("_pg"))
        top = (tm.group_by("team_code", "role").agg(((pl.col("ix_f") * pl.col("tmin")).sum() / pl.col("tmin").sum()).alias("rix"))
               .sort("rix", descending=True).unique("team_code", keep="first").select("team_code", pl.col("role").alias("top_role")))
        res = series_results(x.select("game_id").unique())
        team = (team.join(tg, on="team_code").join(top, on="team_code").join(res, left_on="team_code", right_on="team", how="left")
                .with_columns(pl.col("ix_f").round(0).cast(pl.Int64).alias("ix"), (pl.col("_pg") / pl.col("g")).round(0).cast(pl.Int64).alias("pg"),
                              (pl.col("sw") - pl.col("sl")).alias("sd"), (pl.col("sw") / (pl.col("sw") + pl.col("sl"))).alias("wr"),
                              pl.lit(scope).alias("scope")))
        assert team["W"].null_count() == 0, f"{scope}: 시리즈 결과 없는 팀"
        teams_web.append(team)

    season = pl.concat(players_all).with_columns(pl.lit(SPEC_VERSION).alias("spec_version"), pl.lit(data_snapshot()).alias("data_snapshot"))
    season.write_parquet(L3 / "season_player_lwrc_plus.parquet")

    meta = {"version": SPEC_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "allocation_method": METHOD_OF[chosen], "g_model_version": f"{SPEC_VERSION}+ridge-crossfit5+snapshot-{data_snapshot()}"}
    pw = pl.concat(players_web).sort("scope", "role", "ix_f", descending=[False, False, True])
    players = [{"scope": r["scope"], "player_id": r["puuid"], "name": r["display_name"], "team": r["team"], "lg": r["lg"],
                "role": r["role"], "g": r["g"], "minutes": round(r["minutes"], 2), "ix": r["ix"], "pg": r["pg"],
                "lwrc": round(r["lwrc"], 1), "rate": round(r["rate"], 3), "c": round(r["c"], 1), "d": round(r["d"], 1),
                "risk": round(r["risk"], 1), "role_rank": r["role_rank"], "role_n": r["role_n"],
                "team_share": None if r["team_share"] is None else round(r["team_share"], 1),
                "c_gold": round(r["c_gold"], 1), "c_pie": round(r["c_pie"], 1)} for r in pw.iter_rows(named=True)]
    tw = pl.concat(teams_web).sort("scope", "ix_f", descending=[False, True])
    teams = [{"scope": r["scope"], "team": r["team_code"], "lg": r["lg"], "W": r["W"], "L": r["L"], "sw": r["sw"], "sl": r["sl"],
              "sd": r["sd"], "wr": round(r["wr"], 4), "form": r["form"], "g": r["g"], "ix": r["ix"], "pg": r["pg"],
              "top_role": r["top_role"]} for r in tw.iter_rows(named=True)]
    WEB.mkdir(parents=True, exist_ok=True)
    for name, body in (("players", {**meta, "players": players}), ("teams", {**meta, "teams": teams})):
        jsonschema.validate(body, json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text()))
        (WEB / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))

    report = {**meta, "spec_version": SPEC_VERSION, "data_snapshot": data_snapshot(),
              "games": int(p["game_id"].n_unique()), "player_games": int(p.height),
              "split_half": rel, "split_half_diff_regression_minus_prior": diff, "chosen_variant": chosen,
              "length_corr": length, "length_stratified": stratify, "bench_direction": bench_direction(p, min_rank),
              "winsorize_sigma": sigma, "denominator_shrinkage": "not applied (min 90 player-games per scope×role)"}
    (OUTPUTS / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"정본 가중 = {METHOD_OF[chosen]} (split-half {rel['half'][chosen]['r']:.3f}), 선수 행 {len(players)}, 팀 행 {len(teams)}, "
          f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
