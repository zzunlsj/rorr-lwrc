"""실제 골드 비교 지표 — 게임 내 관측값(`stats_update` totalGold)만으로 만든 선수·팀 × scope 표 (Jun 요청 2026-11-01).

G 모델·클러스터·배분을 쓰지 않는다. LwRC+와 같은 게임(항등식 통과)·같은 scope·같은 포지션/팀 규칙으로 나란히 비교하기 위한 표다.

선수-게임:
  gold    = 마지막 틱의 totalGold (게임 내 표시값, 시작 골드 포함)
  gd_lane = gold_lane_minute(15분) 시점 totalGold − 같은 포지션 상대 선수 totalGold (게임이 그보다 짧으면 결측)
  team_gold = 마지막 틱의 팀 totalGold (stats_update.teams)
선수 × scope (합의 비율 — LwRC+와 같은 집계, 윈저라이즈 없음):
  gpm        = Σ gold / Σ 플레이 분
  gold_plus  = 100 × gpm / (같은 scope·포지션 Σ gold / Σ 분)
  gold_pg    = (gpm − 포지션 평균 gpm) × 분 / 게임 수
  gold_share = Σ gold / Σ 팀 골드 (본인이 뛴 게임)  [%]
  gd15       = 선수-게임 gd_lane 평균
비교: lwrc_ix(같은 scope LwRC+), rank_diff = Gold+ 포지션 순위 − LwRC+ 포지션 순위 (양수 = LwRC+에서 더 높은 순위).
  두 지표는 퍼짐이 달라(LwRC+ 표준편차 ≈ 10, Gold+ ≈ 3.7) 값의 차이 대신 순위 차이를 쓴다.
팀 × scope: gpm(팀) · gold_plus(= 100 × 팀 gpm / scope 팀 평균 gpm) · gold_pg(소속 선수 합, 게임당) · gd15 · gd_end(종료 시 팀 골드 차 평균)

결과: outputs/web/gold.json (→ report.build_site가 site/data/로 복사)
"""

import json
import time
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool

import polars as pl
import yaml

from common.config import data_snapshot
from common.paths import L1, OUTPUTS
from features.g_train import SPEC_VERSION
from ingest.leaguepedia import LWRC_YAML
from metrics.season import load_player_games, scope_rows

WEB = OUTPUTS / "web"


def game_gold(meta: tuple[str, str, str], lane_ms: int) -> pl.DataFrame:
    """한 게임의 선수별 종료 골드 · lane_ms 시점 골드 · 팀 종료 골드."""
    game_id, tournament, stage = meta
    path = f"tournament={tournament}/stage={stage}/{game_id}.parquet"
    fp = pl.read_parquet(L1 / "frames_player" / path, columns=["game_time_ms", "participant_id", "team_id", "total_gold"])
    ft = pl.read_parquet(L1 / "frames_team" / path, columns=["game_time_ms", "team_id", "total_gold"])
    fp, ft = fp.sort("game_time_ms"), ft.sort("game_time_ms")
    end = fp.group_by("participant_id").agg(pl.col("team_id").last(), pl.col("total_gold").last().alias("gold"))
    long_enough = fp["game_time_ms"].max() >= lane_ms
    at = (fp.filter(pl.col("game_time_ms") <= lane_ms).group_by("participant_id")
          .agg(pl.col("total_gold").last().alias("gold_lane")))
    team = ft.group_by("team_id").agg(pl.col("total_gold").last().alias("team_gold"))
    out = end.join(at, on="participant_id", how="left").join(team, on="team_id")
    if not long_enough:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("gold_lane"))
    return out.with_columns(pl.lit(game_id).alias("game_id"))


def player_games(p: pl.DataFrame, lane_ms: int, workers: int) -> pl.DataFrame:
    metas = p.select("game_id", "tournament", "stage").unique().rows()
    with Pool(workers) as pool:
        g = pl.concat(pool.map(partial(game_gold, lane_ms=lane_ms), metas, chunksize=8))
    x = p.select("game_id", "participant_id", "puuid", "display_name", "role", "team_code", "league", "tournament",
                 "play_min", "winning_team").join(g, on=["game_id", "participant_id"])
    opp = x.select("game_id", "role", "team_id", pl.col("gold_lane").alias("opp_lane"), pl.col("team_gold").alias("opp_team_gold"))
    x = x.join(opp, on=["game_id", "role"], suffix="_o").filter(pl.col("team_id") != pl.col("team_id_o")).drop("team_id_o")
    assert x.height == p.height, "라인 상대 매칭 실패"
    return x.with_columns((pl.col("gold_lane") - pl.col("opp_lane")).alias("gd_lane"))


def scope_players(x: pl.DataFrame) -> pl.DataFrame:
    main_role = (x.group_by("puuid", "role").len().sort("len", descending=True).unique("puuid", keep="first")
                 .select("puuid", pl.col("role").alias("main_role")))
    den = x.group_by("role").agg((pl.col("gold").sum() / pl.col("play_min").sum()).alias("den"))
    team_main = (x.group_by("puuid", "team_code", "league").len().sort("len", descending=True).unique("puuid", keep="first")
                 .select("puuid", pl.col("team_code").alias("team"), pl.col("league").alias("lg")))
    r = (x.join(main_role, on="puuid").group_by("puuid").agg(
            pl.col("main_role").first().alias("role"), pl.len().alias("g"), pl.col("play_min").sum().alias("minutes"),
            pl.col("gold").sum(), pl.col("team_gold").sum(), pl.col("gd_lane").mean().alias("gd15"),
            pl.col("gd_lane").count().alias("g15"))
         .join(den, on="role").join(team_main, on="puuid")
         .with_columns((pl.col("gold") / pl.col("minutes")).alias("gpm"))
         .with_columns((100 * pl.col("gpm") / pl.col("den")).alias("gold_plus"),
                       ((pl.col("gpm") - pl.col("den")) * pl.col("minutes") / pl.col("g")).alias("gold_pg"),
                       (100 * pl.col("gold") / pl.col("team_gold")).alias("gold_share")))
    return r


def scope_teams(x: pl.DataFrame, r: pl.DataFrame) -> pl.DataFrame:
    tg = (x.unique(["game_id", "team_code"]).group_by("team_code").agg(
            pl.len().alias("g"), pl.col("league").first().alias("lg"), pl.col("play_min").sum().alias("minutes"),
            pl.col("team_gold").sum(), (pl.col("team_gold") - pl.col("opp_team_gold")).mean().alias("gd_end"),
            (pl.col("team_id") == pl.col("winning_team")).mean().alias("wr")))
    lane = x.group_by("game_id", "team_code").agg(pl.col("gd_lane").sum()).group_by("team_code").agg(pl.col("gd_lane").mean().alias("gd15"))
    tmin = (x.group_by("puuid", "team_code").agg(pl.col("play_min").sum().alias("tmin"))
            .join(r.select("puuid", "gpm", "den"), on="puuid")
            .group_by("team_code").agg(((pl.col("gpm") - pl.col("den")) * pl.col("tmin")).sum().alias("_pg")))
    t = tg.join(lane, on="team_code").join(tmin, on="team_code").with_columns((pl.col("team_gold") / pl.col("minutes")).alias("gpm"))
    avg = t["team_gold"].sum() / t["minutes"].sum()
    return t.with_columns((100 * pl.col("gpm") / avg).alias("gold_plus"), (pl.col("_pg") / pl.col("g")).alias("gold_pg"))


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    sc = cfg["season"]
    t0 = time.time()
    p = load_player_games(cfg)
    x_all = player_games(p, sc["gold_lane_minute"] * 60_000, 6)
    lwrc = json.loads((WEB / "players.json").read_text())["players"]
    lwrc_t = json.loads((WEB / "teams.json").read_text())["teams"]
    ix = {(r["scope"], r["player_id"]): (r["ix"], r["role_rank"]) for r in lwrc}
    ix_t = {(r["scope"], r["team"]): r["ix"] for r in lwrc_t}
    players, teams = [], []
    for scope in sc["output_scopes"]:
        x = scope_rows(x_all, scope, cfg)
        r = scope_players(x)
        ranked = (r.filter(pl.col("g") >= sc["min_games_for_ranking"])
                  .with_columns(pl.col("gold_plus").rank("ordinal", descending=True).over("role").cast(pl.Int64).alias("role_rank"),
                                pl.len().over("role").cast(pl.Int64).alias("role_n")).select("puuid", "role_rank", "role_n"))
        r = r.join(ranked, on="puuid", how="left").join(
            p.sort("game_start_utc").group_by("puuid").agg(pl.col("display_name").last().alias("name")), on="puuid")
        for d in r.sort("role", "gold_plus", descending=[False, True]).iter_rows(named=True):
            lx, lrank = ix[(scope, d["puuid"])]
            players.append({"scope": scope, "player_id": d["puuid"], "name": d["name"], "team": d["team"], "lg": d["lg"],
                            "role": d["role"], "g": d["g"], "minutes": round(d["minutes"], 2), "gold": round(d["gold"]),
                            "gpm": round(d["gpm"], 1), "gold_plus": round(d["gold_plus"]), "gold_pg": round(d["gold_pg"]),
                            "gold_share": round(d["gold_share"], 1),
                            "gd15": None if d["gd15"] is None else round(d["gd15"]), "g15": d["g15"],
                            "role_rank": d["role_rank"], "role_n": d["role_n"],
                            "lwrc_ix": lx, "lwrc_rank": lrank,
                            "rank_diff": None if d["role_rank"] is None else d["role_rank"] - lrank})
        t = scope_teams(x, r)
        lrank_t = {tm: i + 1 for i, tm in enumerate(sorted((k[1] for k in ix_t if k[0] == scope), key=lambda tm: -ix_t[(scope, tm)]))}
        for i, d in enumerate(t.sort("gold_plus", descending=True).iter_rows(named=True)):
            lx = ix_t[(scope, d["team_code"])]
            teams.append({"scope": scope, "team": d["team_code"], "lg": d["lg"], "g": d["g"], "gpm": round(d["gpm"], 1),
                          "gold_plus": round(d["gold_plus"]), "gold_pg": round(d["gold_pg"]), "gd15": round(d["gd15"]),
                          "gd_end": round(d["gd_end"]), "wr": round(d["wr"], 4),
                          "lwrc_ix": lx, "rank": i + 1, "lwrc_rank": lrank_t[d["team_code"]],
                          "rank_diff": i + 1 - lrank_t[d["team_code"]]})
    body = {"version": SPEC_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_snapshot": data_snapshot(), "source": "stats_update totalGold (게임 내 관측값만)",
            "gold_lane_minute": sc["gold_lane_minute"], "players": players, "teams": teams}
    (WEB / "gold.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
    print(f"gold.json: 선수 {len(players)}행, 팀 {len(teams)}행, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
