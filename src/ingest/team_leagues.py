"""팀 → 2026 지역 리그 (웹 `lg`, contracts/output-web.md §1).

Leaguepedia: 팀이 2026년에 참가한 대회의 League 중 config `season.league_codes`에 있는 지역 리그.
Kazel 팀 ID ↔ Leaguepedia 팀 이름은 같은 게임(platform_game_id)의 블루·레드로 연결한다.
지역 리그가 0개이거나 2개 이상이면 멈춘다.
결과: manifest/team_leagues_2026.csv (esports_team_id, team_code, lp_team(개명 시 ' / '로 연결), league)
"""

import polars as pl
import yaml

from common.paths import L1, MANIFEST
from ingest.leaguepedia import LWRC_YAML, cargo_all

OUT = MANIFEST / "team_leagues_2026.csv"


def kazel_to_lp() -> pl.DataFrame:
    lp = pl.read_csv(MANIFEST / "leaguepedia_games_2026.csv").select("platform_game_id", "blue", "red")
    g = pl.read_parquet(L1 / "games" / "**" / "*.parquet").filter(~pl.col("excluded")).select(
        "platform_game_id", "blue_team_id", "blue_team_code", "red_team_id", "red_team_code").join(lp, on="platform_game_id")
    pairs = pl.concat([
        g.select(pl.col("blue_team_id").alias("esports_team_id"), pl.col("blue_team_code").alias("team_code"), pl.col("blue").alias("lp_team")),
        g.select(pl.col("red_team_id").alias("esports_team_id"), pl.col("red_team_code").alias("team_code"), pl.col("red").alias("lp_team")),
    ]).unique()
    return pairs                     # 시즌 중 개명한 팀은 이름이 여럿 (예: BRION → HANJIN BRION)


def fetch() -> pl.DataFrame:
    codes = yaml.safe_load(LWRC_YAML.read_text())["season"]["league_codes"]
    pairs = kazel_to_lp()
    names = ",".join("'" + t.replace("'", "\\'") + "'" for t in pairs["lp_team"].to_list())
    rows = cargo_all({"tables": "TournamentRosters=TR,Tournaments=T", "join_on": "TR.OverviewPage=T.OverviewPage",
                      "fields": "TR.Team=Team,T.League=League", "where": f"T.Year=2026 AND TR.Team IN ({names})"})
    lg = (pl.DataFrame(rows).filter(pl.col("League").is_in(list(codes))).unique()
          .with_columns(pl.col("League").replace_strict(codes).alias("league")).select(pl.col("Team").alias("lp_team"), "league"))
    out = (pairs.join(lg, on="lp_team", how="left").group_by("esports_team_id", "team_code")
           .agg(pl.col("lp_team").unique().sort().str.join(" / "), pl.col("league").drop_nulls().unique().alias("leagues")))
    bad = out.filter(pl.col("leagues").list.len() != 1)
    assert bad.is_empty(), f"지역 리그가 0개 또는 2개 이상인 팀: {bad}"
    out = out.with_columns(pl.col("leagues").list.first().alias("league")).drop("leagues").sort("league", "team_code")
    out.write_csv(OUT)
    return out


if __name__ == "__main__":
    print(fetch())
