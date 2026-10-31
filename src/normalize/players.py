"""선수·팀 ID 매핑 (spec/06 §1 "대회 간 동일 선수 ID 매핑 테이블 필수").

- 선수 키 = puuid. Step 0에서 대회 간 안정적임을 확인 (이름이 바뀐 10명 모두 같은 puuid).
- 팀 키 = Kazel esports_team_id. 팀명 변경(DRX → KIWOOM DRX 등)에도 같은 ID.
- 표시 이름은 raw playerName("KT Bdd")에서 팀 접두어를 뗀 것. 이적하면 접두어만 바뀐다.
- 팀 리그: LCK 리그 게임(Kazel league='LCK')에 한 번이라도 나온 팀 = LCK. 나머지는 'other'
  (해외 팀의 소속 리그명은 출력 단계에서 필요하면 별도 출처로 채운다).

결과: data/l1/players.parquet, data/l1/teams.parquet
"""

import polars as pl

from common.paths import L1


def strip_team_prefix(player_name: str | None) -> str | None:
    if player_name is None:
        return None
    parts = player_name.strip().split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0]


def build_players() -> None:
    games = (pl.scan_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True)
             .select("game_id", "tournament", "game_start_utc", "league_label", "excluded").collect())
    parts = (pl.scan_parquet(L1 / "participants" / "**" / "*.parquet", hive_partitioning=False)
             .select("game_id", "puuid", "player_name", "display_name", "role", "esports_team_id", "team_code", "team_name")
             .collect()
             .join(games, on="game_id"))

    teams = (parts.group_by("esports_team_id").agg(
                 pl.col("team_code").sort_by("game_start_utc").last().alias("team_code"),
                 pl.col("team_name").sort_by("game_start_utc").last().alias("team_name"),
                 pl.col("team_name").unique().sort().alias("names_seen"),
                 (pl.col("league_label") == "LCK").any().alias("is_lck"),
                 pl.col("game_id").n_unique().alias("n_games"))
             .with_columns(pl.when(pl.col("is_lck")).then(pl.lit("LCK")).otherwise(pl.lit("other")).alias("league"))
             .sort("league", "team_code"))

    team_league = dict(zip(teams["esports_team_id"], teams["league"]))
    players = (parts.with_columns(pl.col("esports_team_id").replace_strict(team_league, default=None).alias("team_league"))
               .group_by("puuid").agg(
                   pl.col("display_name").sort_by("game_start_utc").last().alias("display_name"),
                   pl.col("player_name").unique().sort().alias("names_seen"),
                   pl.col("role").mode().first().alias("role"),
                   pl.col("role").n_unique().alias("n_roles"),
                   pl.col("team_code").sort_by("game_start_utc").last().alias("last_team_code"),
                   pl.col("esports_team_id").unique().alias("team_ids"),
                   pl.col("team_code").unique().sort().alias("team_codes"),
                   (pl.col("team_league") == "LCK").sum().alias("games_for_lck_team"),
                   pl.len().alias("n_games"),
                   pl.col("game_start_utc").min().alias("first_game_utc"),
                   pl.col("game_start_utc").max().alias("last_game_utc"))
               .sort("role", "display_name"))

    teams.write_parquet(L1 / "teams.parquet")
    players.write_parquet(L1 / "players.parquet")
