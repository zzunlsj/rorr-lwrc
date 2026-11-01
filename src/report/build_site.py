"""결과 조회 페이지용 데이터 복사 (contracts/output-web.md — 웹은 표시만 한다).

outputs/web/{players,teams}.json(L4, 스키마 검증 통과본)을 site/data/로 그대로 복사하고, 페이지 안내용 meta.json을 만든다.
site/는 GitHub Pages로 배포된다 (.github/workflows/pages.yml). Raw·L1·L2·L3는 넣지 않는다 (CLAUDE.md §10).

사용: PYTHONPATH=src uv run python -m report.build_site
"""

import json
import shutil

import polars as pl
import yaml

from common.paths import L3, OUTPUTS, ROOT
from ingest.leaguepedia import LWRC_YAML

SITE_DATA = ROOT / "site" / "data"


def main() -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    for name in ("players.json", "teams.json", "gold.json"):
        shutil.copyfile(OUTPUTS / "web" / name, SITE_DATA / name)
    games = (pl.scan_parquet(L3 / "game_player_lwrc" / "**" / "*.parquet", hive_partitioning=False)
             .select("tournament", "game_id").unique().group_by("tournament").agg(pl.len().alias("games"))
             .collect())
    inc = json.loads((OUTPUTS / "web" / "teams.json").read_text())["incomplete_matches"]["all"]
    inc_txt = ", ".join(f"{' vs '.join(m['teams'])} {m['first_game_utc'][5:10].replace('-', '/')} (raw {m['raw_games']}/{m['played']}게임)"
                        for m in inc)
    meta = {"min_games_for_ranking": cfg["season"]["min_games_for_ranking"],
            "games_by_tournament": dict(games.sort("tournament").iter_rows()),
            "notes": ["Worlds는 진행 중 — 수령분까지만 반영 (11/15 이후 갱신 예정)",
                      "경기·게임 전적은 raw 기준: 게임 전적 = raw 게임 승패, 경기 전적·최근 5경기 = raw에 게임이 모두 있는 경기만"]
                     + ([f"raw에 일부 게임만 있어 경기 전적·최근 5경기에서 뺀 경기: {inc_txt}"] if inc else [])}
    (SITE_DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"site/data 갱신: {sum(meta['games_by_tournament'].values())}게임")


if __name__ == "__main__":
    main()
