"""Leaguepedia(lol.fandom.com) Cargo 조회 — 경기 → 대회·스테이지 매핑 출처 (결정 18).

- 조인 키: Leaguepedia `MatchScheduleGame.RiotPlatformGameId` = raw `platform_game_id` (예: LOLTMNT01_439669).
- 우리 쪽 ID를 외부로 보내지 않는다. 대회 페이지 단위로 전체를 받아 로컬에서 조인한다.
- 조회는 `Special:CargoExport`(api.php보다 한도가 느슨함). 429·503이면 지수 대기 후 재시도한다.

사용:
  uv run python -m ingest.leaguepedia discover          # 2026 대상 대회 OverviewPage·Tab 목록
  uv run python -m ingest.leaguepedia fetch             # manifest/leaguepedia_{games,groups}_2026.csv 갱신
  uv run python -m ingest.leaguepedia build-map <pairs.csv>   # match_id·platform_game_id 열이 있는 CSV → manifest/match_stage_map.csv
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from common.paths import MANIFEST, ROOT

EXPORT = "https://lol.fandom.com/wiki/Special:CargoExport"
USER_AGENT = "rorr-lwrc/0.1 (LwRC+ research; contact jun@piepie.co)"
PAGE = 500
PAUSE_SEC = 3.0
GAMES_CSV = MANIFEST / "leaguepedia_games_2026.csv"
GROUPS_CSV = MANIFEST / "leaguepedia_groups_2026.csv"
STAGE_MAP_CSV = MANIFEST / "match_stage_map.csv"
LWRC_YAML = ROOT / "config" / "lwrc.yaml"

# 2026 대상 대회 (CLAUDE.md §7). LIKE 패턴은 discover 결과로 확인한다.
OVERVIEW_WHERE = (
    'MS.DateTime_UTC >= "2026-01-01" AND ('
    'MS.OverviewPage LIKE "LCK/2026%" OR MS.OverviewPage LIKE "%2026%First Stand%" '
    'OR MS.OverviewPage LIKE "2026 Mid-Season Invitational%" '
    'OR MS.OverviewPage LIKE "2026 Season World Championship%")'
)

GAME_FIELDS = [
    ("MSG.RiotPlatformGameId", "platform_game_id"),
    ("MSG.GameId", "lp_game_id"),
    ("MSG.MatchId", "lp_match_id"),
    ("MSG.N_GameInMatch", "game_n"),
    ("MSG.Blue", "blue"),
    ("MSG.Red", "red"),
    ("MSG.Winner", "winner"),
    ("MS.OverviewPage", "overview_page"),
    ("MS.Tab", "tab"),
    ("MS.DateTime_UTC", "datetime_utc"),
    ("MS.Team1", "team1"),
    ("MS.Team2", "team2"),
    ("MS.BestOf", "best_of"),
]


def cargo(params: dict, max_tries: int = 6) -> list[dict]:
    # api.php(cargoquery)는 익명 요청 한도가 빡빡해 막히기 쉽다 → Special:CargoExport로 같은 쿼리를 보낸다
    url = EXPORT + "?" + urllib.parse.urlencode({"format": "json", **params})
    wait = 60.0
    for attempt in range(max_tries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            print(f"HTTP {e.code} — {wait:.0f}초 대기 ({attempt + 1}/{max_tries})", file=sys.stderr, flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 900)
            continue
        if not isinstance(body, list):
            raise RuntimeError(f"CargoExport 오류: {str(body)[:300]}")
        time.sleep(PAUSE_SEC)
        return body
    raise RuntimeError("요청 한도 재시도 초과")


def cargo_all(params: dict) -> list[dict]:
    rows, offset = [], 0
    while True:
        page = cargo({**params, "limit": PAGE, "offset": offset})
        rows += page
        if len(page) < PAGE:
            return rows
        offset += PAGE


def discover() -> None:
    rows = cargo_all({
        "tables": "MatchSchedule=MS",
        "fields": "MS.OverviewPage=overview_page,MS.Tab=tab,COUNT(*)=matches",
        "where": OVERVIEW_WHERE,
        "group_by": "MS.OverviewPage,MS.Tab",
    })
    for r in rows:
        print(f"{r['overview_page']}\t{r['tab']}\t{r['matches']}")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        # Cargo는 별칭의 밑줄을 공백으로 돌려준다
        for r in rows:
            w.writerow({c: r.get(c.replace("_", " "), r.get(c, "")) for c in columns})


def fetch() -> None:
    rows = cargo_all({
        "tables": "MatchScheduleGame=MSG,MatchSchedule=MS",
        "join_on": "MSG.MatchId=MS.MatchId",
        "fields": ",".join(f"{src}={dst}" for src, dst in GAME_FIELDS),
        "where": OVERVIEW_WHERE,
        "order_by": "MS.DateTime_UTC,MSG.N_GameInMatch",
    })
    write_csv(GAMES_CSV, [dst for _, dst in GAME_FIELDS], rows)
    print(f"{len(rows)}게임 → {GAMES_CSV}")
    groups = cargo_all({
        "tables": "TournamentGroups=TG",
        "fields": "TG.OverviewPage=overview_page,TG.GroupName=group_name,TG.Team=team",
        "where": 'TG.OverviewPage LIKE "LCK/2026%" OR TG.OverviewPage LIKE "%2026%First Stand%" '
                 'OR TG.OverviewPage LIKE "2026 Mid-Season Invitational%" '
                 'OR TG.OverviewPage LIKE "2026 Season World Championship%"',
    })
    write_csv(GROUPS_CSV, ["overview_page", "group_name", "team"], groups)
    print(f"{len(groups)}행 → {GROUPS_CSV}")


def classify(game: dict, rules: list[dict], groups: dict, group_stage: dict) -> tuple[str, str]:
    """Leaguepedia 게임 1건 → (tournament, stage). 규칙에 없으면 오류로 멈춘다."""
    for rule in rules:
        if game["overview_page"] != rule["overview"]:
            continue
        if "tab_prefix" in rule and not game["tab"].startswith(rule["tab_prefix"]):
            continue
        stage = rule["stage"]
        if stage == "by_group":
            g = {groups.get((game["overview_page"], t)) for t in (game["team1"], game["team2"])}
            if len(g) != 1 or None in g:
                raise ValueError(f"그룹 판정 불가 {game['lp_match_id']}: {game['team1']}/{game['team2']} → {g}")
            stage = group_stage[g.pop()]
        return rule["tournament"], stage
    raise ValueError(f"stage_map 규칙에 없는 페이지: {game['overview_page']} / {game['tab']}")


def build_map(pairs_csv: Path) -> None:
    cfg = yaml.safe_load(LWRC_YAML.read_text())["stage_map"]
    csv.field_size_limit(sys.maxsize)
    with GROUPS_CSV.open(newline="") as fh:
        groups = {(r["overview_page"], r["team"]): r["group_name"] for r in csv.DictReader(fh)}
    with GAMES_CSV.open(newline="") as fh:
        lp = {r["platform_game_id"]: r for r in csv.DictReader(fh)
              if r["platform_game_id"]
              and not any(x in r["overview_page"] for x in cfg["exclude_overview_contains"])}

    by_match: dict[str, set] = {}
    with pairs_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            by_match.setdefault(r["match_id"], set()).add(r["platform_game_id"])

    out, missing, conflict = [], [], []
    for match_id, pgids in sorted(by_match.items()):
        found = {}
        for pgid in sorted(pgids):
            if pgid not in lp:
                missing.append((match_id, pgid))
                continue
            g = lp[pgid]
            found[pgid] = (*classify(g, cfg["rules"], groups, cfg["group_stage"]), g["lp_match_id"])
        if not found:
            continue
        if len({v[:2] for v in found.values()}) != 1:
            conflict.append((match_id, found))
            continue
        tournament, stage, lp_match = next(iter(found.values()))
        out.append({"match_id": match_id, "tournament": tournament, "stage": stage,
                    "stage_source": f"leaguepedia:{lp_match}"})

    for match_id, pgid in missing:
        print(f"Leaguepedia에 없음: match {match_id} game {pgid}", file=sys.stderr)
    for match_id, found in conflict:
        print(f"한 경기 안에서 스테이지가 다름: {match_id} {found}", file=sys.stderr)
    if missing or conflict:
        raise SystemExit(f"매핑 실패 — 없음 {len(missing)}게임, 충돌 {len(conflict)}경기. match_stage_map.csv를 쓰지 않았다")

    write_csv(STAGE_MAP_CSV, ["match_id", "tournament", "stage", "stage_source"], out)
    print(f"{len(out)}경기 → {STAGE_MAP_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "fetch", "build-map"])
    ap.add_argument("pairs", nargs="?", type=Path, help="build-map: match_id·platform_game_id 열이 있는 CSV")
    args = ap.parse_args()
    if args.cmd == "build-map":
        build_map(args.pairs)
    else:
        {"discover": discover, "fetch": fetch}[args.cmd]()


if __name__ == "__main__":
    main()
