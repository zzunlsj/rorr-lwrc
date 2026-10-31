"""L0 적재 (월별 tar): Kazel tar의 게임별 원본 파일을 그대로 L0에 보관한다 (CLAUDE.md §2, §4, 결정 18·21).

tar 구조: <YYYY-MM>/games.csv + <YYYY-MM>/<platformID>/<gameID>/{session.json, events-0000N.ndjson.gz}

- 압축을 풀지 않는다. tar 멤버를 바이트 그대로 복사한다 (events는 이미 gzip) → 재압축 없음, 원본과 바이트 동일.
- 저장 경로: data/raw/2026/<tournament>/<stage>/<match_id>/game_<n>/ (파일 읽기 전용 0444)
- 이미 있는 파일은 덮어쓰지 않는다. 같은 내용이면 건너뛰고, 다르면 중단한다.
- 대회·스테이지는 Leaguepedia 매핑(결정 18). Kazel 라벨(games.csv tournament)과 교차 확인해 어긋나면 중단.
- manifest/received.csv에 게임 단위, manifest/received_files.csv에 파일 단위 sha256을 기록한다.

사용: uv run python -m ingest.ingest_tar data/incoming/2026-01.tar [...]
"""

import argparse
import csv
import hashlib
import io
import os
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common.paths import MANIFEST, RAW, ROOT
from ingest.leaguepedia import LWRC_YAML, map_matches, merge_stage_map
from ingest.raw_reader import RAWSTATS_NDJSON
from ingest.split_raw import SEASON, sha256_file, update_received

RECEIVED_FILES_CSV = MANIFEST / "received_files.csv"
RECEIVED_FILES_COLUMNS = ["game_id", "platform_game_id", "file", "bytes", "sha256", "raw_path", "source_file"]
MEMBER_RE = re.compile(r"^(?P<month>\d{4}-\d{2})/(?P<platform>[A-Z0-9]+)/(?P<game>\d+)/(?P<file>session\.json|events-\d+\.ndjson\.gz)$")
CHUNK = 1 << 20


def read_games_csv(tf: tarfile.TarFile, month: str) -> list[dict]:
    f = tf.extractfile(f"{month}/games.csv")
    return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")))


def copy_member(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> str:
    """멤버를 dest로 바이트 그대로 복사하고 sha256을 돌려준다. 기존 파일은 같은 내용일 때만 허용."""
    h = hashlib.sha256()
    tmp = dest.with_name(dest.name + ".part")
    src = tf.extractfile(member)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as out:
        while block := src.read(CHUNK):
            h.update(block)
            out.write(block)
    sha = h.hexdigest()
    if dest.exists():
        if sha256_file(dest) != sha:
            tmp.unlink()
            raise FileExistsError(f"L0 파일이 이미 있고 내용이 다릅니다 (덮어쓰기 금지): {dest}")
        tmp.unlink()
    else:
        os.replace(tmp, dest)
        dest.chmod(0o444)
    return sha


def ingest(tar_path: Path) -> list[dict]:
    cfg = yaml.safe_load(LWRC_YAML.read_text())["stage_map"]
    source_sha = sha256_file(tar_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with tarfile.open(tar_path, "r:") as tf:
        members: dict[str, list[tarfile.TarInfo]] = {}
        months = set()
        for m in tf.getmembers():
            hit = MEMBER_RE.match(m.name)
            if hit:
                months.add(hit["month"])
                members.setdefault(f"{hit['platform']}_{hit['game']}", []).append(m)
            elif m.isfile() and not m.name.endswith("/games.csv"):
                raise ValueError(f"예상하지 못한 tar 멤버: {m.name}")
        if len(months) != 1:
            raise ValueError(f"tar 하나에 월이 하나여야 합니다: {sorted(months)}")
        month = months.pop()
        games = read_games_csv(tf, month)

        # 무결성: games.csv ↔ 디렉터리 1:1, 게임마다 session.json + events 1개 이상
        listed = {g["platform_game_id"] for g in games}
        if listed != set(members):
            raise ValueError(f"games.csv와 디렉터리 불일치: csv만 {sorted(listed - set(members))[:5]}, 디렉터리만 {sorted(set(members) - listed)[:5]}")
        for pgid, ms in members.items():
            names = {Path(m.name).name for m in ms}
            if "session.json" not in names or not any(n.startswith("events-") for n in names):
                raise ValueError(f"{pgid}: session.json 또는 events 파일 없음 {sorted(names)}")

        # 대회·스테이지 (Leaguepedia) + Kazel 라벨 교차 확인
        pairs: dict[str, set] = {}
        for g in games:
            pairs.setdefault(g["esports_match_id"], set()).add(g["platform_game_id"])
        stage_rows = {r["match_id"]: r for r in map_matches(pairs)}
        for g in games:
            ours = stage_rows[g["esports_match_id"]]["tournament"]
            if cfg["source_labels"][ours] != g["tournament"]:
                raise ValueError(f"{g['platform_game_id']}: Leaguepedia {ours} ↔ Kazel {g['tournament']} 불일치")
        merge_stage_map(list(stage_rows.values()))

        records, file_rows = [], []
        for g in sorted(games, key=lambda g: (g["esports_match_id"], int(g["game_number"]))):
            st = stage_rows[g["esports_match_id"]]
            game_dir = RAW / SEASON / st["tournament"] / st["stage"] / g["esports_match_id"] / f"game_{g['game_number']}"
            rel_dir = game_dir.relative_to(ROOT).as_posix()
            listing, total = [], 0
            for m in sorted(members[g["platform_game_id"]], key=lambda m: m.name):
                name = Path(m.name).name
                sha = copy_member(tf, m, game_dir / name)
                listing.append(f"{name} {sha}")
                total += m.size
                file_rows.append({"game_id": g["esports_game_id"], "platform_game_id": g["platform_game_id"],
                                  "file": name, "bytes": m.size, "sha256": sha,
                                  "raw_path": f"{rel_dir}/{name}", "source_file": tar_path.name})
            records.append({
                "tournament": st["tournament"], "stage": st["stage"], "stage_source": st["stage_source"],
                "match_id": g["esports_match_id"], "game_id": g["esports_game_id"], "game_n": g["game_number"],
                "platform_game_id": g["platform_game_id"], "rows": g["event_count"],
                "bytes_csv": "", "bytes_gz": total, "sha256_csv": "",
                # 게임 단위 해시 = 파일별 "이름 sha256" 목록의 sha256 (파일별 값은 received_files.csv)
                "sha256_gz": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
                "raw_path": rel_dir, "source_file": tar_path.name, "source_sha256": source_sha,
                "ingested_at": now, "raw_format": RAWSTATS_NDJSON, "superseded_by": "",
            })
    update_received(records)
    update_received_files(file_rows)
    return records


def update_received_files(rows: list[dict]) -> None:
    existing: dict[str, dict] = {}
    if RECEIVED_FILES_CSV.exists():
        with open(RECEIVED_FILES_CSV, newline="") as f:
            existing = {r["raw_path"]: r for r in csv.DictReader(f)}
    for r in rows:
        existing[r["raw_path"]] = r
    with open(RECEIVED_FILES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECEIVED_FILES_COLUMNS)
        w.writeheader()
        w.writerows(sorted(existing.values(), key=lambda r: r["raw_path"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tars", nargs="+", type=Path)
    args = ap.parse_args()
    for tar_path in args.tars:
        t0 = time.time()
        records = ingest(tar_path)
        size = sum(int(r["bytes_gz"]) for r in records)
        print(f"{tar_path.name}: {len(records)}게임, {size / 1e9:.2f}GB, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
