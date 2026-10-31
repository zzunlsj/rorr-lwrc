"""L0 적재: 수령 CSV를 경기(게임) 단위로 분리해 pigz 압축 보관한다 (CLAUDE.md §2, §4).

- 원본 행 바이트를 재직렬화 없이 그대로 옮긴다 → 분리본을 합치면 원본과 바이트 동일.
- 저장 경로: data/raw/2026/<tournament>/<stage>/<match_id>/game_<n>.csv.gz (읽기 전용 0444)
- 이미 있는 파일은 덮어쓰지 않는다. 같은 내용이면 건너뛰고, 다르면 중단한다.
- manifest/received.csv에 파일별 sha256을 기록한다.

사용: uv run python -m ingest.split_raw <source.csv> [--stage-map manifest/match_stage_map.csv]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from common.paths import MANIFEST, RAW
from ingest.raw_reader import CSV_EXTRACT, RAWSTATS_NDJSON

SEASON = "2026"
RECEIVED_CSV = MANIFEST / "received.csv"
RECEIVED_COLUMNS = [
    "tournament", "stage", "stage_source", "match_id", "game_id", "game_n",
    "platform_game_id", "rows", "bytes_csv", "bytes_gz", "sha256_csv", "sha256_gz",
    "raw_path", "source_file", "source_sha256", "ingested_at",
    "raw_format", "superseded_by",
]
GAME_N_RE = re.compile(r"\|game(\d+)$")


@dataclass
class GameSink:
    game_id: str
    match_id: str
    platform_game_id: str
    staging_path: Path
    proc: subprocess.Popen
    out: io.BufferedWriter
    sha_csv: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    rows: int = 0
    bytes_csv: int = 0
    first_ts: str = ""
    game_n: int | None = None

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.sha_csv.update(data)
        self.bytes_csv += len(data)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_stage_map(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["match_id"]: r for r in csv.DictReader(f)}


def _unquote(field: bytes) -> str:
    s = field.decode()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] == '"' else s


def _champ_select_game_n(line: bytes) -> int | None:
    row = next(csv.reader([line.decode()]))
    name = json.loads(row[-1]).get("name", "")
    m = GAME_N_RE.search(name)
    return int(m.group(1)) if m else None


def split(source: Path, stage_map: dict[str, dict]) -> list[dict]:
    if shutil.which("pigz") is None:
        sys.exit("pigz가 없습니다: brew install pigz")
    staging = RAW / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(source)
    sinks: dict[str, GameSink] = {}
    total_rows = 0
    with open(source, "rb") as f:
        header = f.readline()
        cols = [c.strip('"') for c in header.decode().rstrip("\r\n").split(",")]
        i_game, i_match = cols.index("game_id"), cols.index("match_id")
        i_pgid, i_ts, i_schema = cols.index("platform_game_id"), cols.index("rfc460_timestamp"), cols.index("rfc461_schema")
        n_meta = cols.index("payload")  # payload 앞 필드에는 쉼표가 없다
        for line in f:
            total_rows += 1
            parts = line.split(b",", n_meta)
            game_id = _unquote(parts[i_game])
            sink = sinks.get(game_id)
            if sink is None:
                path = staging / f"{game_id}.csv.gz"
                out = open(path, "wb")
                proc = subprocess.Popen(["pigz", "-6", "-c"], stdin=subprocess.PIPE, stdout=out)
                sink = GameSink(game_id, _unquote(parts[i_match]), _unquote(parts[i_pgid]), path, proc, out)
                sink.write(header)
                sinks[game_id] = sink
            sink.write(line)
            sink.rows += 1
            ts = _unquote(parts[i_ts])
            if ts and (not sink.first_ts or ts < sink.first_ts):
                sink.first_ts = ts
            if sink.game_n is None and _unquote(parts[i_schema]) == "champ_select":
                sink.game_n = _champ_select_game_n(line)

    for s in sinks.values():
        s.proc.stdin.close()
        if s.proc.wait() != 0:
            raise RuntimeError(f"pigz 실패: {s.game_id}")
        s.out.close()

    # 바이트 보존 검증: Σ(분리본 − 헤더) + 헤더 = 원본
    total_bytes = sum(s.bytes_csv - len(header) for s in sinks.values()) + len(header)
    if total_bytes != source.stat().st_size or sum(s.rows for s in sinks.values()) != total_rows:
        raise RuntimeError("분리 결과가 원본과 바이트/행 수가 다릅니다")

    # champ_select 이름이 없으면 매치 안 첫 타임스탬프 순서로 게임 번호를 매긴다
    by_match: dict[str, list[GameSink]] = {}
    for s in sinks.values():
        by_match.setdefault(s.match_id, []).append(s)
    for games in by_match.values():
        for n, s in enumerate(sorted(games, key=lambda g: g.first_ts), start=1):
            if s.game_n is None:
                s.game_n = n

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for s in sorted(sinks.values(), key=lambda g: (g.match_id, g.game_n)):
        meta = stage_map.get(s.match_id)
        if meta is None:
            raise KeyError(f"stage map에 match_id {s.match_id}가 없습니다")
        dest = RAW / SEASON / meta["tournament"] / meta["stage"] / s.match_id / f"game_{s.game_n}.csv.gz"
        sha_gz = sha256_file(s.staging_path)
        if dest.exists():
            if sha256_file(dest) != sha_gz:
                raise FileExistsError(f"L0 파일이 이미 있고 내용이 다릅니다 (덮어쓰기 금지): {dest}")
            s.staging_path.unlink()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(s.staging_path, dest)
            dest.chmod(0o444)
        records.append({
            "tournament": meta["tournament"], "stage": meta["stage"], "stage_source": meta.get("stage_source", ""),
            "match_id": s.match_id, "game_id": s.game_id, "game_n": s.game_n,
            "platform_game_id": s.platform_game_id, "rows": s.rows,
            "bytes_csv": s.bytes_csv, "bytes_gz": dest.stat().st_size,
            "sha256_csv": s.sha_csv.hexdigest(), "sha256_gz": sha_gz,
            "raw_path": dest.relative_to(RAW.parent.parent).as_posix(),
            "source_file": source.name, "source_sha256": source_sha, "ingested_at": now,
            "raw_format": CSV_EXTRACT, "superseded_by": "",
        })
    staging.rmdir()
    return records


def update_received(records: list[dict]) -> None:
    """(game_id, raw_format) 단위로 병합. 같은 게임을 tar 원본으로도 받았으면 CSV 추출본에 superseded_by를 단다."""
    existing: dict[tuple, dict] = {}
    if RECEIVED_CSV.exists():
        with open(RECEIVED_CSV, newline="") as f:
            for r in csv.DictReader(f):
                r["raw_format"] = r.get("raw_format") or CSV_EXTRACT
                existing[(r["game_id"], r["raw_format"])] = r
    for r in records:
        key = (r["game_id"], r["raw_format"])
        prev = existing.get(key)
        if prev is None or prev["sha256_gz"] != r["sha256_gz"]:
            existing[key] = r
    ndjson_games = {gid for gid, fmt in existing if fmt == RAWSTATS_NDJSON}
    for (gid, fmt), r in existing.items():
        r["superseded_by"] = RAWSTATS_NDJSON if fmt == CSV_EXTRACT and gid in ndjson_games else ""
    with open(RECEIVED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECEIVED_COLUMNS)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda r: (r["tournament"], r["stage"], r["match_id"], int(r["game_n"]), r["raw_format"])):
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--stage-map", type=Path, default=MANIFEST / "match_stage_map.csv")
    args = ap.parse_args()
    records = split(args.source, load_stage_map(args.stage_map))
    update_received(records)
    for r in records:
        print(f"{r['raw_path']}  rows={r['rows']}  {r['bytes_csv']/1e6:.1f}MB -> {r['bytes_gz']/1e6:.1f}MB")


if __name__ == "__main__":
    main()
