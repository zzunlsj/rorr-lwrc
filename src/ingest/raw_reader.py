"""L0 Raw 읽기 — 형식이 달라도 같은 이벤트 스트림 (schema, sequenceIndex, payload)을 낸다.

raw_format (manifest/received.csv):
  csv_extract      game_<n>.csv.gz — CSV 추출본 (rfc461_schema, sequence_index, payload 열). 리허설 10게임
  rawstats_ndjson  game_<n>/ — 월별 tar 원본 그대로: session.json + events-0000N.ndjson.gz
                   (한 줄 = 한 페이지 {fetched_at, cursor_in, cursor_out, events:[...]})

같은 sequenceIndex가 두 번 나오면(페이지 경계 중복) 처음 것만 내고 stats["duplicates"]에 센다.
"""

import csv
import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from common.paths import ROOT

csv.field_size_limit(sys.maxsize)

CSV_EXTRACT = "csv_extract"
RAWSTATS_NDJSON = "rawstats_ndjson"


def iter_events(meta: dict, stats: dict | None = None) -> Iterator[tuple[str, int, dict]]:
    stats = stats if stats is not None else {}
    stats.setdefault("events", 0)
    stats.setdefault("duplicates", 0)
    path = ROOT / meta["raw_path"]
    fmt = meta.get("raw_format") or CSV_EXTRACT
    if fmt == CSV_EXTRACT:
        yield from _dedup(_csv_events(path), stats)
    elif fmt == RAWSTATS_NDJSON:
        yield from _dedup(_ndjson_events(path), stats)
    else:
        raise ValueError(f"알 수 없는 raw_format: {fmt}")


def read_session(meta: dict) -> dict | None:
    """rawstats_ndjson 게임의 Kazel 메타(session.json). CSV 추출본에는 없다."""
    if (meta.get("raw_format") or CSV_EXTRACT) != RAWSTATS_NDJSON:
        return None
    return json.loads((ROOT / meta["raw_path"] / "session.json").read_text())


def _dedup(events: Iterator[tuple[str, int, dict]], stats: dict) -> Iterator[tuple[str, int, dict]]:
    seen: set[int] = set()
    for schema, seq, payload in events:
        if seq in seen:
            stats["duplicates"] += 1
            continue
        seen.add(seq)
        stats["events"] += 1
        yield schema, seq, payload


def _csv_events(path: Path) -> Iterator[tuple[str, int, dict]]:
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            yield row["rfc461_schema"], int(row["sequence_index"]), json.loads(row["payload"])


def _ndjson_events(game_dir: Path) -> Iterator[tuple[str, int, dict]]:
    files = sorted(game_dir.glob("events-*.ndjson.gz"))
    if not files:
        raise FileNotFoundError(f"events-*.ndjson.gz 없음: {game_dir}")
    for file in files:
        with gzip.open(file, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                for e in json.loads(line)["events"]:
                    yield e["rfc461Schema"], int(e["sequenceIndex"]), e
