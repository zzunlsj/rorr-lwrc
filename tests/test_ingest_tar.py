"""결정 21 — 월별 tar 적재와 형식 공용 raw 리더."""

import csv
import gzip
import io
import json
import tarfile

import pytest

from common.paths import MANIFEST, RAW
from ingest import raw_reader
from ingest.ingest_tar import copy_member


def make_tar(path, files: dict[str, bytes]):
    with tarfile.open(path, "w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def ndjson_page(*events):
    return (json.dumps({"fetched_at": "", "cursor_in": None, "cursor_out": None, "events": list(events)}) + "\n").encode()


def ev(seq, schema="stats_update"):
    return {"rfc461Schema": schema, "sequenceIndex": seq, "gameTime": seq * 1000}


def test_copy_member_is_byte_identical_and_read_only(tmp_path):
    data = gzip.compress(ndjson_page(ev(1)))
    make_tar(tmp_path / "m.tar", {"2026-01/LOLTMNT03/1/events-00001.ndjson.gz": data})
    dest = tmp_path / "raw" / "events-00001.ndjson.gz"
    with tarfile.open(tmp_path / "m.tar") as tf:
        copy_member(tf, tf.getmembers()[0], dest)
        copy_member(tf, tf.getmembers()[0], dest)  # 같은 내용 재실행은 통과
    assert dest.read_bytes() == data
    assert dest.stat().st_mode & 0o777 == 0o444


def test_copy_member_refuses_overwrite_with_different_content(tmp_path):
    dest = tmp_path / "raw" / "session.json"
    dest.parent.mkdir()
    dest.write_bytes(b"old")
    make_tar(tmp_path / "m.tar", {"2026-01/LOLTMNT03/1/session.json": b"new"})
    with tarfile.open(tmp_path / "m.tar") as tf, pytest.raises(FileExistsError):
        copy_member(tf, tf.getmembers()[0], dest)
    assert dest.read_bytes() == b"old"


def test_ndjson_reader_spans_files_and_drops_duplicate_sequence(tmp_path, monkeypatch):
    game = tmp_path / "game_1"
    game.mkdir()
    (game / "events-00001.ndjson.gz").write_bytes(gzip.compress(ndjson_page(ev(1), ev(2)) + ndjson_page(ev(3))))
    (game / "events-00002.ndjson.gz").write_bytes(gzip.compress(ndjson_page(ev(3), ev(4, "champion_kill"))))
    monkeypatch.setattr(raw_reader, "ROOT", tmp_path)
    stats = {}
    out = list(raw_reader.iter_events({"raw_path": "game_1", "raw_format": "rawstats_ndjson"}, stats))
    assert [(s, q) for s, q, _ in out] == [("stats_update", 1), ("stats_update", 2), ("stats_update", 3), ("champion_kill", 4)]
    assert stats == {"events": 4, "duplicates": 1}


def test_unknown_raw_format_stops():
    with pytest.raises(ValueError, match="raw_format"):
        list(raw_reader.iter_events({"raw_path": "x", "raw_format": "parquet"}))


# 실데이터: 같은 게임을 CSV 추출본과 tar 원본으로 모두 받은 경우 L1이 같아야 한다
def _both_format_games():
    path = MANIFEST / "received.csv"
    if not path.exists():
        return []
    by: dict[str, dict] = {}
    for r in csv.DictReader(path.open()):
        by.setdefault(r["game_id"], {})[r.get("raw_format") or "csv_extract"] = r
    return [v for v in by.values() if len(v) == 2]


@pytest.mark.skipif(not RAW.exists() or not _both_format_games(), reason="두 형식으로 받은 게임 없음")
def test_l1_identical_across_raw_formats():
    from normalize.build_l1 import parse_game

    meta_only = ("raw_sha256_gz", "raw_format", "n_events", "n_duplicate_events")
    for v in _both_format_games():
        a, b = parse_game(v["csv_extract"]), parse_game(v["rawstats_ndjson"])
        for table in a:
            x = a[table].drop([c for c in meta_only if c in a[table].columns])
            y = b[table].drop([c for c in meta_only if c in b[table].columns])
            assert x.equals(y), (v["csv_extract"]["game_id"], table)
        assert b["games"]["n_events"][0] == int(v["rawstats_ndjson"]["rows"])
