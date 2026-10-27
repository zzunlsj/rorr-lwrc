"""DuckDB 공통 연결 헬퍼 (CLAUDE.md §2-3).

메모리 16GB 장비이므로 모든 DuckDB 작업은 이 함수로 연결해
memory_limit와 디스크 스필 경로를 강제한다.
"""

from pathlib import Path

import duckdb

from common.paths import TMP

DEFAULT_MEMORY_LIMIT = "10GB"


def connect(
    database: str | Path = ":memory:",
    *,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database), read_only=read_only)
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    if threads is not None:
        con.execute(f"SET threads = {int(threads)}")
    return con
