from common.db import connect
from common.paths import TMP


def test_connect_applies_memory_limit_and_spill_dir():
    con = connect()
    limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    temp_dir = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    # DuckDB는 10GB를 GiB 단위로 표시한다 (예: '9.3 GiB')
    assert "GiB" in limit and 9.0 <= float(limit.split()[0]) <= 10.0
    assert temp_dir == TMP.as_posix()
    assert TMP.is_dir()
