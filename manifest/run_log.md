# run_log — 실측 기록 (CLAUDE.md §11)

## 2026-10-27 환경 구축 (3~4단계)

| 항목 | 값 |
|---|---|
| 장비 | Mac mini, macOS 27.2, 메모리 16GB |
| 디스크 | 전체 460GiB, 가용 244GiB (4단계 시점) |
| Python | 3.12.13 (uv 0.11.18) |
| 주요 패키지 | duckdb 2.0.0, polars 2.1.0, pyarrow 26.0.0, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0 |
| 시스템 도구 | duckdb CLI 2.0.0, pigz 2.8, libomp (LightGBM 필수) |
| .venv 크기 | 665MB |

### reference CSV 프로파일 (DuckDB CLI, 컬럼 6개만 읽음)

| 항목 | 값 |
|---|---|
| 파일 | reference/v_lck_recent_raw_frames_202608101931.csv, 1,441,068,060 bytes |
| 행 / 게임 / 매치 | 95,581 / 10 / 5 (게임당 7,949~11,864행) |
| stats_update | 게임당 1,707~2,444행 |
| 스캔 시간 | 3.1초 (wall) |
| 메모리 피크 (RSS) | 548MB |
