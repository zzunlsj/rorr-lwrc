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

## 2026-10-27 5단계 10게임 리허설 (LCK_R3-4, 패치 16.15)

### 단계별 실측 (단일 프로세스, Mac mini 16GB)

| 단계 | 명령 | wall | 경기당 | 메모리 피크(RSS) |
|---|---|---|---|---|
| 5-1 ingest (분리+pigz+sha256) | `ingest.split_raw` | 2.76s | 0.28s | 47MB |
| 5-2 normalize (L1 Parquet) | `normalize.build_l1` | 13.17s | 1.32s | 868MB |
| 5-4 metrics v0.1 (G+클러스터+귀속+QC+정규화) | `metrics.run --config v0_1_coldstart` | 0.56s | 0.009s(+로딩) | 227MB |
| 5-4 metrics v1.0 | `metrics.run --config v1_0` | 0.59s | 0.011s(+로딩) | 228MB |

### 압축·디스크

| 항목 | 값 |
|---|---|
| 원본 CSV | 1,441MB (경기당 144.1MB) |
| L0 csv.gz (pigz -6) | 147.8MB (경기당 14.8MB), **압축률 9.75배** |
| L1 Parquet (zstd, 9테이블) | 3.1MB (경기당 0.31MB) |
| L3 (2개 spec) | 0.4MB |
| 분리 검증 | 정렬 행 sha256 원본=분리본 (바이트 동일), 재실행 시 덮어쓰기 없음 |
| 디스크 가용 (5단계 후) | 243GiB |

### 700게임 추정 (기대 게임 709~732, 상한 732로 계산)

| 항목 | 추정 |
|---|---|
| 처리 시간 | ingest 3.4분 + L1 16분 + metrics 1분 미만(spec당) ≈ **약 20분** (단일 프로세스, 전송 시간 제외) |
| L0 csv.gz | 14.8MB × 732 ≈ **10.8GB** |
| L1 | 0.31MB × 732 ≈ 0.23GB |
| 수령 원본(비압축) 임시 | 144MB × 732 ≈ 105GB — 대회 단위로 받아 ingest 후 정리하면 피크는 최대 대회(LCK_R1-2 204게임) ≈ 29GB |
| 피크 디스크 (대회 단위 처리 시) | 약 29GB(임시) + 11GB(L0) + 1GB ≈ **41GB** → 200GB 한도 대비 여유 |
| 메모리 | 피크 0.9GB (L1 변환). G 학습(2~10초 샘플링)은 별도 측정 필요 |
