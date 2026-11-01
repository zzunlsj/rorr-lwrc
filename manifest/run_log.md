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

## 월별 tar 수령·시험 적재 (2026-10-31)

| 항목 | 실측 |
|---|---|
| 수령 | `data/incoming/2026-01~10.tar` 10개, 7.4GB (678게임, 대회 전체) |
| 형식 | 게임별 `session.json` + `events-*.ndjson.gz`(이미 gzip) — 재압축 없이 바이트 그대로 L0 복사 |
| 시험 적재 `ingest.ingest_tar 2026-08.tar` | 106게임, 1.28GB, **1.3초**, 메모리 피크 **47MB** |
| 재실행 | 덮어쓰기 없음, 같은 내용 확인 후 통과 |
| 형식 동등성 | 겹치는 10게임: CSV 추출본 L1 = tar 원본 L1 (9테이블 전부 동일), 이벤트 수 = session `event_count`, 중복 0 |
| 디스크 | `data/raw` 1.3GB, 가용 243GiB |
| 전체 추정 | L0 ≈ 7.4GB (tar와 같은 크기). 처리 피크 = tar 7.4GB + L0 7.4GB ≈ 15GB |

## 월별 tar 전체 적재 + Step 0 조사 (2026-10-31)

| 항목 | 실측 |
|---|---|
| `ingest.ingest_tar` 나머지 9개월 | 572게임, **6.4초**, 메모리 피크 49MB |
| L0 합계 | 678게임 (+ 리허설 CSV 추출본 10게임, 대체됨), `data/raw` **7.5GB** |
| `report.step0_survey` (6 프로세스) | 678게임 · 이벤트 6,049,799건, **124초**, 메모리 피크 0.9GB |
| 디스크 가용 | 235GiB |

## Step 1 적재·정규화 (2026-10-31)

| 항목 | 실측 |
|---|---|
| `normalize.build_l1 --workers 6` | 678게임, **127초**, 메모리 피크 **1.07GB** |
| L1 | 11테이블 + players/teams, **231MB** (경기당 0.34MB) |
| 제외 (재시작·ID 불일치·승자 미확정) | **0게임** (`manifest/excluded_games.csv`) |
| 선수·팀 | 선수 159명 (puuid), 팀 28 (LCK 10). 이적 6명, 복수 포지션 0 |
| 와드 | 설치 219,749 → 시야 대상 164,491. 제거 매칭 72,505/74,567 (97.2%). 교체 48,448, 만료 31,318 |
| 리허설 L1 (10게임, 이전 스키마) | `data/tmp/l1_rehearsal_2026-10-27`로 이동 보관 |
| 디스크 가용 | 235GiB |

## Step 2 상태 테이블 + G 학습 (2026-11-01)

| 항목 | 실측 |
|---|---|
| `features.state_table --workers 6` | 1Hz 2,620,716행(양 팀), **3초**, 메모리 피크 0.23GB, 56MB |
| `features.g_train --workers 6` | 학습 표본 87,964행 × 55열, 학습·검증 4초, 1Hz G 산출 포함 총 **5초**, 메모리 피크 0.95GB |
| 검증 | 부호 통과(결정 23 규칙), 경기 CV R² 0.970 / 수입 R² 0.469, 시간 홀드아웃 수입 R² 0.441 |

## Step 3 ΔG · Risk · 클러스터 (2026-11-01)

| 항목 | 실측 |
|---|---|
| `features.timeline --workers 6` | 678게임, **5초**, 메모리 피크 0.2GB, 산출 26MB (`data/l2/timeline/`) |
| 클러스터 | 11,118개 (게임당 16.4), 길이 중앙값 28.7초, 커버리지 평균 32.1% (p5 25% · p95 40%), 180초 초과 3 |
| 데스·Risk | 19,965건 (처형 51), Risk 중앙값 −940 (−1,675 ~ −150, 양수 0), 부활 대기 미관측 3 |
| 항등식 | 자팀 잔차 최대 1.5e-11, D 교차검증 잔차 0 (결정 27·28) |
| 디스크 가용 | 235GiB (data/ 15GB) |

## Step 4 배분 β (2026-11-01)

| 항목 | 실측 |
|---|---|
| `features.alloc_features --workers 6` | 선수 × 클러스터 111,180행, **5초** |
| `features.alloc_train` | 클러스터 11,118개, 크로스피팅 + 부트스트랩 200회 포함 **1초** (실행 시간 출력 기준) |
| 검증 | 표본 외 R² 0.633 (통제항만 0.086), fold별 0.608~0.647 |

## Step 5 배분 · 항등식 · 게임별 LwRC (2026-11-01)

| 항목 | 실측 |
|---|---|
| `metrics.game_lwrc --workers 6` | 678게임 × 변형 4개(main · prior_w · legacy · prior_g), prior G 타임라인 재산출 포함 **7초**, 산출 16MB (`data/l3/`) |
| 항등식 | 전 변형 최대 잔차 2.9e-11 (ε = 1.0), 미통과 0게임 |
| r̄ (prior G용 팀 분당 평균 수입) | `data/l3/step5_run.json` |

## Step 6 scope 집계 · LwRC+ · 웹 출력 (2026-11-01)

| 항목 | 실측 |
|---|---|
| `ingest.team_leagues` | Leaguepedia 2026 대회 참가 → 팀 28개 지역 리그 (LCK 10, LPL 5, LCP 4, LEC 3, LCS 3, CBLOL 3), `manifest/team_leagues_2026.csv` |
| `metrics.season` | scope 8개 × 변형 4개, split-half 부트스트랩 200회 포함 **1초** |
| 출력 | `outputs/web/players.json` 586행 (244KB), `teams.json` 106행 (24KB), `outputs/validation_report.json`, `data/l3/season_player_lwrc_plus.parquet` — 스키마 검증 통과 |
| 정본 가중 | 회귀 (split-half 0.469 vs prior 0.461) |

## 결정 30 반영 재실행 — D에 상대 데스 감속 포함 (2026-11-01)

| 항목 | 값 |
|---|---|
| `features.timeline` | 678게임, 5.3초, 메모리 피크 0.20GB |
| `features.alloc_features` · `alloc_train` | 4.9초 · 1.8초 (OOS R² 0.633, 변화 없음 — 회귀 목표 Swing은 D 정의와 무관) |
| `metrics.game_lwrc` | 8.3초, 항등식 미통과 0, 최대 잔차 2.9e-11 |
| `metrics.season` · `metrics.gold` · `report.build_site` | 1.6초 · 3.0초 · 0.1초. 정본 가중 = 회귀 (split-half 0.474 vs prior 0.466) |
| `report.compare_kill_d` | A·B 비교 수치 재현 (D 총량 467만 → 1,493만) |
