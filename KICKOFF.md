# KICKOFF — Claude Code 시작 프롬프트

## 사용법

1. 미니 맥에서 1단계(macOS 기본 세팅)와 2단계(SSH, Tailscale)를 **직접** 끝낸다.
2. `~/lwrc-local/` 폴더를 만들고 `CLAUDE.md`를 넣는다.
3. 기존 자산을 `~/lwrc-local/reference/`에 복사한다:
   - LCK 10게임 CSV (`v_lck_recent_raw_frames_202608101931.csv`, 1.44GB)
   - `lol_live_raw_project/handoff/` (extract.py, analyze.py, wge_coldstart_config.json)
   - `wge_g_smoke.py`, `wge_g_smoke_results.json`, `five_pillars.py`, `wrc_five_pillars_3games.csv`
4. 그 폴더에서 `claude`를 실행하고 아래 프롬프트를 붙여 넣는다.
5. 권한 건너뛰기 옵션(`--dangerously-skip-permissions`)은 쓰지 않는다.

---

## 붙여 넣을 프롬프트

```
CLAUDE.md를 끝까지 읽고 LwRC+ Local 환경을 세팅해 줘.
1~2단계(macOS 설정, SSH, Tailscale)는 내가 이미 끝냈어.
아래 단계를 순서대로 진행하고, 각 단계가 끝날 때마다 보고하고 멈춰. 내가 "다음"이라고 하면 이어서 해.

[3단계] 개발 환경
- 현재 상태 점검: macOS 버전, 메모리, 디스크 여유, Homebrew·git 유무
- Homebrew, git, uv, duckdb, pigz 설치 (sudo가 필요하면 먼저 이유를 말해)
- uv로 Python 3.12 프로젝트 생성, 패키지 추가: duckdb polars pyarrow numpy scikit-learn lightgbm jupyterlab pytest

[4단계] 프로젝트 골격
- CLAUDE.md §4 구조대로 폴더 생성, .gitignore(data/ 제외), git init
- manifest/expected_games.csv를 CLAUDE.md §7 표로 작성 (tournament, stage, matches, games_min, games_max, status, lck_team_involved)
- DuckDB 공통 연결 헬퍼: memory_limit 10GB, temp_directory=data/tmp
- specs/lwrc_v1.0.md 초안: CLAUDE.md §8 내용 + §9 미결 사항을 "미확정"으로 표시

[5단계] 10게임 리허설 — 가장 중요
reference/의 10게임 CSV와 기존 스크립트로 파이프라인을 끝까지 한 번 돌린다.
5-1 ingest: CSV를 경기별로 분리 → pigz 압축 → data/raw/2026/LCK_R1-2/... 저장, sha256, manifest/received.csv 기록
5-2 normalize: CLAUDE.md §5 전처리 규칙 8개를 코드로 구현하고, 규칙마다 pytest 테스트 작성 → L1 Parquet
5-3 QC 게이트: CLAUDE.md §6 전부 구현 → manifest/qc_results.csv
5-4 metrics: 콜드스타트 설정(wge_coldstart_config.json, R(m) 테이블)으로 LwRC 산출 → L3
5-5 대조: reference/의 기존 결과(콜드스타트 v0.1 문서 수치, wge_g_smoke_results.json)와 비교해 차이를 표로 보고
5-6 실측 기록: 경기당 처리 시간, 메모리 피크, 압축률, 디스크 사용량 → manifest/run_log.md
     그리고 이 수치로 700게임 전체 처리 시간과 디스크 사용량을 추정해 줘

리허설 결과가 기존 결과와 맞으면 "Raw 수령 준비 완료" 체크리스트를 만들어 줘.
CLAUDE.md §9 미결 사항에 걸리면 임의로 정하지 말고 질문해.
```

---

## 단계별 완료 판정

| 단계 | 완료 조건 |
|---|---|
| 3 | `uv run python -c "import duckdb, polars, lightgbm"` 성공 |
| 4 | 폴더 구조, expected_games.csv, git 첫 커밋 |
| 5-2 | 전처리 규칙 테스트 전부 통과 |
| 5-3 | 10게임 × 양팀 보존 항등식 잔차 < 1.0 |
| 5-5 | 기존 결과와 수치 일치, 또는 차이 원인 설명 |
| 5-6 | 700게임 처리 시간·디스크 추정치 보고 |
