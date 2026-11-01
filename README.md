# rorr-lwrc

LwRC+ (LoL weighted Resources Created Plus) — RORR의 **선수 퍼포먼스 참고 지표**, 2026 시즌 산출.
승리 기여 지표가 아니다. 승리 해석은 다른 수치와 결합해 따로 다룬다 (결정 1).

- **결과 조회**: https://zzunlsj.github.io/rorr-lwrc/ (선수·팀 검색, scope 8개)
- 범위: 2026 LCK 전 일정 + First Stand + MSI + Worlds (대회 전체, 678게임 · 255경기 반영)
- 출력: scope 8개 × 선수/팀 — `all`, `lck`, `lck_s1`~`lck_s3`, `fs`, `msi`, `worlds`
- 정본: `docs/spec/` · `docs/contracts/` · `config/lwrc.yaml` · 결정 기록 `docs/decisions.md`
- 원본 사양(Notion, 이력 참고용): [LwRC+ v1.0 개정](https://app.notion.com/p/40abc2f961aa811c996df9e84f1f9820)

Claude Code로 작업할 때는 `CLAUDE.md`부터 읽힌다.

## 1. 지표 한 줄 정의

```
LwRC  = Σ (C 생산 + D 차단 + Risk)            선수가 팀 기대 골드 G를 움직인 양
LwRC+ = 100 × (선수 LwRC/분) ÷ (같은 scope · 같은 포지션 평균 LwRC/분)
```

- **G** = 10분 뒤 팀 골드의 학습 예측 (리지, 상태 피처 32개 + 패치·대회 더미, 경기 단위 5-fold 크로스피팅, 1Hz).
- **속도형 분해** (결정 27): G = 현재 골드 + V. 실현 골드는 번 사람의 C, 교전 클러스터에서
  우리 팀 가속(ΔV − Risk − 시간항 − 결정적 감쇠의 양수분)은 **C 파이**, 상대 팀 감속은 **D 파이**.
- **Risk** = 데스 단위 한계 비용 (생존 −1, 부활 대기, 상대에게 넘어간 현상금·셧다운).
- **배분** = 관여 피처 8종(딜·어시스트·피해 흡수·교전 존재·시야 제공·시야 차단·CC·보호) NNLS 리지 회귀로 지분 산출.
- 귀속하지 못한 몫(드리프트 예측 오차, 시간항, 자연 감쇠, 음수 잔여)은 **팀 계정**.
- **보존 항등식** (팀별, ε = 1골드): Σ(실현 골드 + C 파이 + Risk) + 팀 계정 = G(END) − G(0). D는 상대 장부라 넣지 않는다.

## 2. 파이프라인

`docs/workflow.md` Step 0–6. 전 과정 재실행 약 5분 (적재·정규화가 대부분, Step 2–6은 1분 이내). 메모리 피크 약 1.1GB.

| Step | 모듈 (`PYTHONPATH=src uv run python -m …`) | 산출 |
|---|---|---|
| 0 적재·조사 | `ingest.ingest_tar <tar>` → `report.step0_survey` | `data/raw/` (L0, 수정 금지), 조사 리포트 |
| 1 정규화 | `normalize.build_l1 --workers 6` | `data/l1/` frames · events · participants · wards |
| 2 G 학습 | `features.state_table` → `features.g_train` | 상태 테이블, G 계수·1Hz G |
| 3 타임라인 | `features.timeline` | 클러스터 · 드리프트 · 데스 Risk · 항등식 요약 |
| 4 배분 β | `features.alloc_features` → `features.alloc_train` | 관여 피처, β, 선수 지분 |
| 5 게임 LwRC | `metrics.game_lwrc` | `data/l3/game_player_lwrc/` (변형 4종) |
| 6 시즌 LwRC+ | `ingest.team_leagues` → `metrics.season` | `outputs/web/players.json` · `teams.json` |
| 웹 | `report.build_site` | `site/data/` → GitHub Pages 자동 배포 |

- 테스트: `PYTHONPATH=src uv run pytest` (98개)
- 단계별 보고서: `specs/step*_report_*.md` · 실측 기록: `manifest/run_log.md`
- `data/`와 `outputs/`는 커밋하지 않는다. 공개되는 것은 `site/data/`의 L4 JSON 복사본뿐이다.

### 데이터 갱신 (Worlds 잔여분 수령 후)

```bash
PYTHONPATH=src uv run python -m ingest.ingest_tar <새 tar>
# Step 1 → 6 순서대로 재실행
PYTHONPATH=src uv run python -m report.build_site
git add site/data && git commit -m "데이터 갱신" && git push   # Pages 자동 재배포
```

## 3. 검증 결과 (2026-11-01, 678게임)

| 항목 (`spec/07`) | 결과 |
|---|---|
| #1 G 계수 부호 | 통과 (억제기 타워는 규칙 제외 — 결정 23) |
| #2 G 예측력 | R² 0.970, 10분 수입 R² 0.469 (시간 홀드아웃 0.441) |
| #4 보존 항등식 | 전 게임 통과, 최대 잔차 2.9e-11. 제외 0게임 |
| #5 팀 계정 크기 | 학습 G 11.9% vs prior G 51.5% → 학습 G 채택 근거 |
| #6 D/C | 팀·게임 중앙값 0.127 (기존 ΔG 파이는 0.003) |
| 배분 회귀 | 표본 외 R² 0.633 (통제항만 0.086) |
| #9 split-half | 회귀 0.469 vs prior 가중 0.461 → 회귀 채택 (차이는 유의하지 않음) |
| #10 벤치 방향성 | TOP·JG·BOT 기대 방향, MID·SUP 역방향 — 분석 보류 |
| #11 게임 길이 상관 | −0.001 → 분모 층화 불필요 |

## 4. Notion v1.0 개정 대비 바뀐 점

자세한 근거는 `docs/decisions.md` (결정 13–29).

| 영역 | Notion v1.0 개정 | 현재 구현 |
|---|---|---|
| 작업 기준 | Notion 정본 | git 저장소 문서가 기준, Notion은 이력 (결정 13·15) |
| C·D 정의 | 클러스터 ΔG − Risk 양수 = C, 상대 ΔG 음수 = D | **속도형** — 실현 골드는 본인 C, 가속 = C 파이, 상대 감속 = D 파이 (결정 27) |
| 자연 감쇠 | 정의 없음 | 부활 대기·바론·장로 버프의 시간 감소는 팀 계정 (결정 28) |
| 클러스터 병합 | 씨앗 간격 ≤ 10초 | 인접 창 사이 간격 ≤ 10초 (문구대로면 분할이 깨짐 — 결정 25) |
| 처형 | 정의 없음 | killer = 0 → Risk만, 배분 제외. special 이벤트 무시 (결정 26) |
| 배분 피처 | 6종 | CC·보호 추가 8종 (결정 19), prior 수축 없음 (결정 29) |
| 와드 | — | `unknown` 제외 (결정 16), 지속시간 표 + 교체 규칙 (결정 17) |
| G 부호 규칙 | 타워 전체 | 외곽·2차만 (결정 23), 패치는 더미 피처 (결정 24) |
| 메타 매핑 | — | Leaguepedia 자동 매핑 (결정 18), 결측 승자·패치 대체 (결정 22) |
| LCK Split | — | s1 Cup / s2 R1-2 + Road to MSI / s3 R3-4 + PlayIn + PO (결정 20) |

## 5. 남은 항목

1. 킬로 생긴 상대 감속을 킬러 D로 볼지 (결정 27 보류분 — 현재 한타 클러스터 D > 0 비율 14%)
2. #10 서폿·미드 역방향 원인 분석
3. Worlds 잔여분 + 누락 1게임(T1 vs DK 8/14 3세트) 수령 후 재실행 (11/15 이후)
4. 추가 지표 범위 (PCI+, MapPressure, WPA — CLAUDE.md §9-4)
5. 블로그·웹 문구 수정 (`docs/decisions.md` §3, 저장소 밖)
