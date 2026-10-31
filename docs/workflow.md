# 작업 순서와 게이트

## Step 0 — 데이터 조사 리포트 (구현 전, 필수 게이트)

월별 tar(1~10월)를 열어 **구현보다 먼저** 아래 네 가지를 리포트한다. 리포트를 사람에게 보여주고 승인받은 뒤 Step 1로 간다.

1. **필드 존재 리포트** — `docs/contracts/input.md` §1의 필드별 존재 여부·적재율을 **월별·대회별**로.
   특히 확인할 것: 좌표(x, z), 누적 챔피언 대상 피해, 와드 이벤트(위치 포함), 시리즈(경기) ID, Split, 패치, 승리 팀.
2. **표본 집계** — 게임 수 · 경기 수 · 대회별 · Split별 · 패치별 · 포지션별 선수 수.
3. **프레임 해상도** — 실제 프레임 간격(1Hz 가정 확인).
4. **데이터 품질 플래그** — 재시작 게임 수, 선수 ID 불일치 수, 승자 미상 게임 수.

없는 필드가 있으면 해당 성분을 빼는 방안을 함께 제시한다(임의로 빼고 진행하지 않는다).
`docs/decisions.md`의 확정 결정 중 데이터 때문에 바꿔야 할 것이 있으면 이 단계에서 보고한다.

## Step 1~6 — 실행 순서

| Step | 내용 | 사양 |
|---|---|---|
| 1 | 적재·정규화 (정렬, 재시작 제외, 선수 ID 매핑, 와드 이벤트 테이블) | `spec/06-data-prep.md` |
| 2 | 상태 테이블 생성 + **G 크로스피팅 학습** | `contracts/input.md` §2, `spec/01-g-model.md` |
| 3 | 전 게임 **ΔG · Risk · 클러스터** 산출 | `spec/02-timeline.md`, `spec/03-attribution.md` |
| 4 | **배분 β 크로스피팅 학습** (회귀) + prior 가중 병행 | `spec/04-allocation.md` |
| 5 | 배분 · 보존 항등식 검증 → 게임별 LwRC | `spec/03-attribution.md`, `spec/05-identity-season.md` |
| 6 | **scope별** 집계 · 분모 → LwRC+ → 웹 출력 | `spec/05-identity-season.md`, `contracts/output-web.md` |

검증(`spec/07-validation.md`)은 Step 2, 4, 5, 6 각각의 끝에서 해당 항목을 실행한다.

## 산출물

| 파일 | 내용 |
|---|---|
| `g_model` | 학습 G 계수·피처 정의·검증 지표 (fold별) |
| `price_table` | 계수 기반 단가표 (바론·드래곤·데스 등 상태 전이별 기대 ΔG) |
| `game_player_lwrc` | 게임 × 선수: C · D · Risk · LwRC · 플레이 분 · 대회 · 패치 |
| `season_player_lwrc_plus` | 선수 × scope: 게임 수 · 분 · LwRC · rate · LwRC+ · 포지션 |
| `players.json`, `teams.json` | 웹 출력 (`contracts/output-web.md`) |
| `validation_report` | 검증 전 항목 + prior G 대비 비교 + prior/회귀 가중 비교 |
| `excluded_games` | 항등식 미통과 · 재시작 · 식별 불일치로 제외된 게임과 사유 |

## 이번 버전에서 하지 않는 것

시야를 G 상태 피처로 넣는 것 · 교전 판정의 공간 조건 · 팀 효과 분리 · 승리 환산(LoL WAR) · 라이브 산출.
