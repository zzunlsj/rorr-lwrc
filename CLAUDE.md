# CLAUDE.md — rorr-lwrc

이 저장소는 **LwRC+ 2026 시즌 산출**을 위한 사양과 구현을 담는다.
Claude Code는 작업을 시작하기 전에 이 파일과 `docs/` 아래 문서를 읽고 따른다.

## 이 지표는 무엇인가

LwRC+ = **선수 퍼포먼스 지표**. 선수의 플레이가 팀의 기대 골드를 얼마나 움직였는지(생산 C + 차단 D − 허용 Risk)를
같은 범위·같은 포지션 평균 대비 비율로 표현한다. **승리 기여 지표가 아니다** — 승수 환산은 상위 지표(LoL WAR, 이 저장소 범위 밖)의 몫.

## 절대 원칙

1. **사양 문서가 유일한 정의다.** 구현 중 사양과 다른 계산이 필요해 보이면 임의로 바꾸지 말고 보고한다.
2. **웹은 표시만 한다.** LwRC+를 포함한 모든 값은 이 저장소에서 계산한 값을 그대로 출력한다. 웹 쪽 로직을 참고해 계산하지 않는다.
3. **raw 데이터는 커밋하지 않는다.** `data/` 아래에 두며 `.gitignore`로 제외되어 있다.
4. **보존 항등식을 통과하지 못한 게임은 집계에 넣지 않는다.** 제외 사유를 기록한다.
5. **결측을 0으로 채우지 않는다.** 특히 시야 피처 — 0 대체는 시야 담당 선수에게 체계적 불이익이다.

## 문서 지도 — 읽는 순서

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `docs/workflow.md` | **작업 순서와 게이트** — 반드시 먼저 |
| 2 | `docs/spec/00-overview.md` | 지표 정의, 전체 흐름 |
| 3 | `docs/contracts/input.md` | raw에서 필요한 필드, 상태 피처 32개 |
| 4 | `docs/spec/01-g-model.md` | 학습 G (레이블·학습·ΔG) |
| 5 | `docs/spec/02-timeline.md` | 클러스터·드리프트 분할 |
| 6 | `docs/spec/03-attribution.md` | Risk·파이·드리프트 귀속 |
| 7 | `docs/spec/04-allocation.md` | 선수별 배분 (관여 회귀 + 시야) |
| 8 | `docs/spec/05-identity-season.md` | 보존 항등식, 시즌 집계, LwRC+ |
| 9 | `docs/spec/06-data-prep.md` | 데이터 정제 규칙, 와드 추출 |
| 10 | `docs/spec/07-validation.md` | 검증 항목 |
| 11 | `docs/contracts/output-web.md` | 웹 출력 (8개 scope, players/teams JSON) |
| — | `config/lwrc.yaml` | 모든 상수. **코드에 상수를 하드코딩하지 않는다** |
| — | `schemas/*.schema.json` | 출력 JSON 스키마 |
| — | `docs/decisions.md` | 확정 결정, 범위 밖, 공개 전 수정 사항 |
| — | `docs/references.md` | 근거 자료 링크 |

## 용어

- **게임(game)** = 한 판. **경기(match)** = Bo3·Bo5 시리즈 전체. 학습 표본은 게임, **데이터 분할은 경기 단위**.
- **scope** = 집계 범위. `all`, `lck`, `lck_s1`, `lck_s2`, `lck_s3`, `fs`, `msi`, `worlds`.
- **팀 계정** = 개인에게 귀속하지 못한 미귀속 잔여(팀의 골드량이 아님).

## 첫 작업

`docs/workflow.md`의 **Step 0 (데이터 조사 리포트)**부터 시작한다. 리포트를 사람에게 보여주고 승인받기 전에 본 구현으로 넘어가지 않는다.
