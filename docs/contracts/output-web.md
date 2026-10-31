# 웹 출력 계약

> **원칙**: 웹은 표시만 한다. LwRC+를 포함한 모든 값은 이 저장소에서 계산한 값을 그대로 쓴다. 웹에서 재계산하지 않는다.
> 웹 페이지는 "어떤 항목을 보여주는가"만 참고했다(로직은 참고하지 않음).

## 1. scope — 8개 전부 산출

```
전체(all)
LCK ─┬─ ALL      (lck)
      ├─ Split 1  (lck_s1)
      ├─ Split 2  (lck_s2)
      └─ Split 3  (lck_s3)
First Stand        (fs)
MSI                (msi)
Worlds             (worlds)
```

| scope | 포함 게임 | 분모 모집단 |
|---|---|---|
| `all` | 2026 전체: LCK + First Stand + MSI + Worlds | 해당 게임의 모든 선수 (타 리그 포함) |
| `lck` | LCK 전체 Split | LCK 선수 |
| `lck_s1` / `lck_s2` / `lck_s3` | LCK 각 Split | LCK 선수 |
| `fs` / `msi` / `worlds` | 각 국제대회 | 해당 대회 출전 전원 (타 리그 포함) |

- scope마다 rate·분모를 **따로** 계산 (`spec/05` §3). 같은 선수라도 scope별 LwRC+가 다를 수 있다 — 정상.
- 타 리그 선수는 `lg` 필드로 구분.

## 2. `players.json` — 선수 × scope

스키마: `schemas/players.schema.json`

| 필드 | 의미 | 산출 |
|---|---|---|
| `scope` | 표시 범위 | §1 |
| `player_id`, `name` | 선수 | 선수 매핑 테이블 |
| `team`, `lg` | 소속 팀·리그 | 해당 scope에서 가장 많이 뛴 팀 |
| `role` | `TOP` / `JG` / `MID` / `BOT` / `SUP` | |
| `g` | 게임 수 | |
| `minutes` | 플레이 분 | |
| **`ix`** | **LwRC+** (정수, 단위 없음, 음수 허용) | `100 × rate / denom(scope, role)` |
| **`pg`** | **Gold +/G** — 포지션 평균 대비 게임당 초과 골드 (정수) | `(rate − denom) × minutes / g` |
| `lwrc`, `rate` | 범위 LwRC, 분당 LwRC | `spec/05` §3 |
| `c`, `d`, `risk` | 채널별 합 | 상세 보기·검증용 |
| `role_rank`, `role_n` | 같은 scope·포지션 내 LwRC+ 순위 / 대상 수 | `g ≥ min_games_for_ranking` 선수 중, 미달이면 null |
| `team_share` | 팀 내 기여 비중(%) | `max(0, lwrc) / Σ_팀 max(0, lwrc)` — 절대량 기준(캐리가 크게 나오는 것이 정상), 표시 시 "골드 기여 기준" 명시 권장 |

## 3. `teams.json` — 팀 × scope

스키마: `schemas/teams.schema.json`

| 필드 | 의미 | 산출 |
|---|---|---|
| `scope`, `team`, `lg` | | |
| `W`, `L` | 경기(시리즈) 승패 | 메타의 시리즈 결과 |
| `sw`, `sl`, `sd` | 게임 승·패·득실 | |
| `wr` | 게임 승률 | `sw / (sw + sl)` |
| `form` | 최근 5경기 결과 문자열 (예: `"WWLWL"`) | |
| `g` | 게임 수 | |
| **`ix`** | **팀 LwRC+** | 소속 선수 LwRC+의 **플레이 분 가중 평균** (교체 선수 반영) |
| `pg` | 팀 Gold +/G | 소속 선수 pg 합 (게임당) |
| `top_role` | 팀 내 LwRC+ 최고 포지션 | |

## 4. 표시 규칙

- `ix`는 숫자만 (`126` = 평균 대비 +26%), 음수 그대로.
- 최소 게임 필터(0 / 10 / 25)는 웹이 `g`로 거르므로 **전 선수 출력**.
- 각 파일 최상위에 메타 포함: `version`, `generated_at`, `allocation_method`(`prior` | `regression`), `g_model_version`.
- 권장 안내 문구: "선택한 대회 범위 안에서 같은 포지션 평균을 100으로 계산".
