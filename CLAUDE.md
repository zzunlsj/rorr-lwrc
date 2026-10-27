# CLAUDE.md — LwRC+ Local (rorr-mini)

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 항상 따르는 지침이다. 작업 전에 끝까지 읽는다.

## 1. 프로젝트 목적

- RORR(PiePie)의 선수 기여도 지표 **LwRC+ 1.0**을 로컬 Mac mini에서 구현한다.
- 입력: 개발자 Kazel이 전달하는 **2026 LoL Live Raw data**(RFC 461 계열 라이브 피드, 1Hz `stats_update` + 이벤트).
- 범위: 2026 LCK 전 일정 + First Stand + MSI + Worlds. **대회 → 스테이지 → 경기** 단위로 구분한다.
- 산출: LwRC / LwRC+ (경기·선수 단위), 그리고 같은 파이프라인 위의 추가 지표(PCI+, MapPressure 등 — 착수 전 Jun 확인).
- 결과물은 내부 팀·외부 검토자와 공유한다. 공유 대상은 **가공 지표(L3/L4)뿐**이며 Raw는 외부로 나가지 않는다.

### 용어

- **LwRC** = 원값(구 명칭 wGE). 선수 귀속 ΔG의 누적(C채널 + D채널 + Risk).
- **LwRC+** = 정규화값(구 명칭 wGE+). `100 × (선수 LwRC/분) ÷ (같은 기준 집단 포지션 평균 LwRC/분)`.
- **G(state)** = `E[팀 최종 골드 | 현재 게임 상태]` — 이 지표의 유일한 통계 객체. 이벤트 가치는 사람이 정한 상수가 아니라 G의 차분(ΔG).
- Notion 문서에는 아직 wGE/wGE+ 명칭이 남아 있다. 코드·출력에서는 **LwRC/LwRC+**로 쓴다.

## 2. 하드웨어 제약 (반드시 지킬 것)

- **메모리 16GB / 가용 디스크 약 277GB.**
- Raw는 경기당 약 **140MB(CSV)**, 경기당 약 9,500행(JSON 페이로드가 든 넓은 행). 전체 약 700게임 → **Raw만 약 100GB**.
- 따라서:
  1. **1년치를 하나의 DataFrame에 올리지 않는다.** 경기 단위 또는 대회 단위로 스트리밍 처리한다.
  2. Raw는 수령 즉시 **gzip(pigz) 압축 보관**, 처리용 데이터는 **Parquet**로 변환한다. 압축 안 된 Raw를 장기 보관하지 않는다.
  3. DuckDB는 `SET memory_limit='10GB'`와 디스크 스필 경로(`data/tmp/`)를 설정해서 쓴다.
  4. Parquet는 필요한 컬럼만 읽는다.
  5. G 학습 시 1Hz 프레임은 인접 초끼리 거의 같으므로 **2~10초 간격 샘플링**으로 학습셋을 만든다. 평가(귀속)는 1Hz 전체로 한다.
- 디스크 사용량이 200GB를 넘을 것 같으면 작업을 멈추고 Jun에게 보고한다.

## 3. 기술 스택

- Python 3.12 (`uv`로 관리, 버전 고정), DuckDB, Polars, PyArrow, NumPy, scikit-learn, LightGBM, JupyterLab
- 압축: `pigz`
- 버전 관리: Git (private repo). **`data/` 아래는 절대 커밋하지 않는다** (`.gitignore`).
- 원격 접속: Tailscale + SSH. **공인 포트 개방·포트포워딩 금지.** 외부 웹 공유(Cloudflare Tunnel)는 Jun 승인 후 별도 단계.

## 4. 디렉터리 구조 (레이어)

```
lwrc-local/
  CLAUDE.md
  data/                         ← git 제외
    raw/2026/<tournament>/<stage>/<match_id>/game_<n>.csv.gz   ← L0, 수정 금지
    l1/   tournament=/stage=/ 파티션 Parquet  ← 정규화 (frames, events)
    l2/   피처 (G 학습셋, ΔG 스트림, 클러스터)
    l3/   지표 (LwRC, LwRC+, 기타) — spec 버전 태그 필수
    tmp/  DuckDB 스필
  manifest/
    expected_games.csv          ← 대회·스테이지별 기대 게임 수 (§7)
    received.csv                ← 실제 수령 목록 (체크섬 포함)
    qc_results.csv              ← 경기별 QC 게이트 결과
  specs/
    lwrc_v1.0.md                ← 계산 명세 (동결 대상)
  src/
    ingest/  normalize/  features/  metrics/  report/
  tests/
  outputs/                      ← L4 공유용 (CSV/xlsx/HTML)
  reference/                    ← 기존 스크립트·결과 (읽기 전용 참고)
```

`<tournament>` 폴더 이름은 다음으로 고정한다:
`LCK_Cup`, `FirstStand`, `LCK_R1-2`, `Road_to_MSI`, `MSI`, `LCK_R3-4`, `LCK_PlayIn_PO`, `Worlds`

### 레이어 원칙

| 레이어 | 원칙 |
|---|---|
| L0 Raw | 받은 그대로 보관. 수정·덮어쓰기 금지. 파일마다 sha256 기록 |
| L1 Normalized | 경기별 frames(1Hz) / events 테이블. 스키마 단일화 |
| L2 Features | G 학습용 state 벡터, ΔG 스트림, 클러스터 |
| L3 Metrics | 모든 출력에 `spec_version`, `data_snapshot`(manifest 해시) 컬럼 포함 |
| L4 Share | 대상별 포맷. Raw·L1은 포함하지 않음 |

## 5. 확정된 전처리 규칙 (실측으로 검증됨 — 코드와 테스트로 고정할 것)

1. **`gameTime` 단위는 밀리초.** CSV 행은 **시간순이 보장되지 않는다** → 모든 시계열은 `gameTime` 정렬 필수.
2. **`epic_monster_kill`에 일반 정글 캠프가 섞여 있다** (raptor, wolf, krug, gromp, redCamp, blueCamp, scuttleCrab). 에픽 판정은 `monsterType ∈ {dragon, baron, riftHerald, VoidGrub}` 필터.
3. **장로 드래곤은 `monsterType='dragon'` + `dragonType='elder'`.** `teams.dragonKills`는 장로를 포함하므로 영혼 판정 시 장로를 빼야 한다. 바다 드래곤 raw 값은 `'water'`.
4. **`deathRecap`은 dict가 아니라 list**: `[{source, casterId, breakdown:[{physicalDmg, magicDmg, trueDmg, ...}]}]`. `source='Champion'`만 남기고 casterId별로 physical+magic+true를 합산. 잘못 파싱하면 배분이 한 명에게 몰리면서 **조용히 오염된다** (실제 발생 이력).
5. **`champion_kill.bounty`는 109~420 가변.** 셧다운 골드는 별도 필드 `shutdownBounty`.
6. 팀 골드는 `stats_update.teams[].totalGold`를 직접 관측값으로 쓴다 (CS×21 같은 근사 금지). `alive`, `respawnTimer`, `game_end.winningTeam` 존재.
7. **1Hz 전제 성립**: `stats_update` 간격 평균 0.9996s, p95 1.003s. 이 범위를 벗어나는 경기는 QC 경고.
8. 참고: `ward_placed`, `ward_killed`, `item_purchased` 이벤트는 피드에 **존재한다** (일부 Notion 문서의 "수집 결손" 서술은 구버전).

## 6. QC 게이트 (경기마다 자동 실행, 결과는 `manifest/qc_results.csv`)

1. **기대 게임 대조**: `expected_games.csv`와 수령 목록 비교 → 누락·중복 보고.
2. **1Hz 연속성**: 틱 간격 p95, 최대 간격, 결측 구간.
3. **보존 항등식 (ε = 1.0골드)**:
   `Σ개인 귀속 C + Σ Risk + 팀 버킷 = 실제 최종 팀 골드 − G(초기 상태)`
   - 올바른 구현이면 잔차는 1e-12 수준, 버그면 수천 골드 단위로 튄다.
   - **D채널은 항등식에 넣지 않는다** (상대 G의 변화 = 다른 장부). 넣으면 잔차 −7.6k 수준 오염(재현 이력).
4. **틱 구간 완전 분할**: 각 틱 구간 `(t_{i-1}, t_i]`는 **중점(midpoint) 기준으로 정확히 하나의 버킷**(클러스터 k 또는 드리프트)에 귀속. 창 경계 시각의 G값으로 따로 계산하면 이중계상(잔차 약 9,200골드 이력).
5. **팀 버킷 크기**는 G 품질의 진단값으로 리포트에 포함.

QC 실패 경기는 L3 산출에서 제외하고 목록으로 보고한다. 조용히 넘기지 않는다.

## 7. 기대 게임 수 (2026, 기준일 2026-10-27, 출처 Leaguepedia)

| tournament | 매치 | 게임 | 상태 |
|---|---|---|---|
| LCK_Cup | 40 | 125 | 확정 (그룹 25/71, 플레이인 5/13, 플레이오프 10/41) |
| FirstStand | 13 | 45 | 확정 (LCK 팀 출전 20게임: GEN 9, BFX 11) |
| LCK_R1-2 | 90 | 204 | 확정 |
| Road_to_MSI | 5 | 20 | 확정 |
| MSI | 20 | 71 | 확정 (플레이인 6/18, 브래킷 14/53. LCK 팀 출전 41게임: T1 21, HLE 20) |
| LCK_R3-4 | 40 | 99 | 확정 (레전드 20/50, 라이즈 20/49) |
| LCK_PlayIn_PO | 13 | 55 | 확정 (플레이-인 3/14, 플레이오프 10/41) |
| Worlds | 46 | 90~113 | 진행 중 (11/14 결승). 10/27 기준 30매치·51게임 확정 |
| **합계** | **267** | **709~732** | |

Kazel이 대회 전체를 주는지, LCK 팀 출전 경기만 주는지는 미확정이다. `expected_games.csv`에는 둘 다 표시할 수 있게 `lck_team_involved` 컬럼을 둔다.

## 8. LwRC 계산 개요 (상세는 `specs/lwrc_v1.0.md`, 근거는 Notion)

- **채널**: C(우리 팀 기대 골드 상승) / D(상대 팀 기대 골드 하락) / Risk(본인 데스로 인한 반대 방향).
- **클러스터**: 씨앗 이벤트 `champion_kill`, `epic_monster_kill`(에픽 필터 후), `building_destroyed`. 각각 ±10,000ms 창, 간격 ≤10,000ms면 연쇄 병합, 게임 시작·종료에서 절단.
- **δ_noise**: 110 골드/초 (튜닝 범위 60~250).
- **콜드스타트 배분**: 클러스터 ΔG_us > 0 → deathRecap 딜 비중 / 딜 없음 → killer+assistants 균등 / 그것도 없음 → 팀 버킷. 음수 → 클러스터 내 본팀 데스 균등(Risk). 드리프트 → 개인 `totalGold` 증가분을 본인 C로. 상대 정상 획득의 음수 ΔG → 팀 버킷.
- **G 모델 순서**: v1 = **분 버킷 선형회귀**(계수 해석 가능) → v2 = GBM(LightGBM). 처음부터 GBM으로 가지 않는다.
- **G 인터페이스 추상화**: 임시 G(R(m) 테이블) → 학습 G로 함수 하나만 갈아끼우게 설계.
- **누수 방지**: 2026 데이터로 학습과 평가를 같이 하므로 **leave-one-tournament-out**(해당 대회를 빼고 학습한 G로 그 대회를 평가)을 기본으로 한다.
- **후반 생존 편향**: 28분 이후 표본 감소 → R(m)·잔여 기대에 isotonic(단조 비증가) 스무딩.
- **정규화**: 윈저라이즈 ±3σ, prior 수축 `w=(n×실측+k×prior)/(n+k)`, k=10.

## 9. 미결 사항 — 임의로 정하지 말고 반드시 Jun에게 물어볼 것

1. **음수 표기 정책**: Notion 설계 문서는 "음수 허용", 이후 결정은 "음수 없음, 평균=100 지수 표기(예: 126%)". 1.0 스펙에서 하나로 확정 필요.
2. **LwRC+ 기준선(=100) 집단**: 대회별 / 2026 전체 풀 / LCK 기준 중 무엇인지.
3. **패치 처리**: 패치를 G의 피처로 넣을지, 구간별로 나눌지.
4. **추가 지표 범위**: PCI+, MapPressure(19존), WPA 프로토타입 중 무엇을 만들지.
5. **Raw 수령 범위**: 대회 전체 vs LCK 팀 출전 경기만.

이 항목에 걸리는 코드는 설정값(config)으로 빼두고, 기본값을 넣었다면 출력에 "미확정" 표시를 남긴다.

## 10. 금지 사항

- L0 Raw 수정·삭제·덮어쓰기
- `data/` 커밋, Raw·L1을 `outputs/`에 포함
- 공인 포트 개방, 외부 서비스로 데이터 업로드
- `sudo`가 필요한 작업을 확인 없이 진행 (항상 무엇을 왜 하는지 먼저 말할 것)
- §9 미결 사항을 그럴듯한 값으로 조용히 확정

## 11. 작업 방식

- 단계(KICKOFF.md의 단계 단위)가 끝날 때마다 **무엇을 했고, 무엇이 확인됐고, 다음에 무엇을 할지** 짧게 보고하고 멈춘다.
- 실측 수치(처리 시간, 메모리 피크, 압축률, 디스크 사용량)는 `manifest/run_log.md`에 누적 기록한다.
- 기존 자산(`reference/`)의 결과와 새 파이프라인 결과가 다르면, 차이를 표로 보여주고 원인을 설명한 뒤 Jun 판단을 받는다.

## 12. 참고 문서 (Notion — 근거 원문)

- LOL PCI+ 설계: https://app.notion.com/p/395bc2f961aa811cbcfaf71f9ead76bf
- wGE+(=LwRC+) 설계: https://app.notion.com/p/3aabc2f961aa81c5a039fb468eed3da1
- 콜드스타트 설정값 v0.1 (LCK 10게임 실측): https://app.notion.com/p/3c3bc2f961aa818f9739cc7c1102a466
- v0 형상 검증 스모크: https://app.notion.com/p/3babc2f961aa819c83decb7da3ce34fa
- 2026 게임 수 표: https://app.notion.com/p/401bc2f961aa80aca887fbd0a9861c44
