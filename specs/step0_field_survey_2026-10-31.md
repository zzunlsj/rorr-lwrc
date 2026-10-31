# Step 0 데이터 조사 — 사전 점검 (2026-10-31)

> **전체 데이터 리포트는 `specs/step0_report_2026-10-31.md`** (월별 tar 678게임). 이 문서는 10게임 사전 점검 기록이다.

- 기준: `docs/workflow.md` Step 0, `docs/contracts/input.md` §1
- **대상: 보유 중인 10게임 CSV**(LCK_R3-4, 2026-08-07~09, Week 11, 패치 16.15) — 월별 tar(1~10월)는 아직 미수령.
  따라서 "월별·대회별 적재율"은 산출할 수 없고, 이 문서는 **필드 존재 여부와 형식의 사전 확인**이다. tar 수령 후 같은 스크립트로 재실행해야 Step 0이 완료된다.
- 조사 스크립트: 스키마 37종 전수, 키 경로별 행·게임 적재율 (10게임 95,581행, 15초)

## 1. 필드 존재 리포트

| 단위 | 필수 필드 | 원본 위치 | 적재율 | 판정 |
|---|---|---|---|---|
| 프레임·참가자 | 게임 시각 | `stats_update.gameTime` | 100% | 있음 |
| | 참가자 ID · 팀 | `participants[].participantID / teamID` | 100% | 있음 |
| | 누적 획득 골드 | `participants[].totalGold` | 100% | 있음 |
| | 레벨 · 생존 · 부활 잔여 | `level / alive / respawnTimer` | 100% | 있음 |
| | 체력 · 최대 체력 | `health / healthMax` | 100% | 있음 |
| | **좌표 (x, z)** | `participants[].position.x/z` | 100% | 있음 |
| | **누적 챔피언 대상 피해** | `participants[].stats[]` 의 `TOTAL_DAMAGE_DEALT_TO_CHAMPIONS` | 100%, 결측 0, 감소 0, 매 틱 갱신 | **있음** → 딜 피처 `frame_delta` 사용 가능 |
| 챔피언 킬 | 시각 · 킬러 · 피해자 · 어시 · 현상금 · 좌표 | `champion_kill.*` (+ `shutdownBounty`, `killerTeamID`, `victimTeamID`, `fightDuration`) | 100% | 있음 |
| 에픽 몬스터 | 시각 · 처치 팀 · 종류 · 좌표 | `epic_monster_kill.killerTeamID / monsterType / position` | 100% | 있음 |
| | 드래곤 속성 · 장로 구분 | `dragonType` (hextech·water·earth·fire·air·chemtech·**elder**) | 드래곤 행 100% | 있음 |
| 건물 | 시각 · 파괴된 쪽 · 종류 · 라인 · 좌표 | `building_destroyed.teamID / buildingType / lane / position` | 100% (`lane` 93%: 넥서스 타워는 `nexusTurretName`) | 있음 |
| | 티어 | `turretTier` | 타워 행 100% (억제기·넥서스 행에는 없음) | 있음 |
| | 억제기 리스폰 | `building_respawned` | 1게임 3건 | 있음 (피처 12–13 산출 가능) |
| 와드 | 설치: 시각 · 위치 · 설치자 · 종류 | `ward_placed.gameTime / position / placer / wardType` | 100% | 있음 (§3-2 주의) |
| | 제거: 시각 · 제거자 · 위치 · 종류 | `ward_killed.gameTime / killer / position / wardType` | 100% | 있음 |
| | 설치↔제거 연결 키 | — | **없음** | 위치(±50)·시각 매칭: **1,148 / 1,180 = 97.3%** (게임별 92~100%) |
| 메타 | 게임 ID · 경기(시리즈) ID | CSV `game_id`, `match_id` | 100% | 있음 |
| | 시리즈 내 세트 번호 | `gameName` = `<match_id>\|gameN` | 100% | 있음 |
| | 패치 | `game_info.gameVersion` (16.15.800.4844) | 10/10 | 있음 |
| | 날짜 | CSV `match_start_time` | 100% | 있음 |
| | 블루·레드 팀 | `teamID` 100/200 + `playerName` 팀 접두어, `champ_select.teamOne/teamTwo` | 100% | 있음 |
| | 승리 팀 | `game_end.winningTeam` | 10/10 | 있음 |
| | **대회 · Split** | CSV `block_name`은 `"Week 11"`뿐 | — | **없음** → 매핑 테이블 필요 |
| 선수 | 고유 ID · 표시 이름 · 소속 팀 · 포지션 | `puuid`, `riotId.displayName`("T1 Doran"), `role` | 100% | 있음 |

## 2. 표본 집계 (10게임)

| 항목 | 값 |
|---|---|
| 게임 / 경기 | 10 / 5 (Legend 2경기 = 3+3게임, Rise 3경기 = 1+2+1게임) |
| 대회 · Split · 패치 | LCK_R3-4 (Legend 6 / Rise 4, 스테이지는 시간대 추정) · Split 미상 · 16.15 단일 |
| 팀 · 포지션별 선수 | 9팀 · 포지션마다 9명 (출전 20회씩) |

## 3. 프레임 해상도와 품질 플래그

| 항목 | 결과 |
|---|---|
| 선언 간격 | `game_info.statsUpdateInterval = 1000` ms |
| 실측 간격 | mean 0.9996초 · p95 1.003 · max 1.03, 2초 초과 공백 0 (1Hz 가정 확인) |
| 재시작(리메이크·크로노브레이크) | 0 — `generationID`·`parentGameID` 모두 0, 게임 시각 역행 0 |
| 선수 ID 불일치 | 0 — 게임 안 participantID ↔ puuid·이름 변동 없음 |
| 승자 미상 | 0 |
| 일시정지 | 1게임 (`pause_started/ended`) — 틱 간격 QC 통과 |

## 4. 사양과 맞춰야 할 점 (결정 필요)

1. ~~**씨앗 몬스터 이름**~~ → **해결 (2026-10-31)**: 원본 표기 그대로 `riftHerald`, `VoidGrub` 사용. `horde` 명칭은 쓰지 않음 (`docs/decisions.md` 결정 14).
   설정에 없는 monsterType은 L1 적재에서 오류로 멈춤. `atakhan`은 10게임에 없어 원본 표기 미확인 — 다른 표기로 나오면 이 가드에 걸림.
2. ~~**`wardType = unknown`은 와드가 아님**~~ → **해결 (결정 16)**: 시야 피처에서 제외.: 설치 4,202건 중 1,616건(38%). 설치자 챔피언은 요릭 612, 쉔 304, 진 275, 바드 158, 밀리오 76, 칼리스타 18, 니달리 15 등입니다. 챔피언 고유 설치물이 와드 이벤트로 기록된 것으로 보입니다(한 게임에서 요릭 1명이 391건).
   → 시야 피처에서 `unknown`을 제외하자고 제안. 제외하지 않으면 해당 챔피언의 `vision_provided`가 크게 부풀려짐.
3. **와드 생존 구간** → **방침 확정 (결정 17)**: 공식 패치값 표 + 실측 검증. 표 작성은 남음.: 연결 키가 없어 위치·시각으로 매칭(97.3%)합니다. 제거 기록이 없는 와드는 종류별 지속시간으로 만료를 추정해야 합니다.
   → 16.15 기준 지속시간 표(sight, yellowTrinket, blueTrinket, control)를 확정해야 함. 설치자 ID가 없는 설치(`placer` 미매핑) 158건은 제외.
4. ~~**대회 · Split 정보 없음**~~ → **해결 (결정 18)**: Leaguepedia `RiotPlatformGameId`로 정확 조인. 보유 10게임 전부 매핑, 기존 수동 판정과 5경기 모두 일치. 대회별 게임 수도 기대값과 일치(Worlds는 진행 중). LCK Split 구분은 결정 20.: scope 8개(`lck_s1~s3`, `fs`, `msi`, `worlds`)를 나누려면 `match_id → 대회 · Split · 스테이지` 매핑 테이블이 필요합니다. 지금은 수동 `manifest/match_stage_map.csv`로 관리하고 있습니다.
   → tar에 이 정보가 있는지 확인하고, 없으면 Kazel 측 manifest나 Leaguepedia에서 받아와야 함.
5. ~~**사양 §4.6 한계를 보완할 수 있는 필드가 있음**~~ → **해결 (결정 19)**: CC·보호 피처를 정본에 추가.: `stats[]`에 `TOTAL_DAMAGE_SHIELDED_ON_TEAMMATES`, `TOTAL_HEAL_ON_TEAMMATES`, `TOTAL_TIME_CROWD_CONTROL_DEALT_TO_CHAMPIONS`, `TIME_CCING_OTHERS`가 누적값으로 있습니다.
   사양은 "CC 시간·제공 실드/힐 데이터가 없는 한 이니시에이터가 과소평가될 수 있다"고 적었는데, 이 데이터가 실제로 있습니다. 관여 피처에 추가할지는 사양 변경이므로 판단이 필요합니다. 임의로 넣지 않았습니다.
6. **수령 형식**: 조사한 파일은 CSV 추출본(`game_id, match_id, match_start_time, block_name, platform_game_id, sequence_index, rfc460_timestamp, rfc461_schema, game_time, payload`)입니다. 월별 tar의 내부 형식이 같은지는 수령 후 확인해야 합니다.

## 5. 결론

- 사양이 요구하는 필드는 **대회·Split을 빼고 모두 있습니다.** 누적 피해, 좌표, 와드 위치 모두 있으므로 성분을 빼야 할 것은 없습니다. 관여 6피처(시야 2종 포함) 회귀를 그대로 진행할 수 있습니다.
- 진행 전에 정해야 할 것은 §4의 1~5입니다. 6은 tar 수령 후 Step 0을 다시 돌리면 확인됩니다.

## 6. 월별 tar 수령 확인 (2026-10-31, 압축 해제 없이 목록·메타만)

- 수령: `data/incoming/2026-01.tar` ~ `2026-10.tar` 10개, 7.4GB.
- **형식이 이번 CSV 추출본과 다르다**: `<YYYY-MM>/games.csv`(월별 메타) + `<YYYY-MM>/<platformID>/<gameID>/session.json` + `events-0000N.ndjson.gz`(한 줄 = 한 페이지 `{fetched_at, cursor_in, cursor_out, events:[raw 이벤트]}`). 이벤트 자체는 같은 RFC 461 raw.
  → `ingest.split_raw`(CSV 입력)를 이 형식에 맞게 확장해야 한다.
- 게임 678개, 디렉터리 678개 — session.json·events 누락 0, platform_game_id 중복 0, 전부 `is_primary=True`·`event_status=complete`. 이벤트 수 5,748~14,102 (중앙값 8,830). 패치 16.1~16.20.
- **범위 = 대회 전체** (LCK 팀 미출전 경기 포함): First Stand 45, MSI 71, Worlds 60.
- Kazel 라벨 `lck_split_1/2/3_2026`이 결정 20의 Split 구분과 정확히 일치. 스테이지 라벨도 Leaguepedia 분류와 1:1 대응 (R3-4는 둘 다 `groups` → 레전드/라이즈는 Leaguepedia 그룹으로 판정).

| 대회 | 스테이지 | Leaguepedia | tar | 차이 |
|---|---|---|---|---|
| LCK_Cup | Group / PlayIn / Playoffs | 71 / 13 / 41 | 71 / 13 / 41 | — |
| FirstStand | Group / Knockout | 35 / 10 | 35 / 10 | — |
| LCK_R1-2 | Regular | 204 | 204 | — |
| Road_to_MSI | Qualifier | 20 | 20 | — |
| MSI | PlayIn / Bracket | 18 / 53 | 18 / 53 | — |
| LCK_R3-4 | Legend / Rise | 50 / 49 | 48 / 49 | 아래 ①② |
| LCK_PlayIn_PO | PlayIn / Playoffs | 14 / 41 | 14 / 41 | — |
| Worlds | PlayIn / Swiss | 22 / 44 | 22 / 38 | 아래 ③ |

① **GEN vs DK 2026-08-01 1세트**: Leaguepedia에 게임 ID가 없고 tar에는 있음(`LOLTMNT02_443761`). 같은 게임 → 매핑 수동 지정 필요.
② **T1 vs DK 2026-08-14 3세트(`LOLTMNT01_444568`)**: tar에 없음 → Kazel에게 확인 필요.
③ **Worlds 6게임(10/29 23시 이후)**: tar 수집 시점 이후 경기 — 다음 수령분.
