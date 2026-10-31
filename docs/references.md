# 참고 자료

## 1. 게임 규칙 근거

| 근거로 쓴 곳 | 자료 | 링크 |
|---|---|---|
| 클러스터 병합 10초 (멀티킬 판정 기준) | Quora — 멀티킬 조건 | https://www.quora.com/In-the-League-of-legends-how-do-we-go-for-a-multi-kill-and-does-it-give-any-experience-benefit |
| 같은 근거 (펜타 30초 예외 포함) | Reddit r/leagueoflegends | https://www.reddit.com/r/leagueoflegends/comments/2yl0z5/how_long_do_you_have_to_get_that_penta/ |
| 부활 시간 공식 (레벨별 기본치 + 15분 이후 가산, 최대 약 79초) | LoL Wiki — Death | https://wiki.leagueoflegends.com/en-us/Death |
| 같은 근거 (보조) | LoL Fandom Wiki — Death | https://leagueoflegends.fandom.com/wiki/Death |
| 데스 비용 해석 참고 | Dignitas — An in-depth analysis of deaths in League of Legends | https://dignitas.gg/articles/an-in-depth-analysis-of-deaths-in-league-of-legends |

## 2. 야구 계량 방법론 (지표 구조의 원형)

> ⚠️ 작성 시 직접 열어 확인하지 않은 링크다. 공개 전 유효성과 내용 일치를 확인할 것.

| 근거로 쓴 곳 | 개념 | 링크 |
|---|---|---|
| G·ΔG 구조 (상태 기대값의 차분) | RE24 | https://library.fangraphs.com/misc/re24/ |
| LwRC+ 구조 (평균 100 기준 비율 지표) | wRC / wRC+ | https://library.fangraphs.com/offense/wrc/ |
| LoL WAR 구조 | WAR | https://library.fangraphs.com/misc/war/ |
| 대체 수준 (승률 약 .294) | Replacement Level | https://library.fangraphs.com/misc/war/replacement-level/ |

## 3. RORR 웹·디자인 (표시 항목 참고용 — 로직은 참고하지 않음)

| 용도 | 링크 |
|---|---|
| 서비스 LwRC+ 페이지 | https://rorr.club/ko/wrc/ |
| 웹 저장소 (랭킹 페이지 표시 항목 추출 출처 — 데이터는 목업) | https://github.com/zzunlsj/rorr-web |
| 블로그 ③ What is LwRC+? | https://zzunlsj.github.io/rorr-web/#blog/what-is-wrc |
| 블로그 ⑥ How LwRC+ is calculated | https://zzunlsj.github.io/rorr-web/#blog/how-lwrc-is-calculated |
| 블로그 ③ 디자인 (Figma) | https://www.figma.com/design/MuFIQC9A0gBBviYiE1pf5n/RORR-Design-v2.0?node-id=5676-10699 |
| 블로그 ⑥ 디자인 (Figma) | https://www.figma.com/design/MuFIQC9A0gBBviYiE1pf5n/RORR-Design-v2.0?node-id=6298-9352 |

## 4. 내부 문서 (Notion)

**정본 원문**: https://app.notion.com/p/40abc2f961aa811c996df9e84f1f9820

**함께 볼 것**

| 문서 | 링크 |
|---|---|
| LoL WAR 초안 (LwRC+ 상위 지표) | https://app.notion.com/p/40abc2f961aa8104ab25f0abb7814b98 |
| LwRC+ 설계 공유본 — 외부 연구 검토용 | https://app.notion.com/p/402bc2f961aa81d0a768c9dc5d8c5f47 |
| 실질 전투력 지수 (team_power를 G에 쓰지 않는 근거) | https://app.notion.com/p/39ebc2f961aa80c0bf96d5d8e26dab25 |
| 존·반경 근거 (mapzones) | https://app.notion.com/p/3a6bc2f961aa8183bb9ad3da6ca10500 |

**실측 근거 (3게임 프로토타입)**

| 문서 | 링크 |
|---|---|
| 클러스터별 ΔG 실측 내역 + 씨앗 필터 버그 | https://app.notion.com/p/3f3bc2f961aa81518277c580bbdc44de |
| 클러스터 ΔG에서 선수별 wGE까지 | https://app.notion.com/p/3f3bc2f961aa81d9aeddd5a7839eb24c |
| 실제 획득 골드 vs wGE 대조 | https://app.notion.com/p/3f3bc2f961aa81ecb716e4ae575cc622 |
| v0 형상 검증 스모크 (재현 스크립트 첨부) | https://app.notion.com/p/3babc2f961aa819c83decb7da3ce34fa |

**경위 (이 저장소 사양으로 대체됨 — 근거가 필요할 때만)**

| 문서 | 링크 |
|---|---|
| 개발자 설계 페이지 (실 DB 스키마) | https://app.notion.com/p/3dcbc2f961aa80c6bcb1e8573a69c714 |
| 10-08 Feedback — 첫 버전 구현 사양 | https://app.notion.com/p/3f3bc2f961aa819f9cf1cb0e765ea240 |
| 10-20 Feedback — 시간 정규화와 고정 지평 | https://app.notion.com/p/3ffbc2f961aa81fcb25df2efa9796170 |
| 구 v1.0 통합본 (prior G 기준) | https://app.notion.com/p/3ffbc2f961aa817cba6fcc03436d2bb1 |
