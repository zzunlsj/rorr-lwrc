"""게임 시야 점수(VISION_SCORE)가 언제·얼마나 오르나 — 설치·제거 점수 실측 (2026-11-01 Jun 질문, 표본 40게임).

선수별 누적 시야 점수의 증가(1Hz 틱 간 차이)를 같은 선수의 와드 이벤트와 시각으로 맞춘다 (±1.5초 안 가장 가까운 이벤트):
  place = 본인 와드 설치 · own_end = 본인 와드 소멸(제거·만료·교체) · kill = 본인의 와드 제거 · none = 맞는 이벤트 없음
L0는 읽기만 한다. 결과: 스크래치 parquet (리포트 specs/support_vision_check_2026-11-01.md §12)
사용: PYTHONPATH=src uv run python -m report.vision_score_jumps
"""
import csv, numpy as np, polars as pl, random
from multiprocessing import Pool
from ingest.raw_reader import iter_events
from common.paths import MANIFEST, L1, L2
rec={r["game_id"]:r for r in csv.DictReader(open(MANIFEST/"received.csv")) if not r.get("superseded_by")}
g=pl.read_parquet(L1/"games"/"**"/"*.parquet",hive_partitioning=True).filter(~pl.col("excluded")).select("game_id","tournament","stage")
random.seed(0); sample=random.sample(g.rows(),40)
def one(row):
    gid,t_,s_=row
    T,V=[],[]
    for sch,_,p in iter_events(rec[gid]):
        if sch!="stats_update" or p.get("gameTime") is None: continue
        r=np.full(10,np.nan)
        for pp in p.get("participants",[]):
            for s in pp.get("stats",[]):
                if s.get("name")=="VISION_SCORE": r[pp["participantID"]-1]=s["value"]
        T.append(p["gameTime"]);V.append(r)
    o=np.argsort(T);T=np.array(T)[o];V=np.fmax.accumulate(np.nan_to_num(np.array(V)[o],nan=0),axis=0)
    we=pl.read_parquet(L1/"ward_events"/f"tournament={t_}"/f"stage={s_}"/f"{gid}.parquet")
    wk=pl.read_parquet(L1/"ward_kills"/f"tournament={t_}"/f"stage={s_}"/f"{gid}.parquet")
    rows=[]
    for pid in range(1,11):
        d=np.diff(V[:,pid-1]); idx=np.where(d>1e-9)[0]
        jt=T[idx+1]; jv=d[idx]
        pl_=we.filter(pl.col("placer")==pid); kl=wk.filter(pl.col("killer")==pid)
        ev=[("place",x,None) for x in pl_["t_placed_ms"].to_list()]+[("own_end",x,(a,b)) for x,a,b in zip(pl_["t_removed_ms"].to_list(),pl_["ward_type"].to_list(),pl_["t_placed_ms"].to_list()) if x is not None]+[("kill",x,w) for x,w in zip(kl["game_time_ms"].to_list(),kl["ward_type"].to_list())]
        for t,v in zip(jt,jv):
            best=min(ev,key=lambda e:abs(e[1]-t)) if ev else None
            dt=abs(best[1]-t) if best else 1e9
            kind=best[0] if best and dt<=1500 else "none"
            extra={}
            if kind=="own_end": extra={"wtype":best[2][0],"life_s":(best[1]-best[2][1])/1000}
            if kind=="kill": extra={"wtype":best[2]}
            rows.append({"game_id":gid,"pid":pid,"t":int(t),"jump":float(v),"kind":kind,**extra})
        rows.append({"game_id":gid,"pid":pid,"t":-1,"jump":float(V[-1,pid-1]),"kind":"TOTAL"})
    return rows
if __name__ == "__main__":
    with Pool(6) as pool: R=[r for x in pool.map(one,sample) for r in x]
    df=pl.DataFrame(R,infer_schema_length=None)
    df.write_parquet(str(L2 / "vision_score" / "vs_jumps_sample40.parquet"))
    tot=df.filter(pl.col("kind")=="TOTAL")["jump"].sum(); j=df.filter(pl.col("kind")!="TOTAL")
    print("총 시야점수",round(tot),"점프합",round(j["jump"].sum()), "점프 수", j.height)
    print(j.group_by("kind").agg(pl.len(),pl.col("jump").sum().alias("sum"),pl.col("jump").mean().alias("mean"),pl.col("jump").median().alias("med")).with_columns((100*pl.col("sum")/tot).round(1).alias("pct")).sort("sum",descending=True))
    print(j.filter(pl.col("kind")=="own_end").group_by("wtype").agg(pl.len(),pl.col("jump").mean(),pl.col("life_s").mean(),(pl.col("jump")/(pl.col("life_s")/60)).median().alias("pts_per_min")))
    print(j.filter(pl.col("kind")=="kill").group_by("wtype").agg(pl.len().alias("n"),pl.col("jump").mean().alias("mean"),pl.col("jump").median().alias("med")))
