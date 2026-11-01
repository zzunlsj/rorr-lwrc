"""Step 2 — 학습 G: 선형 + 리지, 경기 단위 크로스피팅 (spec/01 §1.3, spec/07 검증 1–3).

- 학습 표본: 상태 테이블에서 g.train_sample_sec 간격 (게임·진영마다 구간의 첫 틱). 산출은 1Hz 전체.
- 피처: 상태 32개 + 패치·대회 더미 (게임 중 불변 → ΔG에는 영향 없음).
- 리지 α: 경기 단위 내부 GroupKFold로 선택.
- 크로스피팅: 경기 단위 K-fold. 각 게임은 자기 경기를 뺀 fold 모델로 G를 산출.
- 검증: 계수 부호(하나라도 어긋나면 반려), 시간 홀드아웃 R²(앞 75% 기간 학습 / 뒤 25% 평가)와 시간대별 잔차, fold별 계수 변동.

결과: data/l2/g_model/ (coefficients, folds, validation.json, price_table.csv), data/l2/g/<...>/<game_id>.parquet (1Hz G)
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from common.paths import L1, L2
from features.state_table import FEATURES, OUT as STATE_DIR
from ingest.leaguepedia import LWRC_YAML

MODEL_DIR = L2 / "g_model"
G_DIR = L2 / "g"
SPEC_VERSION = "lwrc_v1.0-rev"
TIME_SEGMENTS = ((0, 15), (15, 25), (25, 999))


def game_meta() -> pl.DataFrame:
    return (pl.scan_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True)
            .filter(~pl.col("excluded"))
            .select("game_id", "match_id", "tournament", "patch", "game_start_utc").collect())


def load_sample(sample_sec: int) -> pl.DataFrame:
    s = pl.scan_parquet(STATE_DIR / "**" / "*.parquet", hive_partitioning=False)
    return (s.with_columns((pl.col("t_ms") // (sample_sec * 1000)).alias("bucket"))
            .group_by("game_id", "side", "bucket").agg(pl.all().sort_by("t_ms").first())
            .drop("bucket").collect())


class Design:
    """피처 행렬 구성: 32개 + 패치·대회 더미. 학습 때 본 범주 목록을 고정해 산출에도 같은 열을 쓴다."""

    def __init__(self, patches: list[str], tournaments: list[str]):
        self.patches, self.tournaments = sorted(patches), sorted(tournaments)
        self.columns = FEATURES + [f"patch={p}" for p in self.patches] + [f"tournament={t}" for t in self.tournaments]

    def matrix(self, df: pl.DataFrame) -> np.ndarray:
        x = df.select(FEATURES).to_numpy()
        pd = np.stack([(df["patch"] == p).to_numpy() for p in self.patches], axis=1).astype(float)
        td = np.stack([(df["tournament"] == t).to_numpy() for t in self.tournaments], axis=1).astype(float)
        return np.hstack([x, pd, td])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """표준화 후 리지 → 원래 단위 계수(β = 단가)와 절편으로 되돌린다."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    m = Ridge(alpha=alpha).fit((x - mu) / sd, y)
    beta = m.coef_ / sd
    return beta, float(m.intercept_ - (beta * mu).sum())


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, alphas) -> float:
    best, best_mse = None, np.inf
    for a in alphas:
        errs = []
        for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
            beta, b0 = fit_ridge(x[tr], y[tr], a)
            errs.append(np.mean((y[te] - (x[te] @ beta + b0)) ** 2))
        if np.mean(errs) < best_mse:
            best, best_mse = a, float(np.mean(errs))
    return best


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def segment_report(df: pl.DataFrame, yhat: np.ndarray) -> list[dict]:
    """시간대별 R²·평균 잔차. 'income'은 y − 현재 골드(앞으로 10분 수입) 기준 R² — 현재 골드만으로 높아지는 R²를 걸러 본다."""
    out = []
    y, gold, tmin = df["y"].to_numpy(), df["gold_us"].to_numpy(), df["t_min"].to_numpy()
    for lo, hi in ((0, 999),) + TIME_SEGMENTS:
        m = (tmin >= lo) & (tmin < hi)
        if m.sum() < 10:
            continue
        out.append({"segment": f"{lo}-{hi if hi < 999 else ''}분", "n": int(m.sum()),
                    "r2": r2(y[m], yhat[m]), "r2_income": r2(y[m] - gold[m], yhat[m] - gold[m]),
                    "mean_resid": float((y[m] - yhat[m]).mean()), "rmse": float(np.sqrt(((y[m] - yhat[m]) ** 2).mean()))})
    return out


def sign_check(beta: dict[str, float], rules: dict[str, int]) -> list[dict]:
    """spec/07 검증 1 — config g.sign_rules (결정 23: 타워는 외곽·2차만)."""
    return [{"feature": f, "expected": "+" if s > 0 else "−", "beta": beta[f], "ok": bool(np.sign(beta[f]) == s)}
            for f, s in rules.items()]


def price_table(c: dict[str, float], buffs: dict, respawn_examples: list[float]) -> list[dict]:
    """단가표 — 상태 전이 1건의 순간 ΔG (다른 피처 고정). 우리 G와 상대 G(관점 반전 계수)를 함께 적는다."""
    def row(event, us, opp, note=""):
        return {"event": event, "dG_us": us, "dG_opp": opp, "note": note}
    rows = [
        row("골드 1,000 획득", 1000 * c["gold_us"], 1000 * c["gold_opp"]),
        row("레벨 합 +1", c["level_diff"], -c["level_diff"]),
        row("외곽 타워 파괴", c["tower_outer_taken"], c["tower_outer_lost"]),
        row("2차 타워 파괴", c["tower_inner_taken"], c["tower_inner_lost"]),
        row("억제기 타워 파괴", c["tower_base_taken"], c["tower_base_lost"], "결정 23: 같은 골드 조건에서 음수"),
        row("억제기 파괴", c["inhib_down_opp"], c["inhib_down_us"], "리스폰되면 반대로 돌아감"),
        row("넥서스 타워 파괴", c["nexus_tower_down_opp"], c["nexus_tower_down_us"]),
        row("드래곤 (장로 제외, 영혼 전)", c["dragon_stack_us"], c["dragon_stack_opp"]),
        row("드래곤 영혼 (4번째 드래곤)", c["dragon_stack_us"] + c["soul_us"], c["dragon_stack_opp"] + c["soul_opp"]),
        row("바론", buffs["baron_sec"] * c["baron_buff_us"], buffs["baron_sec"] * c["baron_buff_opp"], "처치 순간 버프 남은 초 × 단가, 이후 시간에 따라 줄어듦"),
        row("장로 드래곤", buffs["elder_sec"] * c["elder_buff_us"], buffs["elder_sec"] * c["elder_buff_opp"], "버프분만 (스택 아님)"),
        row("공허 유충 1", c["voidgrub_us"], c["voidgrub_opp"]),
        row("전령·아타칸 첫 처치", c["herald_or_atakhan_us"], c["herald_or_atakhan_opp"]),
    ]
    for r_s in respawn_examples:
        rows.append(row(f"아군 1명 데스 (부활 {r_s:g}초)", -c["alive_us"] + r_s * c["respawn_sum_us"],
                        -c["alive_opp"] + r_s * c["respawn_sum_opp"], "부활 대기는 이후 매초 줄어듦"))
    return rows


def predict_game(row: dict, design_cols: list[str], betas: dict[int, tuple], patches, tournaments) -> int:
    """1Hz G 산출 — 해당 경기 fold 모델 사용."""
    st = pl.read_parquet(STATE_DIR / f"tournament={row['tournament']}" / "**" / f"{row['game_id']}.parquet",
                         hive_partitioning=False)
    st = st.with_columns(pl.lit(row["patch"]).alias("patch"), pl.lit(row["tournament"]).alias("tournament"))
    d = Design(patches, tournaments)
    beta, b0 = betas[row["fold"]]
    x = d.matrix(st)
    contrib_time = x[:, 0] * beta[0] + x[:, 1] * beta[1]   # 시간 항 (spec/01 §1.4) — Step 3에서 팀 계정으로
    out = st.select("game_id", "side", "t_ms", "y", "is_end").with_columns(
        pl.Series("G", x @ beta + b0), pl.Series("G_time_term", contrib_time), pl.lit(row["fold"]).alias("fold"))
    path = G_DIR / f"tournament={row['tournament']}" / f"{row['game_id']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path, compression="zstd")
    return out.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    full_cfg = yaml.safe_load(LWRC_YAML.read_text())
    cfg = full_cfg["g"]
    alphas = cfg["ridge_alphas"]
    t0 = time.time()
    meta = game_meta()
    data = load_sample(cfg["train_sample_sec"]).join(meta, on=["game_id", "match_id"])
    design = Design(meta["patch"].unique().to_list(), meta["tournament"].unique().to_list())
    x, y = design.matrix(data), data["y"].to_numpy()
    groups = data["match_id"].to_numpy()
    k = cfg["crossfit_k"]
    print(f"학습 표본 {data.height}행 × {x.shape[1]}열 ({len(set(groups))}경기)", flush=True)

    # 1) 전체 적합 — 단가표·부호 검증
    alpha_all = choose_alpha(x, y, groups, k, alphas)
    beta_all, b0_all = fit_ridge(x, y, alpha_all)
    coef_all = dict(zip(design.columns, beta_all))

    # 2) 크로스피팅 (경기 단위 K-fold)
    match_ids = np.array(sorted(set(groups)))
    fold_of_match = {}
    for f, (_, te) in enumerate(GroupKFold(n_splits=k).split(match_ids, groups=match_ids)):
        fold_of_match.update({m: f for m in match_ids[te]})
    fold = np.array([fold_of_match[m] for m in groups])
    betas, oof = {}, np.empty_like(y)
    coef_rows = [{"fold": "all", "alpha": alpha_all, "intercept": b0_all, **coef_all}]
    for f in range(k):
        tr = fold != f
        a = choose_alpha(x[tr], y[tr], groups[tr], k, alphas)
        beta, b0 = fit_ridge(x[tr], y[tr], a)
        betas[f] = (beta, b0)
        oof[~tr] = x[~tr] @ beta + b0
        coef_rows.append({"fold": str(f), "alpha": a, "intercept": b0, **dict(zip(design.columns, beta))})
    coefs = pl.DataFrame(coef_rows)

    # 3) 시간 홀드아웃: 경기 시작 시각 기준 앞 75% 경기 학습 / 뒤 25% 평가
    match_start = meta.group_by("match_id").agg(pl.col("game_start_utc").min()).sort("game_start_utc")
    cut = int(match_start.height * (1 - cfg["holdout_tail_fraction"]))
    train_m = set(match_start["match_id"][:cut])
    tr = np.array([m in train_m for m in groups])
    a_h = choose_alpha(x[tr], y[tr], groups[tr], k, alphas)
    beta_h, b0_h = fit_ridge(x[tr], y[tr], a_h)
    hold = data.filter(pl.Series(~tr))
    holdout = {"alpha": a_h, "train_matches": cut, "test_matches": match_start.height - cut,
               "test_from_utc": str(match_start["game_start_utc"][cut]),
               "test_tournaments": sorted(hold["tournament"].unique().to_list()),
               "segments": segment_report(hold, x[~tr] @ beta_h + b0_h)}

    # 4) 검증 묶음
    fold_coef = coefs.filter(pl.col("fold") != "all").select(FEATURES)
    stability = [{"feature": f, "all": coef_all[f], "fold_mean": float(fold_coef[f].mean()),
                  "fold_min": float(fold_coef[f].min()), "fold_max": float(fold_coef[f].max()),
                  "fold_cv": float(fold_coef[f].std() / abs(fold_coef[f].mean())) if fold_coef[f].mean() else None}
                 for f in FEATURES]
    signs = sign_check(coef_all, cfg["sign_rules"])
    fold_signs = {str(f): all(r["ok"] for r in sign_check(dict(zip(design.columns, betas[f][0])), cfg["sign_rules"]))
                  for f in range(k)}
    validation = {
        "spec_version": SPEC_VERSION, "n_rows": data.height, "n_matches": len(match_ids), "n_games": meta.height,
        "sample_sec": cfg["train_sample_sec"], "horizon_min": cfg["label_horizon_min"], "k": k,
        "signs_all": signs, "signs_ok_all": all(r["ok"] for r in signs), "signs_ok_by_fold": fold_signs,
        "crossfit_oof": segment_report(data, oof), "time_holdout": holdout, "stability": stability,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    coefs.write_parquet(MODEL_DIR / "coefficients.parquet")
    pl.DataFrame({"match_id": list(fold_of_match), "fold": list(fold_of_match.values())}).write_parquet(MODEL_DIR / "folds.parquet")
    (MODEL_DIR / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=1, default=float))
    pl.DataFrame(price_table(coef_all, full_cfg["buffs"], cfg["price_examples"]["death_respawn_sec"])).with_columns(
        pl.col("dG_us", "dG_opp").round(1), pl.lit(SPEC_VERSION).alias("spec_version")).write_csv(MODEL_DIR / "price_table.csv")
    (MODEL_DIR / "design.json").write_text(json.dumps({"columns": design.columns, "patches": design.patches,
                                                        "tournaments": design.tournaments}, ensure_ascii=False))
    print(f"학습·검증 {time.time() - t0:.0f}초, 부호 {'통과' if validation['signs_ok_all'] else '반려'}", flush=True)

    # 5) 1Hz G 산출 (크로스피팅 fold 모델)
    rows = meta.with_columns(pl.col("match_id").replace_strict(fold_of_match).alias("fold")).to_dicts()
    job = partial(predict_game, design_cols=design.columns, betas=betas,
                  patches=design.patches, tournaments=design.tournaments)
    with Pool(args.workers) as pool:
        n = sum(pool.imap_unordered(job, rows, chunksize=4))
    print(f"G 산출 {n}행, 총 {time.time() - t0:.0f}초 → {G_DIR}")


if __name__ == "__main__":
    main()
