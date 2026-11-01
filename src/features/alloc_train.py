"""Step 4 — 배분 β 크로스피팅 학습(관여 회귀) + prior 가중 (spec/04 §4.2·§4.4, 검증 spec/07 #8).

클러스터(팀) 단위로 학습하고 개인 단위로 평가한다.
  Swing_c = Σ_k β_k · (우리팀 Σ x_ik − 상대팀 Σ x_jk)_c + γ · 통제항_c + ε,   β ≥ 0 (NNLS + 리지)
  Swing   = 잔여_us − 잔여_opp — 속도형 잔여(ΔV − Risk − 시간항 − 결정적 감쇠, 결정 27·28)
  통제항  = 교전 직전 G 차이 · 게임 시각 · 클러스터 길이 · 참가 인원(presence > 0 선수 수) + 절편
  w_i     = Σ_k β_k · x_ik   (게임의 fold 모델 — G와 같은 경기 단위 fold)

추정: 피처를 표준화하고, 통제항은 β에 제약이 없으므로 미리 투영해 뺀다(Frisch–Waugh). 그 위에서
      [X; √α·I] β ≈ [y; 0], β ≥ 0 을 NNLS로 푼다. α는 경기 단위 내부 CV.
prior: w_i = Σ_k prior_k · share_ik (share = 팀 내 점유율). 팀 prior 합이 0인데 팀 킬이 있으면 killer 1.0 / assist 0.5.
prior 수축은 하지 않는다 (`allocation.shrinkage_k = 0`, 결정 29). 0이 아니면 멈춘다 — 단위가 다른 두 β를 섞는 식은 정의되지 않음.

결과: data/l2/alloc/model/{beta.parquet, bootstrap.parquet, validation.json}, data/l2/alloc/weights/<...>/<game_id>.parquet
"""

import argparse
import json
import time

import numpy as np
import polars as pl
import yaml
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

from common.paths import L1, L2
from features.alloc_features import FEATURES, OUT as FEAT_DIR
from ingest.leaguepedia import LWRC_YAML

TL = L2 / "timeline"
MODEL_DIR = L2 / "alloc" / "model"
W_DIR = L2 / "alloc" / "weights"
CONTROLS = ["g_diff_pre", "t_min", "len_s", "n_present"]


def load_design() -> pl.DataFrame:
    """클러스터 1행(우리 = 100 관점): Swing, 팀 피처 차이, 통제항, match_id."""
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=False)
    team = feat.group_by("game_id", "k", "team_id").agg(*[pl.col(f).sum() for f in FEATURES],
                                                        (pl.col("presence") > 0).sum().alias("n_present"))
    a = team.filter(pl.col("team_id") == 100).drop("team_id")
    b = team.filter(pl.col("team_id") == 200).drop("team_id")
    diff = a.join(b, on=["game_id", "k"], suffix="_o").select(
        "game_id", "k", *[(pl.col(f) - pl.col(f"{f}_o")).alias(f) for f in FEATURES],
        (pl.col("n_present") + pl.col("n_present_o")).alias("n_present"))
    cl = pl.read_parquet(TL / "clusters" / "**" / "*.parquet", hive_partitioning=True)
    sw = (cl.filter(pl.col("side") == 100).select("game_id", "k", "t_s_ms", "len_s", "tournament", pl.col("resid").alias("r1"))
          .join(cl.filter(pl.col("side") == 200).select("game_id", "k", pl.col("resid").alias("r2")), on=["game_id", "k"])
          .with_columns((pl.col("r1") - pl.col("r2")).alias("swing"), (pl.col("t_s_ms") / 60_000).alias("t_min")))
    g = pl.read_parquet(L2 / "g" / "**" / "*.parquet", hive_partitioning=False).select("game_id", "side", "t_ms", "G")
    gd = (g.filter(pl.col("side") == 100).join(g.filter(pl.col("side") == 200), on=["game_id", "t_ms"], suffix="_o")
          .select("game_id", pl.col("t_ms").alias("t_s_ms"), (pl.col("G") - pl.col("G_o")).alias("g_diff_pre")))
    games = pl.read_parquet(L1 / "games" / "**" / "*.parquet").select("game_id", "match_id")
    folds = pl.read_parquet(L2 / "g_model" / "folds.parquet")
    d = (sw.join(gd, on=["game_id", "t_s_ms"], how="left").join(diff, on=["game_id", "k"])
         .join(games, on="game_id").join(folds, on="match_id"))
    assert d["g_diff_pre"].null_count() == 0
    return d.sort("game_id", "k")


class NNLSRidge:
    """표준화 피처 + 통제항 투영 + 비음수 리지. coef_는 원래 단위."""

    def fit(self, x: np.ndarray, c: np.ndarray, y: np.ndarray, alpha: float) -> "NNLSRidge":
        c1 = np.hstack([c, np.ones((c.shape[0], 1))])
        self.sd = x.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        z = x / self.sd
        proj = lambda m: m - c1 @ np.linalg.lstsq(c1, m, rcond=None)[0]
        zr, yr = proj(z), proj(y)
        p = z.shape[1]
        b, _ = nnls(np.vstack([zr, np.sqrt(alpha) * np.eye(p)]), np.concatenate([yr, np.zeros(p)]), maxiter=50 * p)
        self.coef_ = b / self.sd
        self.gamma_ = np.linalg.lstsq(c1, y - x @ self.coef_, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return x @ self.coef_ + np.hstack([c, np.ones((c.shape[0], 1))]) @ self.gamma_


def r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def choose_alpha(x, c, y, groups, k, alphas) -> float:
    scores = {}
    for a in alphas:
        err = []
        for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
            m = NNLSRidge().fit(x[tr], c[tr], y[tr], a)
            err.append(np.mean((y[te] - m.predict(x[te], c[te])) ** 2))
        scores[a] = float(np.mean(err))
    return min(scores, key=scores.get)


def prior_weights(feat: pl.DataFrame, prior: dict[str, float]) -> pl.Series:
    share = [pl.when(pl.col(f).sum().over("game_id", "k", "team_id") > 0)
             .then(pl.col(f) / pl.col(f).sum().over("game_id", "k", "team_id")).otherwise(0.0) * w
             for f, w in prior.items() if w]
    w = feat.select(pl.sum_horizontal(share).alias("w"), "game_id", "k", "team_id", "kill_killer", "assist")
    fallback = (pl.col("w").sum().over("game_id", "k", "team_id") == 0) & \
               ((pl.col("kill_killer") + pl.col("assist")).sum().over("game_id", "k", "team_id") > 0)
    return w.select(pl.when(fallback).then(1.0 * pl.col("kill_killer") + 0.5 * pl.col("assist"))
                    .otherwise(pl.col("w")).alias("w_prior"))["w_prior"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(LWRC_YAML.read_text())
    acfg, k = cfg["allocation"], cfg["g"]["crossfit_k"]
    if acfg["shrinkage_k"] != 0:
        raise ValueError("allocation.shrinkage_k ≠ 0: prior 수축 방식이 정의되지 않음 (결정 29)")
    t0 = time.time()
    d = load_design()
    x, c, y = d.select(FEATURES).to_numpy(), d.select(CONTROLS).to_numpy(), d["swing"].to_numpy()
    groups, fold = d["match_id"].to_numpy(), d["fold"].to_numpy()

    alpha_all = choose_alpha(x, c, y, groups, k, acfg["ridge_alphas"])
    full = NNLSRidge().fit(x, c, y, alpha_all)
    rows = [{"fold": "all", "alpha": alpha_all, **dict(zip(FEATURES, full.coef_))}]
    oos = np.empty_like(y)
    models = {}
    for f in range(k):
        tr, te = fold != f, fold == f
        a = choose_alpha(x[tr], c[tr], y[tr], groups[tr], k, acfg["ridge_alphas"])
        models[f] = NNLSRidge().fit(x[tr], c[tr], y[tr], a)
        oos[te] = models[f].predict(x[te], c[te])
        rows.append({"fold": str(f), "alpha": a, **dict(zip(FEATURES, models[f].coef_)),
                     "oos_r2": r2(y[te], oos[te])})
    beta = pl.DataFrame(rows)

    # 통제항만의 설명력 (β가 더하는 몫 비교용)
    c1 = np.hstack([c, np.ones((c.shape[0], 1))])
    oos_ctrl = np.empty_like(y)
    for f in range(k):
        tr, te = fold != f, fold == f
        oos_ctrl[te] = c1[te] @ np.linalg.lstsq(c1[tr], y[tr], rcond=None)[0]

    # 부트스트랩 (경기 단위 재표집, α 고정)
    rng = np.random.default_rng(args.seed)
    matches = np.unique(groups)
    idx_of = {m: np.where(groups == m)[0] for m in matches}
    boot = []
    for _ in range(acfg["bootstrap_reps"]):
        idx = np.concatenate([idx_of[m] for m in rng.choice(matches, matches.size, replace=True)])
        boot.append(NNLSRidge().fit(x[idx], c[idx], y[idx], alpha_all).coef_)
    boot = np.array(boot)
    # 비율만 의미가 있으므로 dmg 기준 상대 β도 함께
    ci = [{"feature": f, "beta": float(full.coef_[i]), "lo": float(np.percentile(boot[:, i], 2.5)),
           "hi": float(np.percentile(boot[:, i], 97.5)), "p_positive": float((boot[:, i] > 0).mean()),
           "fold_min": float(beta.filter(pl.col("fold") != "all")[f].min()),
           "fold_max": float(beta.filter(pl.col("fold") != "all")[f].max()),
           "sd_x_player": None} for i, f in enumerate(FEATURES)]

    # 선수 가중치 (회귀 = 게임의 fold β, prior = 팀 점유율)
    feat = pl.read_parquet(FEAT_DIR / "**" / "*.parquet", hive_partitioning=True).sort("game_id", "k", "participant_id")
    sd_player = feat.select(FEATURES).std()
    for r in ci:
        r["sd_x_player"] = float(sd_player[r["feature"]][0])
        r["beta_x_sd"] = r["beta"] * r["sd_x_player"]       # 선수 피처 1 표준편차당 지분 — 피처 간 비교용
    fold_of_game = dict(d.select("game_id", "fold").unique().iter_rows())
    bmat = {f: m.coef_ for f, m in models.items()}
    xf = feat.select(FEATURES).to_numpy()
    gf = np.array([fold_of_game[g] for g in feat["game_id"].to_list()])
    w_reg = np.zeros(feat.height)
    for f, b in bmat.items():
        w_reg[gf == f] = xf[gf == f] @ b
    weights = feat.select("game_id", "tournament", "stage", "k", "participant_id", "team_id").with_columns(
        pl.Series("w_reg", w_reg), prior_weights(feat, acfg["prior_weights"]).alias("w_prior"))
    for (t, s), part in weights.partition_by(["tournament", "stage"], as_dict=True).items():
        for (gid,), gpart in part.partition_by(["game_id"], as_dict=True).items():
            path = W_DIR / f"tournament={t}" / f"stage={s}" / f"{gid}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            gpart.drop("tournament", "stage").write_parquet(path, compression="zstd")

    team_zero = weights.group_by("game_id", "k", "team_id").agg(pl.col("w_reg").sum().alias("r"), pl.col("w_prior").sum().alias("p"))
    val = {
        "n_clusters": int(d.height), "n_games": int(d["game_id"].n_unique()), "alpha_all": alpha_all,
        "oos_r2_crossfit": r2(y, oos), "oos_r2_controls_only": r2(y, oos_ctrl),
        "oos_r2_by_fold": {r["fold"]: r["oos_r2"] for r in rows[1:]},
        "beta_ci": ci, "gamma_all": dict(zip(CONTROLS + ["intercept"], map(float, full.gamma_))),
        "team_sum_w_zero": {"regression": float((team_zero["r"] == 0).mean()), "prior": float((team_zero["p"] == 0).mean())},
        "shrinkage_k": acfg["shrinkage_k"], "bootstrap_reps": acfg["bootstrap_reps"],
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    beta.write_parquet(MODEL_DIR / "beta.parquet")
    pl.DataFrame(boot, schema=FEATURES, orient="row").write_parquet(MODEL_DIR / "bootstrap.parquet")
    (MODEL_DIR / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1))
    print(f"클러스터 {d.height}개, α={alpha_all}, OOS R² {val['oos_r2_crossfit']:.3f} (통제항만 {val['oos_r2_controls_only']:.3f}), "
          f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
