"""ΔG 귀속 엔진 — 채널 C / D / Risk / 팀 버킷 / D_unattributed.

두 배분 방식을 같은 회계 골격 위에서 갈아끼운다:
  coldstart_v01 — 콜드스타트 v0.1 (딜 비중 → 균등 → 팀 버킷, Risk = 클러스터 내 데스 균등)
  spec_v10      — 정본 v1.0 §4 (4성분 지분, Risk 시간창, D 규칙)

회계 불변식 (자팀 장부): Σ C + Σ Risk + 팀 버킷 = Σ ΔG_us = G_us(끝) − G_us(시작).
D와 D_unattributed는 상대 장부라 이 식에 넣지 않는다.
"""

from collections import defaultdict

import numpy as np

from features.clusters import DRIFT, Cluster
from features.state import GameState, opp

TEAM = -1  # 팀 계정 행의 participant_id


# ── 지분(weight) ─────────────────────────────────────────────
def _credited(cluster: Cluster, side: int):
    return [s for s in cluster.seeds if s.credited_team == side]


def _dmg_share(state: GameState, cluster: Cluster, side: int, idx: np.ndarray) -> np.ndarray:
    """클러스터 안 우리 팀 킬들의 deathRecap(Champion) 딜을 caster별로 누적."""
    kill_seqs = {s.kill_seq for s in _credited(cluster, side) if s.kind == "kill"}
    out = np.zeros(idx.size)
    if not kill_seqs:
        return out
    pos = {int(state.pids[j]): n for n, j in enumerate(idx)}
    kd = state.kill_damage
    for seq, caster, dmg in zip(kd["kill_seq"].to_list(), kd["caster_id"].to_list(), kd["dmg"].to_list()):
        if seq in kill_seqs and caster in pos:
            out[pos[caster]] += dmg
    return out


def _participation(cluster: Cluster, side: int, idx: np.ndarray, pids: np.ndarray,
                   killer_w: float, assist_w: float) -> np.ndarray:
    pos = {int(pids[j]): n for n, j in enumerate(idx)}
    out = np.zeros(idx.size)
    for s in _credited(cluster, side):
        if s.killer in pos:
            out[pos[s.killer]] += killer_w
        for a in s.assistants:
            if a in pos:
                out[pos[a]] += assist_w
    return out


def _share(x: np.ndarray) -> np.ndarray:
    tot = x.sum()
    return x / tot if tot > 0 else np.zeros_like(x)


class ColdStartV01:
    """딜 비중 → (딜 없음) killer+assistants 균등 → (없음) 0 = 팀 버킷."""

    def __init__(self, **_):
        pass

    def weights(self, state, cluster, side, idx, mask) -> np.ndarray:
        dmg = _dmg_share(state, cluster, side, idx)
        if dmg.sum() > 0:
            return dmg
        part = _participation(cluster, side, idx, state.pids, 1.0, 1.0)
        return (part > 0).astype(float)

    def risk(self, state, cluster, side, idx, mask, dG, pie_neg):
        """클러스터 내 본팀 데스(victim) 균등. 반환: (선수별 Risk, 팀 버킷 잔여)."""
        pos = {int(state.pids[j]): n for n, j in enumerate(idx)}
        deaths = [s.victim for s in cluster.seeds if s.kind == "kill" and s.victim_team == side and s.victim in pos]
        out = np.zeros(idx.size)
        if not deaths:
            return out, pie_neg
        for v in deaths:
            out[pos[v]] += pie_neg / len(deaths)
        return out, 0.0


class SpecV10:
    """정본 v1.0 §4.2 4성분 지분 + §4.4 Risk 시간창."""

    def __init__(self, w_dmg, w_assist, w_absorb, w_presence, presence_radius,
                 fallback_killer, fallback_assist, window_sec, **_):
        self.W = np.array([w_dmg, w_assist, w_absorb, w_presence], dtype=float)
        self.radius = presence_radius
        self.fb_killer, self.fb_assist = fallback_killer, fallback_assist
        self.window_ms = window_sec * 1000

    def components(self, state, cluster, side, idx, mask) -> np.ndarray:
        """(4, 5) — dmg / assist / absorb / presence 각각 팀 점유율."""
        dmg = _dmg_share(state, cluster, side, idx)
        if dmg.sum() <= 0:  # 결측 폴백: killer 1.0 / assist 0.5
            dmg = _participation(cluster, side, idx, state.pids, self.fb_killer, self.fb_assist)
        assist = _participation(cluster, side, idx, state.pids, 0.0, 1.0)

        iv = np.flatnonzero(mask)                 # 구간 i = (t_i, t_{i+1}] (틱 인덱스 기준)
        h = state.p_health[:, idx]
        alive = state.p_alive[:, idx]
        drop = np.maximum(0.0, h[iv] - h[iv + 1]) * alive[iv]
        absorb = drop.sum(axis=0)

        ticks = iv + 1
        present = np.zeros((ticks.size, idx.size), dtype=bool)
        for s in cluster.seeds:
            if s.x is None or s.z is None:
                continue
            near_t = np.abs(state.t[ticks] - s.time_ms) <= self.window_ms
            dist = np.hypot(state.p_x[ticks][:, idx] - s.x, state.p_z[ticks][:, idx] - s.z)
            present |= near_t[:, None] & (dist <= self.radius)
        present &= alive[ticks]
        presence = present.mean(axis=0) if ticks.size else np.zeros(idx.size)

        return np.vstack([_share(dmg), _share(assist), _share(absorb), _share(presence)])

    def weights(self, state, cluster, side, idx, mask) -> np.ndarray:
        return self.W @ self.components(state, cluster, side, idx, mask)

    def risk(self, state, cluster, side, idx, mask, dG, pie_neg):
        """데스 시각부터 respawn_timer 동안 열린 창의 음수 ΔG_us를 그 데스에 귀속.
        창이 겹치면 구간 ΔG를 열린 데스 수로 나눈다. 합이 클러스터 음수 파이를 넘지 않게 맞추고
        나머지는 팀 버킷 (상대의 정상 획득)."""
        pos = {int(state.pids[j]): n for n, j in enumerate(idx)}
        deaths = [s for s in cluster.seeds if s.kind == "kill" and s.victim_team == side and s.victim in pos]
        out = np.zeros(idx.size)
        if not deaths:
            return out, pie_neg
        iv = np.flatnonzero(mask)
        mid = (state.t[iv] + state.t[iv + 1]) / 2.0
        windows = []
        for d in deaths:
            j = idx[pos[d.victim]]
            after = min(np.searchsorted(state.t, d.time_ms, side="left"), state.n - 1)
            respawn_ms = state.p_respawn[after, j] * 1000.0
            windows.append((d.time_ms, d.time_ms + respawn_ms))
        open_ = np.array([(mid > a) & (mid <= b) for a, b in windows])     # (deaths, intervals)
        n_open = open_.sum(axis=0)
        per_iv = np.divide(dG[iv], n_open, out=np.zeros(iv.size), where=n_open > 0)
        raw = (open_ * per_iv).sum(axis=1)                                   # 데스별 순 ΔG
        neg = np.minimum(raw, 0.0)
        s_neg = neg.sum()
        if s_neg >= 0:
            return out, pie_neg
        total = max(pie_neg, s_neg)                                          # 둘 다 음수 — 파이 초과 금지
        for d, v in zip(deaths, neg):
            out[pos[d.victim]] += total * v / s_neg
        return out, pie_neg - total


def allocator_from_config(cfg: dict):
    a = cfg["allocation"]
    if a["method"] == "coldstart_v01":
        return ColdStartV01()
    if a["method"] == "spec_v10":
        return SpecV10(window_sec=cfg["cluster"]["window_sec"], **{k: v for k, v in a.items() if k != "method"})
    raise ValueError(a["method"])


# ── 회계 골격 ────────────────────────────────────────────────
def attribute(state: GameState, G: dict[int, np.ndarray], clusters: list[Cluster],
              bucket: np.ndarray, allocator) -> tuple[list[dict], list[dict]]:
    """반환: (원장 행, 클러스터 진단 행)."""
    ledger: list[dict] = []
    diag: list[dict] = []
    acc: dict[tuple, float] = defaultdict(float)

    for side in G:
        o = opp(side)
        dG, dGo = np.diff(G[side]), np.diff(G[o])
        idx = state.team_idx(side)

        drift = bucket == DRIFT
        realized = np.diff(state.p_gold[:, idx], axis=0)[drift].sum(axis=0)
        for n, j in enumerate(idx):
            acc[(side, int(state.pids[j]), "C", "drift")] += realized[n]
        acc[(side, TEAM, "team_bucket", "drift")] += dG[drift].sum() - realized.sum()

        for c in clusters:
            mask = bucket == c.k
            if not mask.any():
                continue
            g_us, g_opp = dG[mask].sum(), dGo[mask].sum()
            w = allocator.weights(state, c, side, idx, mask)
            has_w = w.sum() > 0
            c_pie, d_pie = max(g_us, 0.0), max(-g_opp, 0.0)
            if c_pie > 0:
                if has_w:
                    for n, j in enumerate(idx):
                        acc[(side, int(state.pids[j]), "C", "cluster")] += c_pie * w[n] / w.sum()
                else:
                    acc[(side, TEAM, "team_bucket", "cluster")] += c_pie
            risk_bucket = 0.0
            if g_us < 0:
                risk, risk_bucket = allocator.risk(state, c, side, idx, mask, dG, g_us)
                for n, j in enumerate(idx):
                    acc[(side, int(state.pids[j]), "Risk", "cluster")] += risk[n]
                acc[(side, TEAM, "team_bucket", "cluster")] += risk_bucket
            if d_pie > 0:
                if has_w:
                    for n, j in enumerate(idx):
                        acc[(side, int(state.pids[j]), "D", "cluster")] += d_pie * w[n] / w.sum()
                else:
                    acc[(side, TEAM, "D_unattributed", "cluster")] += d_pie
            diag.append({
                "side": side, "k": c.k, "start_ms": c.start_ms, "end_ms": c.end_ms,
                "len_s": (c.end_ms - c.start_ms) / 1000, "n_seeds": len(c.seeds), "n_intervals": int(mask.sum()),
                "dG_us": g_us, "dG_opp": g_opp, "sum_w": float(w.sum()), "risk_to_bucket": risk_bucket,
            })

    for (side, pid, channel, source), amount in acc.items():
        ledger.append({"side": side, "participant_id": pid, "channel": channel, "source": source, "amount": amount})
    return ledger, diag
