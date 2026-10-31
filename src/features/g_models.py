"""G(state) = E[팀 최종 골드 | 현재 상태] — 교체 가능한 인터페이스 (CLAUDE.md §8).

모든 모델은 evaluate(state, side) → 틱별 G 배열을 돌려주고,
마지막 틱에서 G = 실제 팀 골드가 되도록 한다 (엔드 콜랩스 → 보존 항등식 우변 정합).
학습 G(v1 선형 / v2 GBM)는 같은 인터페이스로 추가한다.
"""

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression

from features.state import GameState, opp


class GModel(Protocol):
    name: str

    def evaluate(self, state: GameState, side: int) -> np.ndarray: ...


class RmTableG:
    """콜드스타트 v0.1 임시 G: 팀 현재 골드 + R(분), 분 버킷 선형 보간, 최종 틱 R=0."""

    name = "rm_table"

    def __init__(self, r_table: list[float], isotonic: bool = False):
        r = np.asarray(r_table, dtype=float)
        if isotonic:
            minutes = np.arange(r.size)
            r = IsotonicRegression(increasing=False).fit_transform(minutes, r)
        self.r = r

    def residual(self, t_ms: np.ndarray) -> np.ndarray:
        minutes = np.arange(self.r.size)
        return np.interp(t_ms / 60000.0, minutes, self.r)  # 범위 밖은 끝값 유지

    def evaluate(self, state: GameState, side: int) -> np.ndarray:
        r = self.residual(state.t)
        r[-1] = 0.0
        return state.team_gold[side] + r


class PriorV1G:
    """정본 v1.0 §2 임시 G: gold + R_BAR × horizon(t) × M(state), 고정 지평 정본 테이퍼."""

    name = "prior_v1"

    def __init__(self, r_bar: float, h_fixed: float, w_alive: float, w_dragon: float,
                 w_tower: float, w_baron: float, baron_buff_sec: float, **_):
        self.r_bar, self.h_fixed = r_bar, h_fixed
        self.w_alive, self.w_dragon, self.w_tower, self.w_baron = w_alive, w_dragon, w_tower, w_baron
        self.baron_buff_ms = baron_buff_sec * 1000

    def horizon(self, state: GameState) -> np.ndarray:
        return np.minimum(self.h_fixed, (state.t_end - state.t) / 60000.0)

    def multiplier(self, state: GameState, side: int) -> np.ndarray:
        o = opp(side)
        return (1.0
                + self.w_alive * (state.alive_count(side) - state.alive_count(o)) / 5
                + self.w_dragon * (state.dragon_stacks(side) - state.dragon_stacks(o))
                + self.w_tower * (state.towers_lost(o) - state.towers_lost(side))
                + self.w_baron * state.baron_active(side, self.baron_buff_ms))

    def evaluate(self, state: GameState, side: int) -> np.ndarray:
        return state.team_gold[side] + self.r_bar * self.horizon(state) * self.multiplier(state, side)


def from_config(cfg: dict) -> GModel:
    g = cfg["g"]
    if g["model"] == "rm_table":
        return RmTableG(g["r_table"], isotonic=g.get("isotonic", False))
    if g["model"] == "prior_v1":
        return PriorV1G(**{k: v for k, v in g.items() if k != "model"})
    raise ValueError(f"알 수 없는 G 모델: {g['model']}")
