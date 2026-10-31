"""한 경기의 상태 시계열 (L1 → 틱 정렬 배열). G 모델과 귀속 엔진의 공통 입력."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from common.paths import L1

SIDES = (100, 200)


def opp(side: int) -> int:
    return 200 if side == 100 else 100


def _read(table: str, game_id: str) -> pl.DataFrame:
    return pl.read_parquet(L1 / table / "**" / f"{game_id}.parquet", hive_partitioning=True)


@dataclass
class GameState:
    game_id: str
    tournament: str
    stage: str
    patch: str
    t: np.ndarray                          # (n,) 틱 gameTime ms, 오름차순
    team_gold: dict[int, np.ndarray]       # side → (n,)
    pids: np.ndarray                       # (10,) participant_id
    p_team: np.ndarray                     # (10,)
    p_role: list[str]
    p_name: list[str]
    p_champ: list[str]
    p_gold: np.ndarray                     # (n, 10) totalGold
    p_alive: np.ndarray                    # (n, 10) bool
    p_respawn: np.ndarray                  # (n, 10) 초
    p_health: np.ndarray                   # (n, 10)
    p_x: np.ndarray                        # (n, 10)
    p_z: np.ndarray                        # (n, 10)
    kills: pl.DataFrame
    kill_damage: pl.DataFrame
    epics: pl.DataFrame                    # 에픽 필터 적용 후
    buildings: pl.DataFrame

    @property
    def n(self) -> int:
        return self.t.size

    @property
    def t_end(self) -> int:
        return int(self.t[-1])

    @property
    def minutes(self) -> float:
        return (self.t[-1] - self.t[0]) / 60000.0

    def team_idx(self, side: int) -> np.ndarray:
        return np.flatnonzero(self.p_team == side)

    def pid_index(self) -> dict[int, int]:
        return {int(p): i for i, p in enumerate(self.pids)}

    # ── 상태 피처 (정본 v1.0 §1.2) ─────────────────────────────
    def alive_count(self, side: int) -> np.ndarray:
        return self.p_alive[:, self.team_idx(side)].sum(axis=1)

    def _cum_events(self, times_ms: np.ndarray) -> np.ndarray:
        """틱 시각 t까지(포함) 발생한 이벤트 수."""
        return np.searchsorted(np.sort(times_ms), self.t, side="right")

    def dragon_stacks(self, side: int) -> np.ndarray:
        d = self.epics.filter((pl.col("monster_type") == "dragon") & ~pl.col("is_elder") & (pl.col("killer_team") == side))
        return self._cum_events(d["game_time_ms"].to_numpy())

    def towers_lost(self, side: int) -> np.ndarray:
        b = self.buildings.filter((pl.col("building_type") == "turret") & (pl.col("team_id") == side))
        return self._cum_events(b["game_time_ms"].to_numpy())

    def baron_active(self, side: int, buff_ms: int) -> np.ndarray:
        kills = self.epics.filter((pl.col("monster_type") == "baron") & (pl.col("killer_team") == side))["game_time_ms"].to_numpy()
        active = np.zeros(self.n, dtype=bool)
        for k in kills:
            active |= (self.t >= k) & (self.t < k + buff_ms)
        return active.astype(float)


def load_state(game_id: str) -> GameState:
    games = _read("games", game_id).row(0, named=True)
    ft = _read("frames_team", game_id).sort("game_time_ms", "team_id")
    fp = _read("frames_player", game_id).sort("game_time_ms", "participant_id")
    parts = _read("participants", game_id).sort("participant_id")

    t = ft.filter(pl.col("team_id") == SIDES[0])["game_time_ms"].to_numpy()
    team_gold = {s: ft.filter(pl.col("team_id") == s)["total_gold"].to_numpy().astype(float) for s in SIDES}
    pids = parts["participant_id"].to_numpy()
    n, m = t.size, pids.size
    if fp.height != n * m:
        raise ValueError(f"{game_id}: frames_player가 틱×선수 격자가 아닙니다 ({fp.height} != {n}×{m})")

    def grid(col: str, dtype=float) -> np.ndarray:
        return fp[col].to_numpy().astype(dtype).reshape(n, m)

    return GameState(
        game_id=game_id, tournament=games["tournament"], stage=games["stage"], patch=games["patch"],
        t=t, team_gold=team_gold, pids=pids, p_team=parts["team_id"].to_numpy(),
        p_role=parts["role"].to_list(), p_name=parts["player_name"].to_list(), p_champ=parts["champion"].to_list(),
        p_gold=grid("total_gold"), p_alive=grid("alive", bool), p_respawn=grid("respawn_timer"),
        p_health=grid("health"), p_x=grid("x"), p_z=grid("z"),
        kills=_read("kills", game_id), kill_damage=_read("kill_damage", game_id),
        epics=_read("monsters", game_id).filter("is_epic"), buildings=_read("buildings", game_id),
    )


def list_games() -> pl.DataFrame:
    return pl.read_parquet(L1 / "games" / "**" / "*.parquet", hive_partitioning=True).sort("tournament", "stage", "match_id", "game_n")
