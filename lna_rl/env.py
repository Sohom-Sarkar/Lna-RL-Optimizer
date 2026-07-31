"""
Gymnasium environment for LNA sizing.

Action space : N continuous design variables (topology-dependent), each
               normalized to [-1, 1]
Observation  : 5 normalized circuit metrics from the last simulation
Reward       : noise-figure-first scalarized multi-objective

The circuit topology is fixed at construction time (LNAEnv(topology=...));
the agent then sizes that topology's components. See
lna_rl.simulator.TOPOLOGIES for the available topologies and their
design variables.

Reward structure: NF carries the largest weight (primary objective),
followed by gain and IIP3 bonuses, power and S11 penalties, a large
constant penalty for non-converging designs, and a small bonus when all
primary specs are met at once.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import simulate, SimResult, TOPOLOGIES

# ---- design targets --------------------------------------------------------
TARGETS = {
    "nf_db":    2.5,    # primary objective: NF at or below 2.5 dB
    "s21_db":   15.0,   # gain at or above 15 dB
    "iip3_dbm": -5.0,   # IIP3 at or above -5 dBm
    "pdc_mw":   20.0,   # power budget, 20 mW
    "s11_db":   -10.0,  # S11 at or below -10 dB (more negative is better)
}

# ---- reward weights --------------------------------------------------------
# NF dominates by design. IIP3 is held low because the analytical estimate
# clears its target for most of the design space, so a larger weight would
# just add a near-constant offset.
W_NF    = 4.0
W_GAIN  = 1.5
W_IIP3  = 0.8
W_PDC   = 0.8
W_S11   = 1.0
INVALID_PENALTY = -20.0


def _normalize_obs(result: SimResult) -> np.ndarray:
    """Map simulation metrics to a bounded observation vector."""
    # Each metric normalized so 0 = at-target, positive = better-than-target
    nf_norm    = np.clip((TARGETS["nf_db"]    - result.nf_db)    / 5.0,  -2, 2)
    s21_norm   = np.clip((result.s21_db       - TARGETS["s21_db"])/ 10.0, -2, 2)
    iip3_norm  = np.clip((result.iip3_dbm     - TARGETS["iip3_dbm"]) / 15.0, -2, 2)
    pdc_norm   = np.clip((TARGETS["pdc_mw"]   - result.pdc_mw)   / 20.0, -2, 2)
    s11_norm   = np.clip((result.s11_db       - TARGETS["s11_db"]) / 10.0, -2, 2)
    return np.array([nf_norm, s21_norm, iip3_norm, pdc_norm, s11_norm],
                    dtype=np.float32)


def compute_reward(result: SimResult) -> float:
    if not result.valid:
        return INVALID_PENALTY

    # One bounded margin term per metric: positive when the spec is met,
    # negative when it is violated, saturating either way.
    def margin(actual, target, higher_is_better=True, scale=1.0):
        diff = (actual - target) * (1 if higher_is_better else -1)
        return float(np.tanh(diff / scale))

    # The tight NF scale keeps the gradient alive below 2.5 dB instead of
    # saturating as soon as the spec is met.
    r_nf   = W_NF   * margin(result.nf_db,    TARGETS["nf_db"],    higher_is_better=False, scale=0.5)
    r_gain = W_GAIN  * margin(result.s21_db,   TARGETS["s21_db"],   higher_is_better=True,  scale=3.0)
    r_iip3 = W_IIP3  * margin(result.iip3_dbm, TARGETS["iip3_dbm"], higher_is_better=True,  scale=5.0)
    r_pdc  = W_PDC   * margin(result.pdc_mw,   TARGETS["pdc_mw"],   higher_is_better=False, scale=5.0)
    r_s11  = W_S11   * margin(result.s11_db,   TARGETS["s11_db"],   higher_is_better=False, scale=3.0)

    # Extra credit for landing inside the feasible region, not just near it.
    all_met = (
        result.nf_db    <= TARGETS["nf_db"]
        and result.s21_db   >= TARGETS["s21_db"]
        and result.iip3_dbm >= TARGETS["iip3_dbm"]
        and result.pdc_mw   <= TARGETS["pdc_mw"]
    )
    r_bonus = 1.0 if all_met else 0.0

    return r_nf + r_gain + r_iip3 + r_pdc + r_s11 + r_bonus


class LNAEnv(gym.Env):
    """
    Single-episode LNA sizing environment.
    Each step is one SPICE simulation call.
    Episode terminates after max_steps or when all specs are met.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, topology: str = "cascode", max_steps: int = 200,
                 render_mode: Optional[str] = None):
        super().__init__()
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"Unknown topology '{topology}'. "
                f"Available: {list(TOPOLOGIES.keys())}"
            )
        self.topology = topology
        self.bounds = TOPOLOGIES[topology].bounds
        self.param_names = list(self.bounds.keys())
        self.n_params = len(self.param_names)

        self.max_steps = max_steps
        self.render_mode = render_mode
        self._step_count = 0
        self._last_result: Optional[SimResult] = None
        self._last_params: Optional[dict] = None
        self._best_reward = -np.inf
        self._best_result: Optional[SimResult] = None
        self._best_params: Optional[dict] = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32
        )
        # Observation: 5 normalized metrics
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

    def _normalize_action(self, action: np.ndarray) -> dict[str, float]:
        """Map [-1, 1] action vector to physical design variables."""
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            # linear mapping: -1 maps to lo, +1 to hi
            params[name] = float(lo + (action[i] + 1.0) / 2.0 * (hi - lo))
        return params

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._last_result = None
        obs = np.zeros(5, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        params = self._normalize_action(np.clip(action, -1.0, 1.0))
        result = simulate(topology=self.topology, **params)

        self._last_result = result
        self._last_params = params

        reward = compute_reward(result)

        if result.valid and reward > self._best_reward:
            self._best_reward = reward
            self._best_result = result
            self._best_params = dict(params)

        obs = _normalize_obs(result) if result.valid else np.zeros(5, dtype=np.float32)

        # Termination: all specs met simultaneously
        terminated = (
            result.valid
            and result.nf_db    <= TARGETS["nf_db"]
            and result.s21_db   >= TARGETS["s21_db"]
            and result.iip3_dbm >= TARGETS["iip3_dbm"]
            and result.pdc_mw   <= TARGETS["pdc_mw"]
            and result.s11_db   <= TARGETS["s11_db"]
        )
        truncated = self._step_count >= self.max_steps

        info = {
            "result": result,
            "params": params,
            "step": self._step_count,
            "topology": self.topology,
        }

        if self.render_mode == "human":
            self._render_human(result, params, reward)

        return obs, reward, terminated, truncated, info

    def _render_human(self, result: SimResult, params: dict, reward: float):
        if result.valid:
            key_knob = next(iter(params), None)
            knob_str = f"{key_knob}={params[key_knob]:.1f}" if key_knob else ""
            print(
                f"[{self._step_count:4d}] "
                f"NF={result.nf_db:5.2f}dB  "
                f"S21={result.s21_db:5.1f}dB  "
                f"IIP3={result.iip3_dbm:5.1f}dBm  "
                f"Pdc={result.pdc_mw:5.1f}mW  "
                f"S11={result.s11_db:5.1f}dB  "
                f"R={reward:+.3f}  "
                f"{knob_str}"
            )
        else:
            print(f"[{self._step_count:4d}] INVALID  R={reward:.3f}  {result.error[:60]}")

    def render(self):
        if self._last_result is not None:
            self._render_human(self._last_result, self._last_params or {}, 0.0)
