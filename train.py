"""
SAC training script for LNA sizing.

Usage:
    python train.py                                   # cascode, 5000 steps
    python train.py --topology common_gate            # pick a topology
    python train.py --steps 20000                     # longer run
    python train.py --resume runs/lna_sac_cascode     # continue a checkpoint
"""

import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from lna_rl.env import LNAEnv
from lna_rl.pareto import ParetoFront
from lna_rl.simulator import TOPOLOGIES

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


class LNACallback(BaseCallback):
    """Logs metrics and updates the Pareto front at every step."""

    def __init__(self, pareto: ParetoFront, log_interval: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.pareto = pareto
        self.log_interval = log_interval
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", [{}]):
            result = info.get("result")
            params = info.get("params")
            if result is not None and params is not None:
                self.pareto.add(params, result, step=self.num_timesteps)

        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self._t0
            hv = self.pareto.hypervolume()
            best_nf = self.pareto.best_nf()
            nf_str = f"{best_nf.result.nf_db:.2f}dB" if best_nf else "n/a"
            gain_str = f"{best_nf.result.s21_db:.1f}dB" if best_nf else "n/a"
            print(
                f"[step {self.num_timesteps:6d}] "
                f"Pareto={len(self.pareto):3d}  HV={hv:.4f}  "
                f"BestNF={nf_str}  Gain@BestNF={gain_str}  "
                f"t={elapsed:.0f}s"
            )
        return True


def train(steps: int = 5000, resume: str = None, seed: int = 42,
          topology: str = "cascode"):
    env = Monitor(LNAEnv(topology=topology, max_steps=200, render_mode=None))
    pareto = ParetoFront()

    if resume and Path(resume + ".zip").exists():
        print(f"Resuming from {resume}")
        model = SAC.load(resume, env=env)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=50_000,
            learning_starts=200,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=seed,
        )

    callback = LNACallback(pareto, log_interval=50)

    print(f"Training SAC for {steps} steps...")
    print(f"Topology: {topology} ({TOPOLOGIES[topology].description})")
    print(f"Design variables ({len(TOPOLOGIES[topology].bounds)}): "
          f"{list(TOPOLOGIES[topology].bounds.keys())}")
    print(f"ngspice backend: {os.path.exists('ngspice/ngspice_con.exe')}")
    model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=(resume is None))

    save_path = str(RUNS_DIR / f"lna_sac_{topology}")
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    import pickle
    pareto_path = str(RUNS_DIR / f"pareto_{topology}.pkl")
    with open(pareto_path, "wb") as f:
        pickle.dump(pareto, f)
    print(f"Pareto front saved to {pareto_path}")
    print(pareto.summary())

    return model, pareto


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", type=str, default="cascode",
                        choices=list(TOPOLOGIES.keys()),
                        help="Circuit topology the agent will size")
    args = parser.parse_args()
    train(steps=args.steps, resume=args.resume, seed=args.seed,
          topology=args.topology)
