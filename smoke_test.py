"""
Smoke test: run one simulation per topology with a known-reasonable design,
print what ngspice returns, and check that the Gym env steps correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lna_rl.simulator import (
    simulate, build_topology_netlist, TOPOLOGIES, NGSPICE, MODEL_FILE,
)
from lna_rl.env import LNAEnv, compute_reward
from lna_rl.pareto import ParetoFront
import os

print("LNA RL Sizer smoke test")
print("-" * 50)
print(f"ngspice path : {NGSPICE}")
print(f"ngspice found: {os.path.exists(NGSPICE)}")
print(f"model file   : {MODEL_FILE}")
print(f"model found  : {os.path.exists(MODEL_FILE)}")

# known-reasonable test designs per topology
TEST_DESIGNS = {
    "cascode": dict(
        w1_um=40, w2_um=40, l1_nm=180, l2_nm=180,
        lg_nh=10, ls_ph=300, ld_nh=5, ibias_ma=8, vdd_v=1.8,
    ),
    "common_source": dict(
        w1_um=60, l1_nm=180,
        lg_nh=10, ls_ph=300, ld_nh=5, ibias_ma=8, vdd_v=1.8,
    ),
    "common_gate": dict(
        w1_um=100, l1_nm=180,
        ls_nh=5, ld_nh=5, ibias_ma=6, vdd_v=1.8,
    ),
    "resistive_feedback": dict(
        w1_um=120, l1_nm=180,
        rf_kohm=1.0, rd_ohm=300, vdd_v=1.8,
    ),
}

# 1. sample netlist (cascode)
print("\nSample cascode netlist (first 30 lines):")
params = {**TOPOLOGIES["cascode"].defaults, **TEST_DESIGNS["cascode"]}
nl = build_topology_netlist("cascode", params)
for line in nl.splitlines()[:30]:
    print(line)
print("  ...")

# 2. one simulation per topology
last_result = None
for topo, design in TEST_DESIGNS.items():
    print(f"\nSimulating '{topo}' ({TOPOLOGIES[topo].description})")
    result = simulate(topology=topo, **design)
    print(f"  valid   : {result.valid}")
    print(f"  NF      : {result.nf_db:.2f} dB")
    print(f"  S21     : {result.s21_db:.2f} dB")
    print(f"  IIP3    : {result.iip3_dbm:.2f} dBm")
    print(f"  Pdc     : {result.pdc_mw:.2f} mW")
    print(f"  S11     : {result.s11_db:.2f} dB")
    if result.error:
        print(f"  error   : {result.error[:120]}")
    print(f"  reward  : {compute_reward(result):.4f}")
    if topo == "cascode":
        last_result = result

# 3. Gym env, 3 random steps per topology
for topo in TOPOLOGIES:
    print(f"\nGym env '{topo}' (3 random steps)")
    env = LNAEnv(topology=topo, max_steps=10, render_mode="human")
    obs, info = env.reset()
    print(f"  action dims: {env.action_space.shape[0]}  "
          f"({', '.join(env.param_names)})")
    for _ in range(3):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

# 4. Pareto front
print("\nPareto front check")
pf = ParetoFront()
pf.add({"w1_um": 40}, last_result, step=1)
print(f"  Pareto size: {len(pf)}")
print(f"  HV (2D)    : {pf.hypervolume():.4f}")
print(pf.summary())

print("\nSmoke test complete.")
