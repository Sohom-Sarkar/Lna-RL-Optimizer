# LNA RL Sizer

Automated sizing of 2.4 GHz CMOS low-noise amplifiers with reinforcement
learning, optimizing noise figure first. Every training step runs a real
ngspice simulation, so there is no surrogate model in the loop.

## Objectives

The agent trades off five metrics against the following targets:

| Metric | Target | Notes |
|---|---|---|
| NF | <= 2.5 dB | primary objective, carries the largest reward weight |
| S21 | >= 15 dB | power gain |
| IIP3 | >= -5 dBm | linearity, analytical estimate (see below) |
| Pdc | <= 20 mW | DC power budget |
| S11 | <= -10 dB | input return loss |

NF, S21, Pdc and S11 come straight out of ngspice. IIP3 is an analytical
square-law estimate derived from the simulated bias current rather than a
two-tone PSS sweep, which keeps each step fast enough to train on. Treat it
as a shaping term, not a verified number.

## Topologies

Pick the circuit with `--topology`; the agent sizes whatever that topology
exposes. All four share the same 50 Ω test bench and metric extraction, so
results are directly comparable.

| Topology | Design vars | Description |
|---|---|---|
| `cascode` (default) | 13 | Inductively degenerated cascode, M1 CS + M2 CG, Ld-Cd tank load |
| `common_source` | 11 | Single transistor with inductive degeneration, tank load |
| `common_gate` | 10 | CG input stage, 1/gm wideband match, tank load |
| `resistive_feedback` | 7 | Self-biased CS with drain-gate feedback resistor, resistive load |

Adding a topology means writing one `_body_*` function that emits a netlist
body using the shared `n_in` / `n_out` nodes, then registering it in
`TOPOLOGIES` with its design-variable bounds. Nothing else needs to change.

## Design space

Cascode variables and their search ranges:

| Variable | Range | Description |
|---|---|---|
| `w1_um` | 20-400 µm | M1 gate width |
| `w2_um` | 20-400 µm | M2 gate width |
| `l1_nm` | 130-300 nm | M1 gate length |
| `l2_nm` | 130-300 nm | M2 gate length |
| `lg_nh` | 0.5-30 nH | gate inductor |
| `ls_ph` | 20-1500 pH | source degeneration inductor |
| `ld_nh` | 0.5-20 nH | drain load inductor |
| `ibias_ma` | 1-20 mA | bias current |
| `vdd_v` | 1.2-2.5 V | supply voltage |
| `cin_pf` | 1-20 pF | input DC block |
| `cout_pf` | 1-20 pF | output DC block |
| `rbias_kohm` | 10-200 kΩ | gate bias resistor |
| `cd_ff` | 50-2000 fF | tank capacitor |

The other topologies use subsets of these plus their own variables
(`resistive_feedback` adds `rf_kohm` and `rd_ohm`, `common_gate` swaps
`ls_ph` for `ls_nh`). Exact bounds live in `TOPOLOGIES` in
`lna_rl/simulator.py`.

Gate bias voltages are not design variables. They are derived from the
requested bias current with a square-law estimate so the transistors stay in
saturation, and `resistive_feedback` self-biases through `Rf`. If `cd_ff` is
left unset when calling `simulate()` directly, the tank cap is derived for
resonance at 2.4 GHz instead.

## Setup

```bash
pip install -r requirements.txt
```

ngspice is not bundled. Download the ngspice-46 binary from
<https://ngspice.sourceforge.io/download.html> and drop `ngspice_con.exe`
(plus `libomp140.x86_64.dll` on Windows) into `ngspice/`. Verify with:

```bash
python smoke_test.py
```

That simulates one hand-picked design per topology and steps each Gym
environment a few times.

## Usage

```bash
# train the default cascode for 5000 steps
python train.py

# other topologies, longer run
python train.py --topology common_gate --steps 20000

# resume a checkpoint
python train.py --topology cascode --resume runs/lna_sac_cascode --steps 10000

# plot the Pareto front
python visualize.py --pareto runs/pareto_cascode.pkl
```

Checkpoints and Pareto fronts are written per topology to
`runs/lna_sac_<topology>.zip` and `runs/pareto_<topology>.pkl`.

## How it works

`LNAEnv` is a Gymnasium environment whose action vector is the topology's
design variables, each normalized to [-1, 1]. One step maps that vector to
physical units, builds a netlist, runs ngspice, parses the five metrics, and
returns a scalarized reward. The observation is the previous step's metrics
normalized against their targets.

The reward is a weighted sum of `tanh` margin terms, one per metric, plus a
bonus when the primary specs are met simultaneously and a flat penalty for
netlists that fail to converge. NF uses the tightest `tanh` scale so the
gradient stays strong below 2.5 dB. Training uses SAC from
stable-baselines3.

Scalarizing loses the trade-off structure, so `ParetoFront` separately tracks
every non-dominated design over (NF, S21, IIP3, Pdc) across the whole run.
That front, not the final policy, is the useful output.

## Device model

PTM 180nm BSIM3v3 (level 8), published by Arizona State University and
included in `models/`. Centre frequency is 2.4 GHz, set by `FREQ_GHZ` in
`lna_rl/simulator.py`.

## Layout

```
lna_rl/
  simulator.py   topology registry, netlist builders, ngspice runner, parsers
  env.py         Gymnasium environment and reward
  pareto.py      non-dominated front tracker and 2D hypervolume
models/
  ptm180nm.lib   PTM 180nm BSIM3v3 model card
ngspice/         ngspice binary, not tracked
train.py         SAC training entry point
visualize.py     Pareto front plots
smoke_test.py    end-to-end check across all topologies
```

## Limitations

- Inductors and capacitors are ideal. No Q, no self-resonance, no layout
  parasitics beyond a lumped drain capacitance estimate.
- IIP3 is analytical, as described above.
- Bias derivation uses a square-law approximation calibrated by hand against
  this specific model card. It will need retuning for a different PDK.
- Single corner, single temperature. No process or mismatch analysis.
