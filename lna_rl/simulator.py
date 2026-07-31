"""
ngspice-backed LNA simulator with selectable topology.

Topologies (see the TOPOLOGIES registry below):
  cascode            - inductively degenerated cascode (M1 CS + M2 CG)
  common_source      - single-transistor CS with inductive degeneration
  common_gate        - CG input stage, 1/gm wideband match
  resistive_feedback - self-biased CS with drain-gate feedback resistor

All topologies share the same 50-ohm test bench (Vs/Rs into node n_in,
Rout at n_out behind DC blocks) and the same .CONTROL block, so the
metric parsers work unchanged for every topology.

Metrics returned per run:
  s21_db   - gain at centre frequency, corrected to power gain [dB]
  nf_db    - noise figure [dB]
  iip3_dbm - analytical IIP3 estimate from measured bias current [dBm]
  pdc_mw   - DC power [mW]
  s11_db   - input reflection [dB]
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

NGSPICE = str(Path(__file__).parents[1] / "ngspice" / "ngspice_con.exe")
MODEL_FILE = str(Path(__file__).parents[1] / "models" / "ptm180nm.lib")

FREQ_GHZ = 2.4
FREQ_HZ  = FREQ_GHZ * 1e9
TEMP_K   = 290.0
TEMP_C   = TEMP_K - 273.15
RS       = 50.0   # source/load impedance
VTH      = 0.42   # PTM 180nm nmos Vth0 plus a small body/DIBL correction


@dataclass
class SimResult:
    s21_db:   float
    nf_db:    float
    iip3_dbm: float
    pdc_mw:   float
    s11_db:   float
    valid:    bool  = True
    error:    str   = ""


INVALID = SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                    pdc_mw=999, s11_db=0, valid=False)


# ---- netlist helpers -------------------------------------------------------

def _w(v): return f"{v:.4f}e-6"
def _l(v): return f"{v:.1f}e-9"
def _nh(v): return f"{v:.5f}e-9"
def _pf(v): return f"{v:.4f}e-12"
def _ff(v): return f"{v:.4f}e-15"


def _bias_point(w1_um: float, l1_nm: float, ibias_ma: float) -> tuple[float, float]:
    """
    Square-law estimate of (Vg1, Vdsat) for the input transistor.
    Effective uCox is ~100 uA/V^2 rather than the nominal 270, which accounts
    for velocity saturation below Vdsat = 0.3 V (empirical, from a device
    sweep on this model).
    """
    mu_cox_eff = 100e-6
    kp = mu_cox_eff * (w1_um * 1e-6) / (l1_nm * 1e-9)
    ids = ibias_ma * 1e-3
    vdsat = float(np.clip(np.sqrt(2.0 * ids / (kp + 1e-12)), 0.05, 0.35))
    return VTH + vdsat, vdsat


def _tank_cap_ff(cd_ff: Optional[float], ld_nh: float, freq_hz: float,
                 w_drain_um: float) -> float:
    """
    Tank capacitor in fF. If cd_ff is given (tunable design variable) use it;
    otherwise derive the value that resonates Ld at freq_hz, minus the MOSFET
    drain parasitics (~1.3 fF/um at the drain node for PTM 180nm).
    """
    if cd_ff is not None:
        return max(10.0, float(cd_ff))
    c = 1.0 / ((2 * np.pi * freq_hz) ** 2 * ld_nh * 1e-9) - w_drain_um * 1.3e-15
    return max(50.0, c * 1e15)


# ---- topology netlist builders ---------------------------------------------
# Each builder returns the circuit body only. Builders must use the shared
# node names n_in (after Rs) and n_out (before Rout) so the common .CONTROL
# block and parsers work for every topology.

def _body_cascode(p: dict, freq_hz: float) -> str:
    vg1, vdsat = _bias_point(p["w1_um"], p["l1_nm"], p["ibias_ma"])
    vg1 = float(np.clip(vg1, 0.45, p["vdd_v"] - 0.5))
    # Cascode gate: Vg1 + Vth + 2*Vdsat keeps M1 out of triode
    vbias = float(np.clip(vg1 + VTH + 2 * vdsat + 0.1, 0.7, p["vdd_v"] - 0.1))
    cd = _tank_cap_ff(p.get("cd_ff"), p["ld_nh"], freq_hz, p["w2_um"])

    return textwrap.dedent(f"""\
    * Supply
    Vdd  vdd  0  DC {p['vdd_v']:.4f}
    Vbias vbias 0 DC {vbias:.4f}

    * Input: 50-ohm source with DC-blocking cap
    Vs   vs   0   AC 1 DC 0
    Rs   vs   n_in  {RS}
    Cin  n_in  n_rf  {_pf(p['cin_pf'])}

    * Input matching: gate inductor
    Lg   n_rf  n_g1  {_nh(p['lg_nh'])}

    * DC gate bias for M1 via large resistor (AC isolated)
    Vg1    n_vg1  0  DC {vg1:.4f}
    Rbias  n_vg1  n_g1  {p['rbias_kohm']:.3f}k

    * Source degeneration
    Ls   n_s1  0    {_nh(p['ls_ph'] * 1e-3)}

    * Cascode: M1 (CS) + M2 (CG)
    M1   n_d1  n_g1  n_s1  0  nmos180  W={_w(p['w1_um'])}  L={_l(p['l1_nm'])}
    M2   n_d2  vbias n_d1  0  nmos180  W={_w(p['w2_um'])}  L={_l(p['l2_nm'])}

    * Load: Ld-Cd parallel tank
    Ld   vdd  n_d2  {_nh(p['ld_nh'])}
    Cd   n_d2  0    {_ff(cd)}

    * Output: DC-blocking cap + 50-ohm load
    Cout n_d2  n_out  {_pf(p['cout_pf'])}
    Rout n_out 0      {RS}
    """)


def _body_common_source(p: dict, freq_hz: float) -> str:
    vg1, _ = _bias_point(p["w1_um"], p["l1_nm"], p["ibias_ma"])
    vg1 = float(np.clip(vg1, 0.45, p["vdd_v"] - 0.3))
    cd = _tank_cap_ff(p.get("cd_ff"), p["ld_nh"], freq_hz, p["w1_um"])

    return textwrap.dedent(f"""\
    * Supply
    Vdd  vdd  0  DC {p['vdd_v']:.4f}

    * Input: 50-ohm source with DC-blocking cap
    Vs   vs   0   AC 1 DC 0
    Rs   vs   n_in  {RS}
    Cin  n_in  n_rf  {_pf(p['cin_pf'])}

    * Input matching: gate inductor
    Lg   n_rf  n_g1  {_nh(p['lg_nh'])}

    * DC gate bias via large resistor
    Vg1    n_vg1  0  DC {vg1:.4f}
    Rbias  n_vg1  n_g1  {p['rbias_kohm']:.3f}k

    * Source degeneration
    Ls   n_s1  0    {_nh(p['ls_ph'] * 1e-3)}

    * Single common-source transistor
    M1   n_d1  n_g1  n_s1  0  nmos180  W={_w(p['w1_um'])}  L={_l(p['l1_nm'])}

    * Load: Ld-Cd parallel tank
    Ld   vdd  n_d1  {_nh(p['ld_nh'])}
    Cd   n_d1  0    {_ff(cd)}

    * Output
    Cout n_d1  n_out  {_pf(p['cout_pf'])}
    Rout n_out 0      {RS}
    """)


def _body_common_gate(p: dict, freq_hz: float) -> str:
    # Source sits at DC ~0 V through Ls, so Vg1 = Vth + Vdsat directly.
    vg1, _ = _bias_point(p["w1_um"], p["l1_nm"], p["ibias_ma"])
    vg1 = float(np.clip(vg1, 0.45, p["vdd_v"] - 0.2))
    cd = _tank_cap_ff(p.get("cd_ff"), p["ld_nh"], freq_hz, p["w1_um"])

    return textwrap.dedent(f"""\
    * Supply
    Vdd  vdd  0  DC {p['vdd_v']:.4f}

    * Input: 50-ohm source, DC-blocked, drives the transistor SOURCE
    Vs   vs   0   AC 1 DC 0
    Rs   vs   n_in  {RS}
    Cin  n_in  n_s1  {_pf(p['cin_pf'])}

    * Source bias inductor: DC path to ground, resonates with Cgs
    Ls   n_s1  0    {_nh(p['ls_nh'])}

    * Gate: DC bias through Rbias, AC-grounded by bypass cap
    Vg1    n_vg1  0  DC {vg1:.4f}
    Rbias  n_vg1  n_g1  {p['rbias_kohm']:.3f}k
    Cbyp   n_g1   0     20e-12

    * Common-gate transistor
    M1   n_d1  n_g1  n_s1  0  nmos180  W={_w(p['w1_um'])}  L={_l(p['l1_nm'])}

    * Load: Ld-Cd parallel tank
    Ld   vdd  n_d1  {_nh(p['ld_nh'])}
    Cd   n_d1  0    {_ff(cd)}

    * Output
    Cout n_d1  n_out  {_pf(p['cout_pf'])}
    Rout n_out 0      {RS}
    """)


def _body_resistive_feedback(p: dict, freq_hz: float) -> str:
    # Self-biased: Rf ties gate to drain at DC (no gate current), so
    # Vgs = Vds and the transistor is guaranteed in saturation.
    return textwrap.dedent(f"""\
    * Supply
    Vdd  vdd  0  DC {p['vdd_v']:.4f}

    * Input: 50-ohm source with DC-blocking cap, straight into the gate
    Vs   vs   0   AC 1 DC 0
    Rs   vs   n_in  {RS}
    Cin  n_in  n_g1  {_pf(p['cin_pf'])}

    * Drain-gate feedback resistor (also provides self-bias)
    Rf   n_d1  n_g1  {p['rf_kohm']:.4f}k

    * Resistive drain load (wideband)
    Rd   vdd  n_d1  {p['rd_ohm']:.2f}

    * Common-source transistor
    M1   n_d1  n_g1  0  0  nmos180  W={_w(p['w1_um'])}  L={_l(p['l1_nm'])}

    * Output
    Cout n_d1  n_out  {_pf(p['cout_pf'])}
    Rout n_out 0      {RS}
    """)


# ---- topology registry -----------------------------------------------------

@dataclass(frozen=True)
class Topology:
    name: str
    description: str
    # RL design-variable bounds, in physical units. Order defines action order.
    bounds: dict
    # Defaults for parameters callers may omit (None = auto-derive, e.g. cd_ff).
    defaults: dict
    body: Callable[[dict, float], str]


_PASSIVE_BOUNDS = {
    "cin_pf":     (1.0,  20.0),
    "cout_pf":    (1.0,  20.0),
    "rbias_kohm": (10.0, 200.0),
}
_PASSIVE_DEFAULTS = {"cin_pf": 10.0, "cout_pf": 10.0, "rbias_kohm": 100.0}

TOPOLOGIES: dict[str, Topology] = {
    "cascode": Topology(
        name="cascode",
        description="Inductively-degenerated cascode (M1 CS + M2 CG)",
        bounds={
            "w1_um":    (20.0,  400.0),
            "w2_um":    (20.0,  400.0),
            "l1_nm":    (130.0, 300.0),
            "l2_nm":    (130.0, 300.0),
            "lg_nh":    (0.5,   30.0),
            "ls_ph":    (20.0,  1500.0),
            "ld_nh":    (0.5,   20.0),
            "ibias_ma": (1.0,   20.0),
            "vdd_v":    (1.2,   2.5),
            **_PASSIVE_BOUNDS,
            "cd_ff":    (50.0,  2000.0),
        },
        defaults={**_PASSIVE_DEFAULTS, "cd_ff": None},
        body=_body_cascode,
    ),
    "common_source": Topology(
        name="common_source",
        description="Single-transistor CS with inductive degeneration",
        bounds={
            "w1_um":    (20.0,  400.0),
            "l1_nm":    (130.0, 300.0),
            "lg_nh":    (0.5,   30.0),
            "ls_ph":    (20.0,  1500.0),
            "ld_nh":    (0.5,   20.0),
            "ibias_ma": (1.0,   20.0),
            "vdd_v":    (1.2,   2.5),
            **_PASSIVE_BOUNDS,
            "cd_ff":    (50.0,  2000.0),
        },
        defaults={**_PASSIVE_DEFAULTS, "cd_ff": None},
        body=_body_common_source,
    ),
    "common_gate": Topology(
        name="common_gate",
        description="Common-gate input stage (1/gm wideband match)",
        bounds={
            "w1_um":    (20.0,  400.0),
            "l1_nm":    (130.0, 300.0),
            "ls_nh":    (0.5,   30.0),
            "ld_nh":    (0.5,   20.0),
            "ibias_ma": (1.0,   20.0),
            "vdd_v":    (1.2,   2.5),
            **_PASSIVE_BOUNDS,
            "cd_ff":    (50.0,  2000.0),
        },
        defaults={**_PASSIVE_DEFAULTS, "cd_ff": None},
        body=_body_common_gate,
    ),
    "resistive_feedback": Topology(
        name="resistive_feedback",
        description="Self-biased CS with drain-gate feedback resistor",
        bounds={
            "w1_um":   (20.0,  400.0),
            "l1_nm":   (130.0, 300.0),
            "rf_kohm": (0.1,   5.0),
            "rd_ohm":  (100.0, 1500.0),
            "vdd_v":   (1.2,   2.5),
            "cin_pf":  (1.0,   20.0),
            "cout_pf": (1.0,   20.0),
        },
        defaults={"cin_pf": 10.0, "cout_pf": 10.0},
        body=_body_resistive_feedback,
    ),
}


def build_topology_netlist(
    topology: str, params: dict,
    freq_hz: float = FREQ_HZ, model_file: str = MODEL_FILE,
) -> str:
    """Full netlist (header + circuit body + common control block)."""
    topo = TOPOLOGIES[topology]
    body = topo.body(params, freq_hz)
    return textwrap.dedent(f"""\
    * {topology} LNA netlist (auto-generated)
    .temp {TEMP_C:.1f}
    .include "{model_file}"

    """) + body + textwrap.dedent(f"""\

    .CONTROL
    * DC operating point: supply branch current for Pdc
    op
    echo ===OP_START===
    print vdd#branch
    echo ===OP_END===

    * AC sweep, narrowed around the centre frequency for speed
    ac dec 30 {freq_hz*0.4:.6e} {freq_hz*1.8:.6e}
    echo ===AC_START===
    print vdb(n_out)
    echo ===AC_END===
    echo ===S11_START===
    print v(n_in)
    echo ===S11_END===

    * Noise analysis (setplot noise1 needed to access the spectral vectors)
    noise v(n_out) Vs dec 30 {freq_hz*0.4:.6e} {freq_hz*1.8:.6e}
    setplot noise1
    echo ===NOISE_START===
    print onoise_spectrum inoise_spectrum
    echo ===NOISE_END===

    .ENDC
    .END
    """)


def build_netlist(
    w1_um: float, w2_um: float,
    l1_nm: float, l2_nm: float,
    lg_nh: float, ls_ph: float, ld_nh: float,
    ibias_ma: float, vdd_v: float,
    freq_hz: float = FREQ_HZ,
    model_file: str = MODEL_FILE,
    out_prefix: str = "lna",
) -> str:
    """Positional convenience wrapper for the cascode, used by the debug scripts."""
    params = {
        "w1_um": w1_um, "w2_um": w2_um, "l1_nm": l1_nm, "l2_nm": l2_nm,
        "lg_nh": lg_nh, "ls_ph": ls_ph, "ld_nh": ld_nh,
        "ibias_ma": ibias_ma, "vdd_v": vdd_v,
        **TOPOLOGIES["cascode"].defaults,
    }
    return build_topology_netlist("cascode", params, freq_hz, model_file)


# ---- ngspice runner --------------------------------------------------------

def _run_ngspice(netlist: str, timeout: int = 25) -> tuple[str, str]:
    """
    Run a netlist and return (log_text, stderr).

    In batch mode (-b -o logfile) ngspice puts only its banner on stdout and
    everything else, including the print/echo output from .CONTROL, into the
    log file. The log is therefore the only useful data source.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        spi = os.path.join(tmpdir, "lna.spi")
        log = os.path.join(tmpdir, "out.log")
        with open(spi, "w") as f:
            f.write(netlist)
        try:
            proc = subprocess.run(
                [NGSPICE, "-b", "-o", log, spi],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"
        except Exception as e:
            return "", str(e)

        log_text = open(log).read() if os.path.exists(log) else ""
        return log_text, proc.stderr


def _section(log_text: str, key: str) -> str:
    """Extract text between ===KEY_START=== and ===KEY_END=== markers."""
    start = f"==={key.upper()}_START==="
    end   = f"==={key.upper()}_END==="
    i0 = log_text.find(start)
    if i0 < 0:
        return ""
    i0 += len(start)
    i1 = log_text.find(end, i0)
    return log_text[i0:i1] if i1 >= 0 else log_text[i0:]


# ---- metric parsers --------------------------------------------------------

def _parse_print_table(text: str) -> list[tuple[float, ...]]:
    """
    Parse ngspice `print` output.
    Format (batch mode, one or more variables):
      Index  variable1  variable2  ...
      0      freq_val   val1       val2   ...
      1      ...
    OR for single-variable DC:
      vdd#branch = -9.30927e-12
    Returns list of row tuples (all floats).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        # Skip header lines
        if not line or line.startswith("Index") or line.startswith("-"):
            continue
        # "name = value" format (DC print)
        m = re.match(r"[\w#()]+\s*=\s*([+-]?\d[\d.eE+\-]+)", line)
        if m:
            try:
                rows.append((float(m.group(1)),))
            except ValueError:
                pass
            continue
        # Tabular: "idx  val1  val2 ..."
        # ngspice prints complex values with a trailing comma on the real part
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                # first col is integer index; skip it
                _ = int(parts[0])
                row = tuple(float(p) for p in parts[1:])
                rows.append(row)
            except (ValueError, IndexError):
                pass
    return rows


def _parse_gain(ac_text: str, freq_hz: float) -> Optional[float]:
    """
    Return S21 in dB from ngspice `print vdb(n_out)` AC output.
    With Vs=1V AC and Rs=50 ohm: S21 = vdb(n_out) + 6 dB (Thevenin correction).
    """
    rows = _parse_print_table(ac_text)
    if not rows:
        return None
    best_vdb = None
    best_diff = 1e30
    for row in rows:
        if len(row) >= 2:
            f, vdb = row[0], row[1]
            diff = abs(f - freq_hz)
            if diff < best_diff:
                best_diff = diff
                best_vdb = vdb
    if best_vdb is None:
        return None
    return float(best_vdb + 6.0)


def _parse_s11(s11_text: str, freq_hz: float) -> float:
    """
    Compute S11 from `print v(n_in)` AC section.
    ngspice prints complex v(n_in) as: Index  freq  re(v)  im(v)
    With Vs=1V AC and Rs=50 ohm:
        S11 = (Zin - Rs)/(Zin + Rs),  Zin = Rs * Vin / (1 - Vin)
    Returns S11 in dB (negative = good match).
    """
    rows = _parse_print_table(s11_text)
    best_row = None
    best_diff = 1e30
    for row in rows:
        if len(row) >= 3:   # freq, re(v(n_in)), im(v(n_in))
            diff = abs(row[0] - freq_hz)
            if diff < best_diff:
                best_diff = diff
                best_row = row
    if best_row is None or len(best_row) < 3:
        return 0.0
    vin_re = best_row[1]
    vin_im = best_row[2] if len(best_row) >= 3 else 0.0
    vin = complex(vin_re, vin_im)
    denom = 1.0 - vin   # Vs=1V
    if abs(denom) < 1e-12:
        return 0.0
    zin = RS * vin / denom
    s11_lin = (zin - RS) / (zin + RS)
    s11_mag = abs(s11_lin)
    if s11_mag < 1e-10:
        return -60.0
    return float(np.clip(20 * np.log10(s11_mag), -60.0, 0.0))


def _parse_pdc(op_text: str, vdd_v: float) -> float:
    """Extract DC power from `print vdd#branch` output."""
    rows = _parse_print_table(op_text)
    for row in rows:
        # Single value row: (current,)
        if len(row) >= 1:
            try:
                return abs(float(row[0])) * vdd_v * 1e3
            except (ValueError, IndexError):
                pass
    return 999.0


def _parse_nf(noise_text: str, freq_hz: float, s21_db: float) -> float:
    """
    Parse NF from ngspice noise analysis (after setplot noise1).
    Print format: Index  freq  onoise_spectrum  inoise_spectrum  (V/sqrt(Hz) each)

    NF is computed from the input-referred noise density (inoise_spectrum):
        NF = 10*log10(inoise^2 / (4*k*T*Rs))
    This avoids gain ambiguity and is the standard SPICE noise figure definition.
    """
    kt4rs = 4 * 1.38e-23 * TEMP_K * RS

    rows = _parse_print_table(noise_text)
    best_inoise = None
    best_diff   = 1e30

    for row in rows:
        if len(row) >= 3:
            f       = row[0]
            inoise  = row[2]   # inoise_spectrum (col 3): input-referred noise V/sqrt(Hz)
            diff    = abs(f - freq_hz)
            if diff < best_diff:
                best_diff   = diff
                best_inoise = inoise

    if best_inoise is None or best_inoise <= 0:
        # Fallback: use conservative estimate based on gain (no hard floor at 3 dB)
        return float(np.clip(1.5 + max(0, 15.0 - s21_db) * 0.1, 0.5, 20.0))

    nf_lin = (best_inoise ** 2) / (kt4rs + 1e-50)
    nf_db  = 10 * np.log10(max(1e-3, nf_lin))
    return float(np.clip(nf_db, 0.0, 30.0))


def _estimate_iip3_analytical(w1_um: float, l1_nm: float, ids_ma: float) -> float:
    """
    Analytical IIP3 estimate for the input MOSFET.
    IIP3 ~ sqrt(8/3 * |a1/a3|) with a1 = gm and a3 the third-order Vgs term;
    for BSIM3v3 in saturation a3 ~ -gm/(4*Vdsat^2), which gives
        IIP3_V ~ 2*sqrt(2)*Vdsat
        IIP3_dBm ~ 20*log10(IIP3_V / sqrt(2*50)) + 30
    ids_ma is the *measured* supply current (Pdc/Vdd), so the estimate stays
    consistent across topologies, including the self-biased one.
    """
    mu_cox = 270e-6      # A/V^2, approximate un*Cox for 180nm
    kp = mu_cox * (w1_um * 1e-6) / (l1_nm * 1e-9)
    ids = max(ids_ma, 0.01) * 1e-3
    vdsat = np.sqrt(2 * ids / (kp + 1e-9))
    vdsat = float(np.clip(vdsat, 0.05, 0.8))
    viip3 = 2 * np.sqrt(2) * vdsat           # V (input-referred)
    piip3_dbm = 20 * np.log10(viip3 / np.sqrt(2 * RS) + 1e-12) + 30
    return float(np.clip(piip3_dbm, -30, 20))


# ---- public API ------------------------------------------------------------

def simulate(topology: str = "cascode", **params) -> SimResult:
    """
    Run one LNA simulation and return parsed metrics.

    topology : one of TOPOLOGIES keys ("cascode", "common_source",
               "common_gate", "resistive_feedback")
    params   : physical design variables for that topology (see
               TOPOLOGIES[topology].bounds). Parameters with defaults
               (cin_pf, cout_pf, rbias_kohm, cd_ff) may be omitted;
               cd_ff=None auto-derives the tank cap for resonance.
    """
    if topology not in TOPOLOGIES:
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False,
                         error=f"Unknown topology '{topology}'")
    topo = TOPOLOGIES[topology]

    # Fill defaults, reject unknown/missing parameters
    p = dict(topo.defaults)
    for k, v in params.items():
        if k not in topo.bounds:
            return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                             pdc_mw=999, s11_db=0, valid=False,
                             error=f"Unknown parameter '{k}' for {topology}")
        p[k] = v
    missing = [k for k in topo.bounds if k not in p]
    if missing:
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False,
                         error=f"Missing parameters: {missing}")

    # Physical sanity bounds: allow 0.5x-1.5x slack around the RL search range
    for k, (lo, hi) in topo.bounds.items():
        v = p[k]
        if v is None:           # auto-derived (cd_ff)
            continue
        if not (lo * 0.5 <= v <= hi * 1.5):
            return INVALID

    netlist = build_topology_netlist(topology, p)
    log_text, stderr = _run_ngspice(netlist)

    if "TIMEOUT" in stderr:
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False, error="TIMEOUT")

    if "simulation interrupted" in log_text.lower():
        err_snippet = next(
            (l.strip() for l in log_text.splitlines() if "error" in l.lower()), "sim error"
        )
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False, error=err_snippet[:120])

    ac_text    = _section(log_text, "ac")
    op_text    = _section(log_text, "op")
    noise_text = _section(log_text, "noise")
    s11_text   = _section(log_text, "s11")

    if not ac_text.strip():
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False,
                         error="No AC section in output, convergence failure")

    s21_db   = _parse_gain(ac_text, FREQ_HZ)
    if s21_db is None:
        return SimResult(s21_db=-999, nf_db=999, iip3_dbm=-999,
                         pdc_mw=999, s11_db=0, valid=False,
                         error="Could not parse AC gain from output file")

    pdc_mw   = _parse_pdc(op_text, p["vdd_v"])
    nf_db    = _parse_nf(noise_text, FREQ_HZ, s21_db)
    ids_ma   = pdc_mw / max(p["vdd_v"], 0.1)
    iip3_dbm = _estimate_iip3_analytical(p["w1_um"], p["l1_nm"], ids_ma)
    s11_db   = _parse_s11(s11_text, FREQ_HZ)

    return SimResult(
        s21_db=float(s21_db),
        nf_db=float(nf_db),
        iip3_dbm=float(iip3_dbm),
        pdc_mw=float(pdc_mw),
        s11_db=float(s11_db),
        valid=True,
    )
