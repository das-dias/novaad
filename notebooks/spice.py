"""Minimal ngspice driver shared by the example notebooks."""

import os
import subprocess
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from pandas import DataFrame, read_csv

PDK_ROOT = Path(os.environ.get("PDK_ROOT", Path.home() / ".ciel"))
PDK = os.environ.get("PDK", "gf180mcuD")
MODELS = PDK_ROOT / PDK / "libs.tech" / "ngspice"


def header(corner: str = "typical", temp: float = 27) -> str:
    """Model includes for the gf180mcu 3.3 V devices (nfet_03v3 / pfet_03v3)."""
    return (
        "* novaad testbench\n"  # SPICE ignores the first line: it is the title
        f".include {MODELS / 'design.ngspice'}\n"
        f".lib {MODELS / 'sm141064.ngspice'} {corner}\n"
        f".temp {temp}\n"
        ".option savecurrents\n"
    )


def run(netlist: str) -> Path:
    """Run `ngspice -b` in a fresh temp dir; files written by `wrdata` land there."""
    wd = Path(mkdtemp(prefix="novaad_spice_"))
    (wd / "tb.spice").write_text(netlist)
    proc = subprocess.run(
        ["ngspice", "-b", "tb.spice"], cwd=wd, capture_output=True, text=True
    )
    log = proc.stdout + proc.stderr
    (wd / "tb.log").write_text(log)
    if proc.returncode != 0 or "error" in log.lower():
        tail = "\n".join(log.splitlines()[-20:])
        raise RuntimeError(f"ngspice failed in {wd}:\n{tail}")
    return wd


def read(path: Path) -> DataFrame:
    """Read a `wrdata` file written with `set wr_singlescale` and `set wr_vecnames`."""
    df = read_csv(path, sep=r"\s+")
    # device parameters "@m1[gm]" -> "m1.gm"; node voltages/currents keep their name
    df.columns = [c[1:].replace("[", ".").rstrip("]") if c.startswith("@") else c for c in df.columns]
    return df


def bandwidth(f, gdb) -> tuple[float, float, float]:
    """(DC gain [dB], -3 dB frequency [Hz], gain-bandwidth product A0·f-3dB [Hz]) of a low-pass response.

    GBW is computed as A0·f-3dB rather than read at the 0 dB crossing: capacitive feedthrough
    (e.g. a large Cgd) flattens |H| near 0 dB at high frequency and pushes that crossing far out.
    """
    f, gdb = np.asarray(f), np.asarray(gdb)
    a0 = gdb[0]
    i = int(np.argmax(gdb < a0 - 3))   # first point below the -3 dB level
    if i == 0:
        raise ValueError("no -3 dB corner inside the AC sweep")
    f3 = float(np.exp(np.interp(a0 - 3, [gdb[i], gdb[i - 1]], np.log([f[i], f[i - 1]]))))
    return float(a0), f3, 10 ** (a0 / 20) * f3


def slew_rate(t, v, t0: float, t1: float) -> float:
    """Slew rate [V/s] of the transition of `v` inside [t0, t1]: its slope between 40 % and 60 % of the swing.

    Measuring at mid-swing ignores both the kick a fast input edge couples to the output through Cgd
    (at the start) and the linear-settling tail (at the end).
    """
    t, v = np.asarray(t), np.asarray(v)
    inside = (t >= t0) & (t <= t1)
    t, v = t[inside], v[inside]
    frac = (v - v[0]) / (v[-1] - v[0])
    t40, t60 = t[np.argmax(frac >= 0.4)], t[np.argmax(frac >= 0.6)]
    return 0.2 * abs(v[-1] - v[0]) / (t60 - t40)
