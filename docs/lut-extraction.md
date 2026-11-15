# LUT extraction with ngspice

novaad sizes devices from **look-up tables**. These are tables of small-signal parameters over a grid of `vgs × vds × vsb × lch`, all characterized at one reference width `REF_WIDTH`. This page builds them for the 3.3 V devices of the open **GlobalFoundries gf180mcu** PDK with a single ngspice testbench.

!!! example "Notebook"
    Everything here is runnable in [`notebooks/01_pdk_ngspice_lut_extraction.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/01_pdk_ngspice_lut_extraction.ipynb). It writes `notebooks/luts/gf180_nch.csv` and `gf180_pch.csv`, which the [examples](examples/index.md) read.

## 1. Install the gf180mcu PDK with ciel

[ciel](https://github.com/fossi-foundation/ciel) is the FOSSi Foundation's PDK version manager, the successor of `volare`. It downloads prebuilt releases of [google/gf180mcu-pdk](https://github.com/google/gf180mcu-pdk) into a PDK root, by default `~/.ciel`.

```python
PDK_FAMILY = "gf180mcu"
PDK_VARIANT = "gf180mcuD"   # 5LM, 11 kÅ top metal: the variant used by most open-source tapeouts
PDK_VERSION = None          # None -> latest release, or pin a commit hash from `ciel ls-remote`
PDK_ROOT = Path("~/.ciel").expanduser()

versions = subprocess.run(
    ["ciel", "ls-remote", "--pdk-family", PDK_FAMILY], capture_output=True, text=True, check=True
).stdout.split()
version = PDK_VERSION or versions[0]
subprocess.run(
    ["ciel", "enable", "--pdk-root", str(PDK_ROOT), "--pdk-family", PDK_FAMILY, version],
    check=True,
)

MODELS = PDK_ROOT / PDK_VARIANT / "libs.tech" / "ngspice"
assert (MODELS / "sm141064.ngspice").is_file()
```

The 3.3 V devices `nfet_03v3` and `pfet_03v3` are defined in `sm141064.ngspice`. They are binned BSIM4 (level 54) models, selected by the `typical`, `ff`, `ss`, `fs` and `sf` corner libraries, and are instantiated as plain MOSFETs: `M1 d g s b nfet_03v3 w=... l=...`.

## 2. Install ngspice

=== "macOS"

    ```bash
    brew install ngspice
    ```

=== "Debian / Ubuntu"

    ```bash
    sudo apt install ngspice   # version >= 40
    ```

## 3. A minimal ngspice driver

All the notebooks share `notebooks/spice.py`, a small ngspice driver of about 50 lines:

| function | purpose |
|---|---|
| `header(corner, temp)` | title line, `.include design.ngspice`, `.lib sm141064.ngspice <corner>`, `.temp`, `.option savecurrents` |
| `run(netlist)` | writes the netlist to a fresh temp dir, runs `ngspice -b`, raises on errors, returns the dir |
| `read(path)` | loads a `wrdata` file (`set wr_singlescale`, `set wr_vecnames`) into a DataFrame; `@m1[gm]` becomes `m1.gm` |
| `bandwidth(f, gdb)` | `(A0 [dB], f−3dB, A0·f−3dB)` of a low-pass response |
| `slew_rate(t, v, t0, t1)` | slope between 40 % and 60 % of a transition |

```python
def header(corner: str = "typical", temp: float = 27) -> str:
    return (
        "* novaad testbench\n"  # SPICE ignores the first line: it is the title
        f".include {MODELS / 'design.ngspice'}\n"
        f".lib {MODELS / 'sm141064.ngspice'} {corner}\n"
        f".temp {temp}\n"
        ".option savecurrents\n"
    )


def run(netlist: str) -> Path:
    wd = Path(mkdtemp(prefix="novaad_spice_"))
    (wd / "tb.spice").write_text(netlist)
    proc = subprocess.run(["ngspice", "-b", "tb.spice"], cwd=wd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (wd / "tb.log").write_text(log)
    if proc.returncode != 0 or "error" in log.lower():
        raise RuntimeError(f"ngspice failed in {wd}:\n" + "\n".join(log.splitlines()[-20:]))
    return wd
```

## 4. What goes into the LUT

| column | meaning | source |
|---|---|---|
| `vgs vds vsb lch` | grid axes | sweep |
| `ids gm gds` | drain current and conductances | BSIM4 operating point |
| `cgg cgs cgd` | gate capacitances (intrinsic + overlap) | `cgg+cgso+cgdo`, `−cgs+cgso`, `−cgd+cgdo` |
| `cdb csb` | junction capacitances | `capbd`, `capbs` |
| `gmoverid jd ft av ron fom_bw` | derived figures | `gm/ids`, `ids/W`, `gm/(2π·cgg)`, `gm/gds`, `1/gds`, `ft·av` |

!!! info "BSIM4 capacitances in ngspice"
    ngspice reports the off-diagonal BSIM4 capacitances as **transcapacitances**, `cij = −∂Qi/∂Vj`, which is why `cgs` and `cgd` are negated. BSIM4's `cgg` is intrinsic only, so the overlap terms `cgso` and `cgdo` are added back.

PMOS is swept with negated voltages. novaad takes `abs()` of the whole LUT, so both device types end up with positive values.

## 5. The testbench

One ngspice run per `(lch, vsb)` pair. Each run does a nested `vgs × vds` DC sweep of a single transistor and writes the operating-point parameters with `wrdata`:

```python
# sweep settings
VDD = 3.3                                   # sweep range for |vgs| and |vds| (3.3 V devices)
REF_WIDTH = 10e-6                           # characterization width; novaad scales W from it
LCH = [0.28e-6, 0.5e-6, 1e-6, 2e-6, 4e-6]   # channel lengths (0.28 um is the 3.3 V minimum)
VSB = [0.0, 0.4, 0.8, 1.2]                  # source-bulk voltages
VGS_STEP = 0.05
VDS_STEP = 0.1
CORNER = "typical"
TEMP = 27

DEVICES = {"nch": "nfet_03v3", "pch": "pfet_03v3"}
PARAMS = ["id", "gm", "gds", "cgg", "cgs", "cgd", "cgso", "cgdo", "capbd", "capbs"]


def testbench(model: str, lch: float, vsb: float, sign: int) -> str:
    # sign=+1 for nmos, -1 for pmos (source grounded, all voltages negated)
    vecs = " ".join(["v(g)", "v(d)"] + [f"@m1[{p}]" for p in PARAMS])
    return spice.header(CORNER, TEMP) + f'''
* {model} LUT sweep: L={lch}, VSB={vsb}
VG g 0 0
VD d 0 0
VB b 0 {-sign * vsb}
M1 d g 0 b {model} w={REF_WIDTH} l={lch}
.control
set wr_singlescale
set wr_vecnames
save {vecs}
dc VG 0 {sign * VDD} {sign * VGS_STEP} VD {sign * VDS_STEP} {sign * VDD} {sign * VDS_STEP}
wrdata lut.txt {vecs}
.endc
.end
'''
```

For `nfet_03v3` at `L = 0.28 µm`, `VSB = 0` this produces:

```spice
* novaad testbench
.include ~/.ciel/gf180mcuD/libs.tech/ngspice/design.ngspice
.lib ~/.ciel/gf180mcuD/libs.tech/ngspice/sm141064.ngspice typical
.temp 27
.option savecurrents

* nfet_03v3 LUT sweep: L=2.8e-07, VSB=0.0
VG g 0 0
VD d 0 0
VB b 0 -0.0
M1 d g 0 b nfet_03v3 w=1e-05 l=2.8e-07
.control
set wr_singlescale
set wr_vecnames
save v(g) v(d) @m1[id] @m1[gm] @m1[gds] @m1[cgg] @m1[cgs] @m1[cgd] @m1[cgso] @m1[cgdo] @m1[capbd] @m1[capbs]
dc VG 0 3.3 0.05 VD 0.1 3.3 0.1
wrdata lut.txt v(g) v(d) @m1[id] @m1[gm] @m1[gds] @m1[cgg] @m1[cgs] @m1[cgd] @m1[cgso] @m1[cgdo] @m1[capbd] @m1[capbs]
.endc
.end
```

!!! warning "Start the vds sweep above zero"
    The sweep starts at `|vds| = VDS_STEP`. At `vds = 0` the drain current is zero, so `gm/id` is infinite and breaks the interpolation.

## 6. Extract and write the LUTs

```python
from math import pi
from pandas import DataFrame, concat


def extract(dev: str) -> DataFrame:
    frames = []
    for lch in LCH:
        for vsb in VSB:
            wd = spice.run(testbench(DEVICES[dev], lch, vsb, +1 if dev == "nch" else -1))
            raw = spice.read(wd / "lut.txt")
            frames.append(DataFrame({
                "vgs": raw["v(g)"], "vds": raw["v(d)"], "vsb": vsb, "lch": lch,
                "ids": raw["m1.id"], "gm": raw["m1.gm"], "gds": raw["m1.gds"],
                "cgg": raw["m1.cgg"] + raw["m1.cgso"] + raw["m1.cgdo"],
                "cgs": -raw["m1.cgs"] + raw["m1.cgso"],
                "cgd": -raw["m1.cgd"] + raw["m1.cgdo"],
                "cdb": raw["m1.capbd"], "csb": raw["m1.capbs"],
            }))
    lut = concat(frames, ignore_index=True).abs()
    lut[["vgs", "vds"]] = lut[["vgs", "vds"]].round(6)   # exact grid values (the GUI needs them)
    lut = lut[lut["ids"] > 0].copy()
    lut["gmoverid"] = lut["gm"] / lut["ids"]
    lut["jd"] = lut["ids"] / REF_WIDTH
    lut["ft"] = lut["gm"] / (2 * pi * lut["cgg"])
    lut["av"] = lut["gm"] / lut["gds"]
    lut["ron"] = 1 / lut["gds"]
    lut["fom_bw"] = lut["ft"] * lut["av"]
    return lut


OUT_DIR = Path("luts")
OUT_DIR.mkdir(exist_ok=True)
for dev in DEVICES:
    extract(dev).to_csv(OUT_DIR / f"gf180_{dev}.csv", index=False)
```

Then register the tables with the CLI:

```toml title="pyproject.toml"
[tool.novaad.nch]
lut_path = "notebooks/luts/gf180_nch.csv"
ref_width = 10e-6

[tool.novaad.pch]
lut_path = "notebooks/luts/gf180_pch.csv"
ref_width = 10e-6
```

## 7. Sanity check

Load the tables with novaad and plot the classic gm/id charts at `|vds| = VDD/2` and `vsb = 0`, one curve per channel length:

```python
from novaad import Device

nch = Device("luts/gf180_nch.csv", device_type="nch", ref_width=REF_WIDTH)
lut = nch.lut
vds = lut["vds"].unique()[abs(lut["vds"].unique() - VDD / 2).argmin()]
for lch in LCH:
    curve = lut.query("abs(vds - @vds) < 1e-6 and vsb == 0 and abs(lch - @lch) < 1e-9").sort_values("vgs")
    ...  # plot gmoverid vs vgs, jd / ft / av vs gmoverid
```

In weak inversion, NMOS gm/id should approach roughly 25–30 V⁻¹. Short-channel `ft` should reach the GHz range. The `novaad --gui --type=nch` explorer shows the same charts interactively.
