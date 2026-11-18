# LUT extraction with ngspice

novaad sizes devices from **look-up tables**. These are tables of small-signal parameters over a grid of `vgs × vds × vsb × lch`, all characterized at one reference width `REF_WIDTH`. This page builds them for the 1.2 V core devices of the open **IHP SG13G2** 130 nm BiCMOS PDK with a single ngspice testbench.

!!! example "Notebook"
    Everything here is runnable in [`notebooks/01_pdk_ngspice_lut_extraction.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/01_pdk_ngspice_lut_extraction.ipynb). It writes `notebooks/luts/sg13_lv_nch.csv` and `sg13_lv_pch.csv`, which the [examples](examples/index.md) read.

!!! note "Why not gf180mcu?"
    The open gf180mcu PDK has no 1.8 V core devices. Its shortest MOSFETs are the 3.3 V `nfet_03v3` / `pfet_03v3`, binned from L = 0.28 µm, and `l=0.18u` fails in ngspice with *could not find a valid modelname*. This is tracked in [gdsfactory/gf180mcu#205](https://github.com/gdsfactory/gf180mcu/issues/205). IHP SG13G2's `sg13_lv_nmos` / `sg13_lv_pmos` go down to L = 130 nm.

## 1. Install the IHP SG13G2 PDK with ciel

[ciel](https://github.com/fossi-foundation/ciel) is the FOSSi Foundation's PDK version manager, the successor of `volare`. It downloads prebuilt releases of [IHP-GmbH/IHP-Open-PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) into a PDK root, by default `~/.ciel`.

```python
PDK_FAMILY = "ihp-sg13g2"
PDK_VARIANT = "ihp-sg13g2"  # SG13G2 has a single variant
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
assert (MODELS / "models" / "cornerMOSlv.lib").is_file()
```

The low-voltage MOSFETs `sg13_lv_nmos` and `sg13_lv_pmos` (1.2 V, L ≥ 130 nm) are defined in `cornerMOSlv.lib`, with the corners `mos_tt`, `mos_ss`, `mos_ff`, `mos_sf` and `mos_fs`. Each one is a subcircuit around a single **PSP103** device. Instances therefore use `X`: `XM1 d g s b sg13_lv_nmos w=... l=... ng=...`, where `w` is the total width, split over `ng` fingers.

## 2. Install ngspice and compile the OSDI models

=== "macOS"

    ```bash
    brew install ngspice
    ```

=== "Debian / Ubuntu"

    ```bash
    sudo apt install ngspice   # version >= 40, built with OSDI support
    ```

PSP103 is a Verilog-A compact model. ngspice loads it as a compiled **OSDI** library, and the PDK ships the Verilog-A sources. [OpenVAF-Reloaded](https://github.com/OpenVAF/OpenVAF-Reloaded) compiles them:

```bash
cd ~/.ciel/ihp-sg13g2/libs.tech/verilog-a/psp103
mkdir -p ../../ngspice/osdi
openvaf-r psp103.va --output ../../ngspice/osdi/psp103.osdi
openvaf-r psp103_nqs.va --output ../../ngspice/osdi/psp103_nqs.osdi
```

The notebook does this for you. If `openvaf-r` isn't on `PATH`, it downloads the release build for your platform to `~/.local/opt/openvaf-r`.

!!! warning "macOS: openvaf-r is killed at launch"
    The macOS release bundles `libLLVM`, `libz3` and `libzstd` with invalid code signatures, so macOS kills `openvaf-r` with exit code 137. Re-sign them ad hoc, which the notebook also does:

    ```bash
    codesign --force -s - ~/.local/opt/openvaf-r/lib/*.dylib
    ```

## 3. A minimal ngspice driver

All the notebooks share `notebooks/spice.py`, a small ngspice driver of about 80 lines:

| function | purpose |
|---|---|
| `header(corner, temp)` | title line, `.lib cornerMOSlv.lib <corner>`, `.temp`, `.option savecurrents` |
| `run(netlist)` | writes the netlist and a `.spiceinit` that loads the OSDI models to a fresh temp dir, runs `ngspice -b`, raises on errors, returns the dir |
| `mos(inst, model, *params)` | PSP103 operating-point vectors of a transistor: `mos("xm1", NMOS, "gm")` → `@n.xm1.nsg13_lv_nmos[gm]` |
| `read(path)` | loads a `wrdata` file (`set wr_singlescale`, `set wr_vecnames`) into a DataFrame; `@n.xm1.nsg13_lv_nmos[gm]` becomes `xm1.gm` |
| `bandwidth(f, gdb)` | `(A0 [dB], f−3dB, A0·f−3dB)` of a low-pass response |
| `slew_rate(t, v, t0, t1)` | slope between 40 % and 60 % of a transition |

```python
OSDI = ("psp103", "psp103_nqs")
NMOS, PMOS = "sg13_lv_nmos", "sg13_lv_pmos"


def header(corner: str = "mos_tt", temp: float = 27) -> str:
    return (
        "* novaad testbench\n"  # SPICE ignores the first line: it is the title
        f".lib {MODELS / 'models' / 'cornerMOSlv.lib'} {corner}\n"
        f".temp {temp}\n"
        ".option savecurrents\n"
    )


def run(netlist: str) -> Path:
    wd = Path(mkdtemp(prefix="novaad_spice_"))
    # ngspice reads ./.spiceinit at start-up: it loads the OSDI models before the netlist is parsed
    (wd / ".spiceinit").write_text("".join(f"osdi '{MODELS / 'osdi' / m}.osdi'\n" for m in OSDI))
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
| `ids gm gds` | drain current and conductances | PSP103 operating point |
| `cgg cgs cgd` | gate capacitances (intrinsic + overlap) | `cgg+cgsol+cgdol`, `cgs+cgsol`, `cgd+cgdol` |
| `cdb csb` | junction capacitances | `cjd`, `cjs` |
| `gmoverid jd ft av ron fom_bw` | derived figures | `gm/ids`, `ids/W`, `gm/(2π·cgg)`, `gm/gds`, `1/gds`, `ft·av` |

!!! info "PSP103 capacitances"
    PSP103 reports its intrinsic capacitances as positive values, and the gate overlap capacitances separately as `cgsol` and `cgdol`. Its `cgg` is intrinsic only, so the overlaps are added back.

PMOS is swept with negated voltages. novaad takes `abs()` of the whole LUT, so both device types end up with positive values.

## 5. The testbench

One ngspice run per `(lch, vsb)` pair. Each run does a nested `vgs × vds` DC sweep of a single transistor and writes the operating-point parameters with `wrdata`:

```python
# sweep settings
VDD = 1.2                                   # sweep range for |vgs| and |vds| (1.2 V core devices)
REF_WIDTH = 10e-6                           # characterization width; novaad scales W from it
LCH = [0.13e-6, 0.18e-6, 0.25e-6, 0.5e-6, 1e-6, 2e-6]   # channel lengths (0.13 um is the minimum)
VSB = [0.0, 0.2, 0.4, 0.6]                  # source-bulk voltages
VGS_STEP = 0.025
VDS_STEP = 0.05
CORNER = "mos_tt"
TEMP = 27

DEVICES = {"nch": spice.NMOS, "pch": spice.PMOS}
PARAMS = ["ids", "gm", "gds", "cgg", "cgs", "cgd", "cgsol", "cgdol", "cjd", "cjs"]


def testbench(model: str, lch: float, vsb: float, sign: int) -> str:
    # sign=+1 for nmos, -1 for pmos (source grounded, all voltages negated)
    vecs = " ".join(["v(g)", "v(d)", spice.mos("xm1", model, *PARAMS)])
    return spice.header(CORNER, TEMP) + f'''
* {model} LUT sweep: L={lch}, VSB={vsb}
VG g 0 0
VD d 0 0
VB b 0 {-sign * vsb}
XM1 d g 0 b {model} w={REF_WIDTH} l={lch} ng=1
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

For `sg13_lv_nmos` at `L = 0.13 µm`, `VSB = 0` this produces:

```spice
* novaad testbench
.lib ~/.ciel/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib mos_tt
.temp 27
.option savecurrents

* sg13_lv_nmos LUT sweep: L=1.3e-07, VSB=0.0
VG g 0 0
VD d 0 0
VB b 0 -0.0
XM1 d g 0 b sg13_lv_nmos w=1e-05 l=1.3e-07 ng=1
.control
set wr_singlescale
set wr_vecnames
save v(g) v(d) @n.xm1.nsg13_lv_nmos[ids] @n.xm1.nsg13_lv_nmos[gm] ... @n.xm1.nsg13_lv_nmos[cjs]
dc VG 0 1.2 0.025 VD 0.05 1.2 0.05
wrdata lut.txt v(g) v(d) @n.xm1.nsg13_lv_nmos[ids] @n.xm1.nsg13_lv_nmos[gm] ... @n.xm1.nsg13_lv_nmos[cjs]
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
                "ids": raw["xm1.ids"], "gm": raw["xm1.gm"], "gds": raw["xm1.gds"],
                "cgg": raw["xm1.cgg"] + raw["xm1.cgsol"] + raw["xm1.cgdol"],
                "cgs": raw["xm1.cgs"] + raw["xm1.cgsol"],
                "cgd": raw["xm1.cgd"] + raw["xm1.cgdol"],
                "cdb": raw["xm1.cjd"], "csb": raw["xm1.cjs"],
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
    extract(dev).to_csv(OUT_DIR / f"sg13_lv_{dev}.csv", index=False)
```

To distribute the tables, convert them to the compact protobuf format, which is about 6× smaller (see [Binary LUTs](cli.md#binary-luts)):

```bash
python -m schema.io luts/sg13_lv_nch.csv luts/sg13_lv_nch.pb --type nch --ref-width 10e-6
```

Then register the tables with the CLI:

```toml title="pyproject.toml"
[tool.novaad.nch]
lut_path = "notebooks/luts/sg13_lv_nch.csv"
ref_width = 10e-6

[tool.novaad.pch]
lut_path = "notebooks/luts/sg13_lv_pch.csv"
ref_width = 10e-6
```

## 7. Sanity check

Load the tables with novaad and plot the classic gm/id charts at `|vds| = VDD/2` and `vsb = 0`, one curve per channel length:

```python
from novaad import Device

nch = Device("luts/sg13_lv_nch.csv", device_type="nch", ref_width=REF_WIDTH)
lut = nch.lut
vds = lut["vds"].unique()[abs(lut["vds"].unique() - VDD / 2).argmin()]
for lch in LCH:
    curve = lut.query("abs(vds - @vds) < 1e-6 and vsb == 0 and abs(lch - @lch) < 1e-9").sort_values("vgs")
    ...  # plot gmoverid vs vgs, jd / ft / av vs gmoverid
```

In weak inversion, NMOS gm/id should approach roughly 25–30 V⁻¹. At the 130 nm minimum length, NMOS `ft` reaches about 70 GHz and PMOS about 30 GHz. The `novaad --gui --type=nch` explorer shows the same charts interactively.
