# Quick start

This page sizes an NMOS transistor twice: from Python and from the CLI. It uses the test LUT shipped in the repository, `test/data/test_nch_lut_renamed.csv`, which is a 180 nm NMOS characterized at `W = 10 µm`. To use your own process, extract the LUTs first ([LUT extraction](lut-extraction.md)).

## 1. Python API

### Load a device

A `Device` wraps one LUT. It reads `.csv` and `.h5` files and takes `abs()` of every column, so PMOS tables can be stored with negative voltages and currents.

```python
from novaad import Device, DeviceSizingSpecification

nch = Device(
    "test/data/test_nch_lut_renamed.csv",
    device_type="nch",      # "nch" or "pch"
    ref_width=10e-6,        # width the LUT was characterized at
)
nch.lut.columns.tolist()
# ['vgs', 'lch', 'vds', 'vsb', 'gm', 'ids', 'gds', 'cgg', 'gmoverid', 'jd', 'ft', 'av', 'fom_bw', 'ron', 'cgd', 'cgs', 'cdb', 'csb']
```

### Size it: the forward gm/id problem

State the operating point, the channel length, the gm/id (inversion level) and the current, or `gm` instead of `ids`. novaad returns the width and, with `return_dcop=True`, the operating point it settled on, including the `vgs` that gives the requested gm/id.

```python
spec = DeviceSizingSpecification(vds=0.6, vsb=0.0, lch=180e-9, gmoverid=16.0, ids=100e-6)
dcop, sizing = nch.sizing(spec, return_dcop=True)

dcop.to_df()
#      vgs  vds  vsb     ids
# 0  0.575  0.6  0.0  0.0001

sizing.to_df()
#             lch       wch
# 0  1.800000e-07  0.000014
```

### Characterize it: the reverse gm/id problem

From an operating point and a sizing, `electric_model` predicts the small-signal model. You can use it to check gain, bandwidth and parasitic capacitances before running a simulator.

```python
em = nch.electric_model(dcop, sizing)
em.gm[0], em.gds[0], em.av[0], em.ft[0]
# (0.002025, 5.017e-05, 40.47, 1.3e10)    gm = 2.0 mS, gm/gds ≈ 40 V/V, fT = 13 GHz
em.cgg[0], em.cgs[0], em.cgd[0], em.cdb[0], em.csb[0]
```

Each field of `DcOp`, `Sizing` and `ElectricModel` is a list, because every call accepts vectors of targets and returns one row per target. `.to_df()` turns them into pandas DataFrames.

### Look up any LUT column

`look_up` interpolates arbitrary output columns at arbitrary target points:

```python
nch.look_up(
    ["av", "jd", "ft"],
    {"vgs": [0.7], "vds": [0.9], "vsb": [0.0], "lch": [180e-9], "gmoverid": [10.0]},
)
#         av        jd            ft
# 1  40.5005  28.19065  2.400000e+10
```

!!! tip "Inside an optimization loop"
    `sizing()` and `electric_model()` are fast table operations, so they can run inside a circuit optimizer. The [examples](examples/index.md) pair them with ngspice and pymoo to size whole amplifiers automatically.

!!! note "vgs is not a priority target"
    Unlike `gmoverid`, `vgs` is not prioritized during the look-up. The nearest-row search and the interpolation can settle on a slightly different operating point, so that gm/id, `vds`, `lch` and `vsb` are met together. Always read the final `vgs` back from the returned `DcOp`.

## 2. Command line

### Point the CLI at your LUTs

The CLI reads its configuration from the `[tool.novaad]` table of a `pyproject.toml`. By default it uses the one in the current directory; `--config` selects another.

```toml title="pyproject.toml"
[tool.novaad.nch]
lut_path = "test/data/test_nch_lut_renamed.csv"
ref_width = 10e-6

# [tool.novaad.pch]
# lut_path = "path/to/pch_lut.csv"
# ref_width = 10e-6
```

### Describe the instances to size

```toml title="input.toml"
[device]
m0 = { type = "nch", vds = 0.5, vsb = 0.0, lch = 180e-9, gmid = 26, gm = 1e-3 }
m2 = { type = "nch", vgs = 0.8, vds = 0.5, vsb = 0.0, lch = 180e-9, gmid = 10, ids = 500e-6 }
m3 = { type = "nch", vgs = 0.8, vds = 0.5, vsb = 0.0, lch = 1e-6, wch = 12e-6, ids = 100e-6 }
```

`m0` and `m2` are sized: novaad finds `W`. `m3` gives `wch`, so it is characterized instead.

### Run it

```bash
novaad device --input=input.toml
```

```text
                          Device Sizing Results
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Parameter              ┃         m0 ┃         m2 ┃               m3 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Type                   │        nch │        nch │              nch │
│ Vgs [V]                │       0.38 │       0.70 │             0.80 │
│ Vds [V]                │       0.50 │       0.50 │             0.50 │
│ Wch [um]               │   380.5633 │    16.2422 │          12.0000 │
│ Lch [um]               │     0.1800 │     0.1900 │           1.0000 │
│ Cgg [fF]               │   456.6760 │    31.3474 │          83.4000 │
│ Objective              │     sizing │     sizing │ characterization │
│ Gm/Id                  │    39.4750 │     9.0469 │           7.9500 │
│ Gm [uS]                │  1545.0870 │  4523.4390 │         795.0000 │
│ Av [dB]                │    32.6913 │    28.7616 │          41.0860 │
│ Ft [GHz]               │     0.5385 │    23.0000 │           1.5200 │
│ ...                    │        ... │        ... │              ... │
└────────────────────────┴────────────┴────────────┴──────────────────┘
```

Add `--output=results.csv` to save the table, or `--gui` to open it in the browser. To explore the LUT interactively instead, run `novaad --gui --type=nch`. The [CLI reference](cli.md) lists every option and input section.
