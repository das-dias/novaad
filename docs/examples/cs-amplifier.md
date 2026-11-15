# Common-source amplifier

A single-stage amplifier designed by hand with the gm/id method: novaad sizes it, and ngspice verifies it with operating point, AC, transient and noise simulations.

[:material-notebook: `notebooks/02_cs_amplifier_design.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/02_cs_amplifier_design.ipynb){ .md-button }

## Architecture

<!-- Architecture figure: add docs/figs/architectures/cs_amplifier.svg and uncomment
![Active-load common-source amplifier](../figs/architectures/cs_amplifier.svg)
-->

```text
        VDD
         │
VBP ──┤ M2  (pfet_03v3, current-source load)
         │
         ├──── out ──┬── CL
         │           │
 in ──┤ M1          ═╧═
         │  (nfet_03v3)
        GND
```

An NMOS common-source transistor `M1`, loaded by a PMOS current source `M2`, drives a capacitive load `CL`. The PMOS gate is held at the ideal bias `VBP = VDD − Vsg2` predicted by novaad.

## Specifications

```python
# ── Design specifications ───────────────────────────────────────────────────
VDD = 3.3                # supply [V]
CL = 0.5e-12             # load capacitance [F]
GBW_MIN = 470e6          # gain-bandwidth product [Hz]
GAIN_MIN_DB = 33         # DC gain [dB]
INPUT_NOISE_MAX = 20e-6  # input-referred noise [Vrms]: output noise (NOISE_FMIN..GBW) / DC gain
NOISE_FMIN = 1.0         # lower bound of the noise integration [Hz]
POWER_MAX = 500e-6       # power budget [W]
VOUT_DC = VDD / 2        # output DC operating point [V]

# ── Design choices (the knobs of the gm/id method) ──────────────────────────
GBW_MARGIN = 1.2         # design for GBW_MIN·margin: covers the parasitic output capacitance
L_N = 0.5e-6             # M1 channel length [m]
L_P = 0.5e-6             # M2 channel length [m]
GMID_N = 16              # M1 gm/id [1/V]: moderate inversion, good gm per current
GMID_P = 8               # M2 gm/id [1/V]: stronger inversion, low gm (noise) and good matching
```

## Sizing with novaad

The gm/id design flow:

1. **GBW** fixes `gm1 = 2π·GBW·CL`.
2. The chosen **gm/id of M1** sets the inversion level, so `Ids = gm1 / (gm/id)`. novaad returns `W1` and `Vgs1`.
3. **M2** carries the same `Ids` at its own gm/id. novaad returns `W2` and `Vsg2`, which gives the load bias `VBP = VDD − Vsg2`.
4. novaad's electric model predicts the gain `gm1/(gds1 + gds2)` and the output capacitance before any simulation.

```python
from math import pi
from novaad import Device, DeviceSizingSpecification

nch = Device("luts/gf180_nch.csv", device_type="nch", ref_width=10e-6)
pch = Device("luts/gf180_pch.csv", device_type="pch", ref_width=10e-6)

gm1 = 2 * pi * GBW_MIN * GBW_MARGIN * CL

dcop_n, size_n = nch.sizing(
    DeviceSizingSpecification(vds=VOUT_DC, vsb=0.0, lch=L_N, gmoverid=GMID_N, gm=gm1),
    return_dcop=True,
)
IDS = dcop_n.ids[0]
dcop_p, size_p = pch.sizing(
    DeviceSizingSpecification(vds=VDD - VOUT_DC, vsb=0.0, lch=L_P, gmoverid=GMID_P, ids=IDS),
    return_dcop=True,
)
W_N, W_P = size_n.wch[0], size_p.wch[0]
VGS_N, VSG_P = dcop_n.vgs[0], dcop_p.vgs[0]

# reverse gm/id: predict gain, output capacitance and GBW before simulating
em_n = nch.electric_model(dcop_n, size_n)
em_p = pch.electric_model(dcop_p, size_p)
gain_pred = em_n.gm[0] / (em_n.gds[0] + em_p.gds[0])
c_out = CL + em_n.cdb[0] + em_p.cdb[0] + em_n.cgd[0] + em_p.cgd[0]
gbw_pred = em_n.gm[0] / (2 * pi * c_out)
```

```text
predicted DC gain : 41.9 dB   (spec >= 33 dB)
predicted GBW     : 451.1 MHz (spec >= 470 MHz, Cout = 625 fF)
predicted power   : 398.9 uW  (spec <= 500 uW)
```

The electric model already shows that the parasitic drain capacitances (125 fF on top of `CL`) cost more GBW than the 1.2× margin covers.

## Testbench

```python
def netlist(vin: float, control: str, source: str | None = None) -> str:
    return spice.header() + f'''
* active-load common-source amplifier
VDD vdd 0 {VDD}
VIN in 0 {source or f"dc {vin} ac 1"}
VBP vbp 0 {VDD - VSG_P}
{fet("M1", "out", "in", "0", "0", "nfet_03v3", W_N, L_N)}
{fet("M2", "out", "vbp", "vdd", "vdd", "pfet_03v3", W_P, L_P)}
CL out 0 {CL}
.control
set wr_singlescale
set wr_vecnames
{control}
.endc
.end
'''
```

`control` holds the analysis (`op`, `dc`, `ac`, `tran`, `noise`), and `source` replaces the input waveform for the transient runs. The generated device lines are:

```spice
VIN in 0 dc 0.75 ac 1
VBP vbp 0 2.3
M1 out in 0 0 nfet_03v3 w=9.4005e-06 l=5e-07 m=9
M2 out vbp vdd vdd pfet_03v3 w=9.65798e-06 l=5e-07 m=6
CL out 0 5e-13
```

## Verification

### DC bias

The output of a high-gain stage is very sensitive to its input bias: one LSB of model error moves `vout` to a rail. A fine DC transfer sweep therefore finds the `vin` that puts `vout` at `VOUT_DC`. A real circuit would set it with feedback.

```python
dc = spice.read(spice.run(netlist(0.0, f"dc VIN 0 {VDD} 0.2m\nwrdata dc.txt v(out)")) / "dc.txt")
VIN_DC = float(np.interp(VOUT_DC, dc["v(out)"][::-1], dc["v-sweep"][::-1]))   # vout falls with vin
```

```text
input bias for vout = 1.650 V: vin = 750.0 mV (novaad predicted Vgs = 750 mV)
```

The novaad prediction of `Vgs1` matches the simulated bias exactly.

### DC operating point

```python
from utils import dc_tables, dc_vectors

bench = netlist(VIN_DC, "op")
op = spice.read(spice.run(netlist(VIN_DC, f"op\nwrdata op.txt {dc_vectors(bench)}")) / "op.txt")
devices, nodes = dc_tables(bench, op)
```

| element | I [µA] | vgs [V] | vds [V] | vth [V] | vdsat [V] | gm [µS] | gds [µS] | region |
|---|---|---|---|---|---|---|---|---|
| M1 | 120.9 | 0.750 | 1.650 | 0.712 | 0.108 | 1772.5 | 9.94 | saturation |
| M2 | 120.9 | 1.000 | 1.650 | 0.788 | 0.215 | 962.5 | 4.28 | saturation |

### AC, transient and noise

```python
ac = spice.read(spice.run(netlist(VIN_DC, "ac dec 50 1k 10G\nwrdata ac.txt vdb(out) vp(out)")) / "ac.txt")
gain_db, f3db, gbw = spice.bandwidth(ac["frequency"], ac["vdb(out)"])   # GBW = A0·f-3dB
```

The transient tests are:

- a 1 mV sine at 20 kHz;
- a 0 → 3.3 V ramp in 200 µs, which traces the whole transfer curve rail to rail;
- a full-supply pulse that measures the slew rate.

When the input falls, M1 turns off and only the PMOS current charges the output, so the rising slew rate should be close to `Ids/Cout`.

Noise is integrated from `NOISE_FMIN` to the GBW and referred to the input through the DC gain. The novaad estimate counts thermal noise only, `S_in = 4kT·γ·(1 + gm2/gm1)/gm1` over the noise bandwidth `π/2·f−3dB`. The gap to ngspice is flicker noise.

### Results

| metric | spec | novaad prediction | ngspice |
|---|---|---|---|
| DC gain [dB] | ≥ 33 | 41.9 | **41.91** |
| GBW [MHz] | ≥ 470 | 451.1 | **450.1** :material-close: |
| −3 dB bandwidth [kHz] | | | 3613 |
| Transient gain at 20 kHz [dB] | = AC gain | | 41.90 |
| Slew rate, output rising [V/µs] | | 193.4 (`Ids/Cout`) | 263.7 |
| Slew rate, output falling [V/µs] | | | 3587 |
| Input-referred noise [µVrms] | ≤ 20 | 7.40 (thermal only) | **14.28** |
| Power [µW] | ≤ 500 | 398.9 | 399 |
| All devices saturated | yes | | yes |

!!! tip "Closing the GBW gap"
    The miss on GBW was predicted before simulating: novaad's electric model gives 451 MHz, and ngspice measures 450 MHz. Raising `GBW_MARGIN` to about 1.25 covers the drain parasitics at the cost of a little more power.
