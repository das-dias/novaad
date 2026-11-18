# Voltage buffer with R-NSGA-III

A unity-gain voltage buffer: a 5-transistor OTA with a PMOS current-mirror load, wired as a follower. novaad and ngspice run inside **pymoo's R-NSGA-III**, which balances gain, bandwidth, power, offset, noise and linear input range against area and the Schreier figure of merit.

[:material-notebook: `notebooks/04_buffer_widlar_rnsga3.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/04_buffer_widlar_rnsga3.ipynb){ .md-button }

## Architecture

<!-- Architecture figure: add docs/figs/architectures/voltage_buffer.svg and uncomment
![OTA follower voltage buffer](../figs/architectures/voltage_buffer.svg)
-->

```text
            VDD ──┬──────────────┬──────────────────┬──
                  │              │                  │
            M3 ├──┤(diode)  ┌─┤ M4                IREF_N ↓
                  │         │    │                  │
                outn ───────┘   outp ──┬── out      nb ──┬──────┐
                  │              │     │          M6 ├──┘ (diode)
     vin ─────┤ M1          M2 ├─┘     CL                       │
                  └──────┬───────┘ (gate = drain)               │
                       tail                                     │
                         └──────────────────────────────────┤ M5 (tail)
                                                               GND
```

- **M1/M2:** NMOS differential pair. The signal drives the positive input (M1's gate).
- **M3/M4:** PMOS current-mirror load. M3 is diode-connected on `outn`, and M4 mirrors its current onto `outp`.
- **Unity feedback:** the output `outp` drives the negative input (M2's gate).
- **M5/M6:** an NMOS current mirror that sets the tail current `I_TAIL`.

The OTA's open-loop gain `A = gm1·(ro2 ∥ ro4)` is closed in unity feedback, so `vout/vin = A/(1+A) ≈ 1`. The bandwidth is about the OTA's GBW, `gm1/(2π·CL)`. Ideally the mirror steers the whole tail current to the output in either direction, so slewing is close to `I_TAIL/CL`.

## Specifications

Besides two-sided targets and one-sided upper limits, this example also has **minima**: specs where only a shortfall counts as error.

```python
VDD = 1.2              # supply [V]: IHP SG13G2 1.2 V core devices
VIN_DC = 0.6           # input DC level [V]: the output should follow it
CL = 1.0e-12           # load capacitance [F]
NOISE_FMIN = 1.0       # noise integration from NOISE_FMIN up to 10·f-3dB [Hz]
LIN_TOL = 0.01         # linear range: input interval where the incremental gain stays within ±1 % of A0
DC_STEP = 0.01         # DC sweep step of the input [V]

SPEC_TARGETS = {
    "gain": 1.0,       # small-signal gain [V/V]
    "bw": 250e6,       # -3 dB bandwidth [Hz]
    "power": 500e-6,   # power dissipation VDD·I(VDD) [W]
}
SPEC_LIMITS = {        # one-sided upper limits
    "offset": 20e-3,   # |vout - vin| at VIN_DC [V]
    "vn_in": 150e-6,   # input-referred noise [Vrms]
}
SPEC_MINIMA = {        # one-sided lower limits
    "lin_range": 0.8,  # linear input range [V]
}

# design space: 7 variables
GMID_IN_BOUNDS = (5, 28)
GMID_LOAD_BOUNDS = (5, 15)
GMID_TAIL_BOUNDS = (5, 15)
I_TAIL_BOUNDS = (10e-6, 1e-3)

# R-NSGA-III
REF_POINTS = [         # aspiration points (spec error [%], area [um²], power [uW], -FoM_S [dB])
    (0, 60, 400, -180),
    (0, 200, 600, -182),
]
POP_PER_REF_POINT = 20 # reference directions per aspiration point
MU = 0.1               # spread of the directions around each aspiration point
N_GEN = 40
```

## Sizing with novaad

The design vector is `x = [gmid_in, gmid_load, gmid_tail, iL_in, iL_load, iL_tail, i_tail]`, and each half of the pair carries `I_TAIL/2`:

- **M3/M4:** M3 is diode-connected, so `Vsd = Vsg`. A fixed-point iteration gives its width and `v(outn) = VDD − Vsg3`. M4 matches M3.
- **M1/M2:** M2's gate and drain sit at the output, about `VIN_DC`, so the common source is at `Vs = VIN_DC − Vgs` and `Vsb = Vs`. The pair is sized on M1, whose drain is at `outn` (`Vds = v(outn) − Vs`).
- **M5/M6:** carry `I_TAIL` with `Vds = Vs`.

```python
def size(x) -> dict:
    gmid_in, gmid_load, gmid_tail = x[:3]
    l_in, l_load, l_tail = (LCH[int(i)] for i in x[3:6])
    i_tail = x[6]
    i_half = i_tail / 2

    vsd = VDD / 2
    for _ in range(2):   # M3: diode, vsd = vsg; M4 mirrors it
        dcop, s_load = pch.sizing(Spec(vds=vsd, vsb=0.0, lch=l_load, gmoverid=gmid_load, ids=i_half), return_dcop=True)
        vsd = dcop.vgs[0]
    v_outn = VDD - vsd

    vgs = 0.4
    for _ in range(2):   # sized on M1 (drain at outn); M2 shares the width
        vs = VIN_DC - vgs
        dcop, s_in = nch.sizing(Spec(vds=v_outn - vs, vsb=vs, lch=l_in, gmoverid=gmid_in, ids=i_half), return_dcop=True)
        vgs = dcop.vgs[0]
    vs = VIN_DC - vgs

    _, s_tail = nch.sizing(Spec(vds=vs, vsb=0.0, lch=l_tail, gmoverid=gmid_tail, ids=i_tail), return_dcop=True)
    return dict(w_in=s_in.wch[0], l_in=l_in, w_load=s_load.wch[0], l_load=l_load,
                w_tail=s_tail.wch[0], l_tail=l_tail, i_tail=i_tail)
```

## Testbench: OP + DC sweep + AC + noise in one ngspice run

```python
# devices checked for saturation
SAT = {f"xdut.xm{i}": spice.NMOS if i in (1, 2, 5) else spice.PMOS for i in (1, 2, 3, 4, 5)}


def measure() -> str:
    sat = " ".join(spice.mos(m, model, "vds", "vdss") for m, model in SAT.items())
    return f'''set sqrnoise
op
wrdata op.txt v(outp) v(xdut.outn) v(xdut.tail) i(vdd) {sat}
dc VIN 0 {VDD} {DC_STEP}
wrdata dc.txt v(outp)
ac dec 20 1 100G
let gdb = db(v(outp))
meas ac a0 find gdb at=1
let a3 = a0 - 3
meas ac f3 when gdb=$&a3
let fmax = 10 * f3
wrdata ac.txt gdb
noise v(outp) VIN dec 20 {NOISE_FMIN} $&fmax
setplot noise1
wrdata noise_spec.txt inoise_spectrum onoise_spectrum
setplot noise2
wrdata noise_tot.txt onoise_total'''


def dut(d: dict, name: str = "ota") -> str:
    # pins: supplies, differential input, output, bias current
    nmos, pmos = spice.NMOS, spice.PMOS
    return f'''.subckt {name} avdd avss inp inn out ibias
* 5-transistor OTA: NMOS pair, PMOS mirror load (diode on outn), mirrored tail
{fet("M3", "outn", "outn", "avdd", "avdd", pmos, d["w_load"], d["l_load"])}
{fet("M4", "out", "outn", "avdd", "avdd", pmos, d["w_load"], d["l_load"])}
{fet("M6", "ibias", "ibias", "avss", "avss", nmos, d["w_tail"], d["l_tail"])}
{fet("M5", "tail", "ibias", "avss", "avss", nmos, d["w_tail"], d["l_tail"])}
{fet("M1", "outn", "inp", "tail", "avss", nmos, d["w_in"], d["l_in"])}
{fet("M2", "out", "inn", "tail", "avss", nmos, d["w_in"], d["l_in"])}
.ends {name}'''


def testbench(d: dict, control: str | None = None, vin_source: str = f"dc {VIN_DC} ac 1") -> str:
    return spice.header() + dut(d) + f'''
VDD vdd 0 {VDD}
IREFN vdd nb {d["i_tail"]}
VIN in 0 {vin_source}
* voltage buffer: the output drives the negative input
XDUT vdd 0 in outp outp nb ota
CL outp 0 {CL}
.control
set wr_singlescale
set wr_vecnames
{control or measure()}
.endc
.end
'''
```

The **linear input range** is the contiguous interval of `VIN` around `VIN_DC` where the incremental gain `dvout/dvin`, taken from the DC sweep, stays within `LIN_TOL` of `A0`:

```python
def linear_range(vin, vout, a0) -> float:
    ok = np.abs(np.gradient(vout, vin) - a0) <= LIN_TOL * a0
    k = int(np.argmin(np.abs(vin - VIN_DC)))
    if not ok[k]:
        return 0.0
    lo, hi = k, k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(vin[hi] - vin[lo])
```

For the Schreier FoM, `SNR = (A²/2)/Vn²` uses half the linear range as the amplitude `A`, and `FoM_S = SNR + 10·log10(f−3dB/P)`.

## The optimization problem

| objective | definition |
|---|---|
| minimize **spec error** | RMS of the relative errors: two-sided for `SPEC_TARGETS`, the excess over `SPEC_LIMITS`, and the shortfall `max(0, 1 − measured/minimum)` under `SPEC_MINIMA` |
| minimize **area** | `Σ W·L` of all transistors |
| minimize **power** | `VDD·I(VDD)`, including the tail reference |
| maximize **Schreier FoM** | `SNR_dB + 10·log10(BW / P)` |

```python
def spec_errors(r) -> dict:
    errors = {k: r[k] / t - 1 for k, t in SPEC_TARGETS.items()}
    errors.update({k: max(0.0, r[k] / t - 1) for k, t in SPEC_LIMITS.items()})
    errors.update({k: max(0.0, 1 - r[k] / t) for k, t in SPEC_MINIMA.items()})
    return errors


class OtaFollower(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=7, n_obj=4, xl=XL, xu=XU, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            r = simulate(size(x))
        except Exception:
            out.update(FAILED)
            return
        out["F"] = [spec_error(r) * 100, r["area"] * 1e12, r["power"] * 1e6, -r["fom"]]
```

**R-NSGA-III** is NSGA-III steered by designer preferences. It places `POP_PER_REF_POINT` reference directions around each aspiration point, and `MU` sets how tightly they cluster. The population size follows from these settings: points × directions + objectives, which is 2 × 20 + 4 = 44 here.

```python
from pymoo.algorithms.moo.rnsga3 import RNSGA3

algorithm = RNSGA3(ref_points=np.array(REF_POINTS, dtype=float), pop_per_ref_point=POP_PER_REF_POINT, mu=MU)
```

The run, the live Pareto front and the closest-design list work exactly as in the [fully-differential example](fd-amplifier.md#live-pareto-front).

## Results

1760 evaluations in 58 s (8 workers). The closest design:

| | gm/id in / load / tail | L in / load / tail [µm] | W in / load / tail [µm] | I tail [µA] |
|---|---|---|---|---|
| sizing | 21.9 / 8.1 / 5.8 | 1.0 / 0.25 / 0.18 | 214.3 / 14.9 / 3.3 | 223.2 |

DC operating point: `outp` sits at 0.6009 V for a 0.6 V input, and every transistor is in saturation.

| element | I [µA] | vgs [V] | vds [V] | vdss [V] | gm [µS] | region |
|---|---|---|---|---|---|---|
| XDUT.XM1 | 95.5 | 0.273 | 0.291 | 0.119 | 2157 | saturation |
| XDUT.XM2 | 95.9 | 0.274 | 0.274 | 0.119 | 2162 | saturation |
| XDUT.XM3 | 95.5 | 0.582 | 0.582 | 0.208 | 843 | saturation |
| XDUT.XM4 | 95.9 | 0.582 | 0.599 | 0.208 | 845 | saturation |
| XDUT.XM5 | 191.4 | 0.690 | 0.327 | 0.268 | 1180 | saturation |

| metric | spec | ngspice |
|---|---|---|
| Gain [V/V] | target 1 | 0.945 (−5.5 %) |
| −3 dB bandwidth [MHz] | target 250 | 249.0 (−0.4 %) |
| Power [µW] | target 500 | 497.5 (−0.5 %) |
| Offset `|vout − vin|` [mV] | ≤ 20 | 0.93 |
| Input-referred noise [µVrms] | ≤ 150 | 91.3 |
| Linear input range [V] | ≥ 0.8 | 0.66 (shortfall 17.5 %) |
| **Spec error, RMS** | minimize | **7.49 %** |
| Transient gain at 25 MHz [V/V] | = AC gain | 0.9414 |
| Slew rate up / down [V/µs] | reference `I(M5)/CL` = 191 | 120 / 112 |
| Saturation margin [V] | > 0 | 0.060 |
| SNR [dB] | | 68.6 |
| Area | minimize | 437 µm² (20.9 µm × 20.9 µm) |
| Schreier FoM [dB] | maximize | 185.6 |

!!! info "Gain, linear range and headroom at 1.2 V"
    In unity feedback the gain is `A/(1+A)`, and at 1.2 V the OTA's open-loop gain `A = gm1·(ro2 ∥ ro4)` is only about 17, so the follower gain is 0.945 rather than 0.98. The 0.8 V linear-range minimum is the binding spec: the stack of tail, pair and mirror load leaves the output only about 0.66 V of range where the incremental gain stays within 1 % of `A0`. Both mirror devices sit at almost the same `Vsd` (0.582 V and 0.599 V), so their currents match within 0.4 % and the systematic offset is under 1 mV.
