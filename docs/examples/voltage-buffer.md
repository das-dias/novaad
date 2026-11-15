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
VDD = 3.3              # supply [V]
VIN_DC = 1.65          # input DC level [V]: the output should follow it
CL = 1.0e-12           # load capacitance [F]
NOISE_FMIN = 1.0       # noise integration from NOISE_FMIN up to 10·f-3dB [Hz]
LIN_TOL = 0.01         # linear range: input interval where the incremental gain stays within ±1 % of A0
DC_STEP = 0.01         # DC sweep step of the input [V]

SPEC_TARGETS = {
    "gain": 1.0,       # small-signal gain [V/V]
    "bw": 350e6,       # -3 dB bandwidth [Hz]
    "power": 1.5e-3,   # power dissipation VDD·I(VDD) [W]
}
SPEC_LIMITS = {        # one-sided upper limits
    "offset": 20e-3,   # |vout - vin| at VIN_DC [V]
    "vn_in": 150e-6,   # input-referred noise [Vrms]
}
SPEC_MINIMA = {        # one-sided lower limits
    "lin_range": 1.5,  # linear input range [V]
}

# design space: 7 variables
GMID_IN_BOUNDS = (5, 20)
GMID_LOAD_BOUNDS = (5, 15)
GMID_TAIL_BOUNDS = (5, 15)
I_TAIL_BOUNDS = (10e-6, 500e-6)

# R-NSGA-III
REF_POINTS = [         # aspiration points (spec error [%], area [um²], power [uW], -FoM_S [dB])
    (0, 200, 500, -184),
    (0, 800, 800, -187),
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

    vgs = 0.8
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
def measure() -> str:
    sat = " ".join(f"@m{i}[vds] @m{i}[vdsat]" for i in (1, 2, 3, 4, 5))
    return f'''set sqrnoise
op
wrdata op.txt v(outp) v(outn) v(tail) i(vdd) {sat}
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


def netlist(d: dict, control: str | None = None, vin_source: str = f"dc {VIN_DC} ac 1") -> str:
    return spice.header() + f'''
* voltage buffer: 5-transistor OTA follower, PMOS mirror load, outp -> M2 gate (negative input)
VDD vdd 0 {VDD}
{fet("M3", "outn", "outn", "vdd", "vdd", "pfet_03v3", d["w_load"], d["l_load"])}
{fet("M4", "outp", "outn", "vdd", "vdd", "pfet_03v3", d["w_load"], d["l_load"])}
IREFN vdd nb {d["i_tail"]}
{fet("M6", "nb", "nb", "0", "0", "nfet_03v3", d["w_tail"], d["l_tail"])}
{fet("M5", "tail", "nb", "0", "0", "nfet_03v3", d["w_tail"], d["l_tail"])}
VIN in 0 {vin_source}
{fet("M1", "outn", "in", "tail", "0", "nfet_03v3", d["w_in"], d["l_in"])}
{fet("M2", "outp", "outp", "tail", "0", "nfet_03v3", d["w_in"], d["l_in"])}
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

1760 evaluations in 49 s (8 workers). The closest design:

| | gm/id in / load / tail | L in / load / tail [µm] | W in / load / tail [µm] | I tail [µA] |
|---|---|---|---|---|
| sizing | 18.0 / 10.8 / 14.1 | 0.5 / 0.28 / 0.28 | 200.3 / 67.8 / 94.1 | 228.7 |

DC operating point: `outp` sits at 1.664 V for a 1.65 V input, and every transistor is in saturation.

| element | I [µA] | vgs [V] | vds [V] | vdsat [V] | gm [µS] | region |
|---|---|---|---|---|---|---|
| M1 | 107.7 | 0.911 | 1.716 | 0.083 | 1971 | saturation |
| M2 | 128.2 | 0.925 | 0.925 | 0.088 | 2271 | saturation |
| M3 | 107.7 | 0.845 | 0.845 | 0.150 | 1288 | saturation |
| M4 | 128.2 | 0.845 | 1.636 | 0.157 | 1463 | saturation |
| M5 | 235.8 | 0.679 | 0.739 | 0.105 | 3351 | saturation |

| metric | spec | ngspice |
|---|---|---|
| Gain [V/V] | target 1 | 0.982 (−1.8 %) |
| −3 dB bandwidth [MHz] | target 350 | 321.2 (−8.2 %) |
| Power [µW] | target 1500 | 1533 (+2.2 %) |
| Offset `|vout − vin|` [mV] | ≤ 20 | 14.3 |
| Input-referred noise [µVrms] | ≤ 150 | 90.2 |
| Linear input range [V] | ≥ 1.5 | 2.20 |
| **Spec error, RMS** | minimize | **3.55 %** |
| Transient gain at 32 MHz [V/V] | = AC gain, 0.9791 | 0.9791 |
| Slew rate up / down [V/µs] | reference `I(M5)/CL` = 236 | 329 / 181 |
| Saturation margin [V] | > 0 | 0.63 |
| SNR [dB] | | 78.9 |
| Area | minimize | 291 µm² (17.1 µm × 17.1 µm) |
| Schreier FoM [dB] | maximize | 192.1 |

!!! info "Offset and slew asymmetry"
    The mirror devices sit at different `Vsd`: 0.85 V for the diode M3 and 1.64 V for M4. With a short 0.28 µm load, channel-length modulation makes M4 source 128 µA while M3 carries 108 µA. M2 has to sink the larger current, so its `Vgs` exceeds M1's, and the difference shows up as a systematic 14 mV offset. The mirror is also not ideal during slewing: on the rising edge the output gets more than `I_TAIL/CL`, because the tail node lifts and the short-channel tail delivers more current, and on the falling edge it gets less.
