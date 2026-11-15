# Fully-differential amplifier with R-NSGA-II

A fully-differential pair with active loads and resistive common-mode sensing. novaad and ngspice run inside **pymoo's R-NSGA-II**, which drives every measured spec toward its target while trading off area, power and the Schreier figure of merit. The Pareto front is plotted live while the optimizer runs.

[:material-notebook: `notebooks/03_fd_amplifier_rnsga2.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/03_fd_amplifier_rnsga2.ipynb){ .md-button }

## Architecture

<!-- Architecture figure: add docs/figs/architectures/fd_amplifier.svg and uncomment
![Fully-differential amplifier](../figs/architectures/fd_amplifier.svg)
-->

```text
                 VDD ─────────┬───────────────────┬──────────
                              │                   │
          cmfb ──────────┤ M3            M4 ├──────── cmfb
                              │                   │
              outn ───────────┼──┬─/\/\─┬─/\/\─┬──┼─────────── outp
                              │  │ R_CM │ R_CM │  │
                              │  │      cmfb   │  │
              inp ───────┤ M1 │  CL           CL  │ M2 ├───── inn
                              └────────┬──────────┘
                                     tail
   IREF ↓                              │
        nb ──┬──────────────────┤ M5  (active tail current source)
       M6 ├──┘ (diode)                 │
            GND ──────────────────────GND
```

- **M1/M2:** NMOS differential pair.
- **M3/M4:** PMOS active loads with their gates at `cmfb`, the midpoint of the two `R_CM` resistors.
    - For the common mode, the loads act as diodes and set the output CM to `VDD − Vsg`.
    - For the differential signal, `cmfb` is a virtual ground, so the loads act as current sources. The gain is then `gm1·(ro1 ∥ ro3 ∥ R_CM)`.
- **M5/M6:** NMOS current mirror, the active source of the tail current `I_TAIL`.

## Specifications

The specs are **targets**, not bounds. The optimizer drives every measured parameter toward its target.

```python
VDD = 3.3              # supply [V]
VIN_CM = 1.2           # input common mode [V]
CL = 0.5e-12           # load capacitance per output [F]
NOISE_FMIN = 1.0       # lower bound of the noise integration [Hz] (upper bound = GBW)

SPEC_TARGETS = {
    "gain_db": 36,     # differential DC gain [dB]
    "gbw": 500e6,      # differential gain-bandwidth product A0·f-3dB [Hz]
    "vocm": 2.0,       # output common mode [V]: set by the loads, VDD - Vsg
    "power": 1.2e-3,   # power dissipation VDD·I(VDD) [W], including the IREF branch
}
# one-sided limits: only exceeding the limit counts as error
SPEC_LIMITS = {
    "vn_in": 20e-6,    # input-referred noise [Vrms]: output noise (NOISE_FMIN..GBW) / DC gain
}

RES_SHEET = 1000       # poly resistor sheet resistance [ohm/sq] (ppolyf_u_1k)
RES_WIDTH = 1e-6       # poly resistor width [m]

# design space: 8 variables
GMID_IN_BOUNDS = (12, 20)
GMID_LOAD_BOUNDS = (6, 18)
GMID_TAIL_BOUNDS = (6, 12)
I_TAIL_BOUNDS = (10e-6, 500e-6)
R_CM_BOUNDS = (200e3, 2e6)
# channel lengths are picked from the LUT grid (L_IN, L_LOAD, L_TAIL)

# R-NSGA-II
REF_POINTS = [         # aspiration points (spec error [%], area [um²], power [uW], -FoM_S [dB])
    (0, 1500, 1500, -176),
    (0, 500, 1200, -174),
]
EPSILON = 0.05
POP_SIZE = 48
N_GEN = 40
```

## Sizing with novaad

A design vector is `x = [gmid_in, gmid_load, gmid_tail, iL_in, iL_load, iL_tail, i_tail, r_cm]`, where `iL_*` index into the LUT channel lengths. The bias voltages depend on each other, so a short fixed-point iteration resolves them:

- **Load:** its gate sits at the output CM, so `Vsd = Vsg`. Size it, read `Vsg`, and repeat. Then `VOCM = VDD − Vsg`.
- **Input pair:** the source is at `Vs = VIN_CM − Vgs1` with the bulk grounded, so `Vsb = Vs` and `Vds = VOCM − Vs`.
- **Tail:** `Vds = Vs`, carrying `I_TAIL`.

```python
from novaad import DeviceSizingSpecification as Spec

def size(x) -> dict:
    gmid_in, gmid_load, gmid_tail = x[:3]
    l_in, l_load, l_tail = (LCH[int(i)] for i in x[3:6])
    i_tail, r_cm = x[6], x[7]

    vsd = VDD / 2
    for _ in range(2):
        dcop, s_load = pch.sizing(Spec(vds=vsd, vsb=0.0, lch=l_load, gmoverid=gmid_load, ids=i_tail / 2), return_dcop=True)
        vsd = dcop.vgs[0]
    vocm = VDD - vsd

    vs = VIN_CM - 0.8
    for _ in range(2):
        dcop, s_in = nch.sizing(Spec(vds=vocm - vs, vsb=vs, lch=l_in, gmoverid=gmid_in, ids=i_tail / 2), return_dcop=True)
        vs = VIN_CM - dcop.vgs[0]

    _, s_tail = nch.sizing(Spec(vds=vs, vsb=0.0, lch=l_tail, gmoverid=gmid_tail, ids=i_tail), return_dcop=True)
    return dict(w_in=s_in.wch[0], l_in=l_in, w_load=s_load.wch[0], l_load=l_load,
                w_tail=s_tail.wch[0], l_tail=l_tail, i_tail=i_tail, r_cm=r_cm)


def area(d: dict) -> float:
    fets = 2 * d["w_in"] * d["l_in"] + 2 * d["w_load"] * d["l_load"] + 2 * d["w_tail"] * d["l_tail"]
    return fets + 2 * d["r_cm"] / RES_SHEET * RES_WIDTH ** 2
```

## Testbench: OP + AC + noise in one ngspice run

The differential input comes from two E-sources around `VIN_CM`, both driven by one source `Vd`. `Vd` is therefore the input of the AC transfer and the reference source of the noise analysis. `meas` finds the DC gain `A0` and the −3 dB corner `f3`, and the noise is integrated up to `GBW = A0·f3`.

```python
def measure() -> str:
    return f'''set sqrnoise
op
wrdata op.txt v(outp) v(outn) v(tail) i(vdd) @m1[vds] @m1[vdsat] @m3[vds] @m3[vdsat] @m5[vds] @m5[vdsat]
ac dec 20 1 100G
let gdb = db(v(outp) - v(outn))
meas ac a0 find gdb at=1
let a3 = a0 - 3
meas ac f3 when gdb=$&a3
let gbw = f3 * 10^(a0 / 20)
wrdata ac.txt gdb
noise v(outp,outn) Vd dec 20 {NOISE_FMIN} $&gbw
setplot noise1
wrdata noise_spec.txt inoise_spectrum onoise_spectrum
setplot noise2
wrdata noise_tot.txt onoise_total'''


def netlist(d: dict, control: str | None = None, vd_source: str = "dc 0 ac 1") -> str:
    return spice.header() + f'''
* fully-differential pair, PMOS active loads, resistive CM sensing, mirrored tail
VDD vdd 0 {VDD}
IREF vdd nb {d["i_tail"]}
{fet("M6", "nb", "nb", "0", "0", "nfet_03v3", d["w_tail"], d["l_tail"])}
{fet("M5", "tail", "nb", "0", "0", "nfet_03v3", d["w_tail"], d["l_tail"])}
VCM cmi 0 {VIN_CM}
Vd vd 0 {vd_source}
Ep inp cmi vd 0 0.5
En cmi inn vd 0 0.5
{fet("M1", "outn", "inp", "tail", "0", "nfet_03v3", d["w_in"], d["l_in"])}
{fet("M2", "outp", "inn", "tail", "0", "nfet_03v3", d["w_in"], d["l_in"])}
{fet("M3", "outn", "cmfb", "vdd", "vdd", "pfet_03v3", d["w_load"], d["l_load"])}
{fet("M4", "outp", "cmfb", "vdd", "vdd", "pfet_03v3", d["w_load"], d["l_load"])}
R1 outp cmfb {d["r_cm"]}
R2 outn cmfb {d["r_cm"]}
C1 outp 0 {CL}
C2 outn 0 {CL}
.control
set wr_singlescale
set wr_vecnames
{control or measure()}
.endc
.end
'''
```

`simulate(d)` reads the output files and returns:

- gain, GBW, output CM and power;
- the peak differential swing allowed by the headroom of M1 and M3;
- `SNR = (A²/2)/Vn²`;
- `FoM_S = SNR + 10·log10(GBW/P)`;
- the smallest saturation margin `|Vds| − |Vdsat|`.

!!! warning "Why GBW is A0·f−3dB, not the 0 dB crossing"
    Large input devices couple the input to the outputs through `Cgd`. This flattens `|H|` just above 0 dB at high frequency, and pushes the 0 dB crossing far beyond the real gain-bandwidth product. An optimizer quickly learns to exploit that, so GBW is computed from the single-pole asymptote.

## The optimization problem

pymoo minimizes four objectives, `[spec error %, area µm², power µW, −FoM_S dB]`:

| objective | definition |
|---|---|
| minimize **spec error** | RMS of `measured/target − 1` over `SPEC_TARGETS`, and of the excess `max(0, measured/limit − 1)` over `SPEC_LIMITS` |
| minimize **area** | `Σ W·L` of M1–M6 plus the two poly resistors |
| minimize **power** | `VDD·I(VDD)` |
| maximize **Schreier FoM** | `FoM_S = SNR_dB + 10·log10(GBW / P)` |

There are no hard constraints. A candidate that novaad can't size, or that ngspice can't solve, gets the worst objectives and R-NSGA-II discards it.

```python
from pymoo.core.problem import ElementwiseProblem

def spec_errors(r) -> dict:
    errors = {k: r[k] / target - 1 for k, target in SPEC_TARGETS.items()}
    errors.update({k: max(0.0, r[k] / limit - 1) for k, limit in SPEC_LIMITS.items()})
    return errors


def spec_error(r) -> float:
    return float(np.sqrt(np.mean(np.square(list(spec_errors(r).values())))))


class FullyDifferentialAmp(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=8, n_obj=4, xl=XL, xu=XU, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            r = simulate(size(x))
        except Exception:
            out.update(FAILED)
            return
        out["F"] = [spec_error(r) * 100, r["area"] * 1e12, r["power"] * 1e6, -r["fom"]]
```

**R-NSGA-II** is NSGA-II with designer preferences. Instead of spreading solutions over the whole Pareto front, it concentrates them around the aspiration points in `REF_POINTS`, and `EPSILON` controls how tightly they cluster.

## Live Pareto front

Three pieces run at the same time:

- **Optimizer thread:** `minimize(...)` evaluates each generation through a `ThreadPool`. ngspice runs as subprocesses, so threads give real parallelism.
- **Callback:** after every generation it copies the non-dominated set into a shared `state` dict, guarded by a `Lock`.
- **Plot thread:** every `PLOT_PERIOD` seconds it renders the front as a self-contained Bokeh page inside an `<iframe>`, and swaps it into the notebook through an IPython display handle. This works in Jupyter, JupyterLab and VS Code, where Bokeh's `push_notebook` does not.

```python
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization


class Snapshot(Callback):
    def notify(self, algorithm):
        with lock:
            state["gen"] = algorithm.n_gen
            state["F"] = algorithm.opt.get("F").copy()


def optimize():
    try:
        with ThreadPool(N_WORKERS) as pool:
            problem = FullyDifferentialAmp(elementwise_runner=StarmapParallelization(pool.starmap))
            algorithm = RNSGA2(ref_points=np.array(REF_POINTS, dtype=float), epsilon=EPSILON, pop_size=POP_SIZE)
            state["res"] = minimize(problem, algorithm, ("n_gen", N_GEN), seed=SEED, callback=Snapshot())
    finally:
        done.set()


def as_iframe(layout, title: str) -> HTML:
    page = file_html(layout, INLINE, title)   # BokehJS inlined: renders anywhere
    return HTML(f'<iframe srcdoc="{escape(page)}" style="width:1000px; height:410px; border:0"></iframe>')


handle = display(HTML("<i>Pareto front: waiting for generation 1…</i>"), display_id=True)


def live_plot():
    last_gen = -1
    while True:
        finished = done.wait(PLOT_PERIOD)
        with lock:
            gen, F = state["gen"], state["F"]
        if gen != last_gen and len(F):
            handle.update(as_iframe(front_plot(F, f"Generation {gen}/{N_GEN}"), "front"))
            last_gen = gen
        if finished:
            return


Thread(target=optimize, daemon=True).start()
Thread(target=live_plot, daemon=True).start()
```

`front_plot` draws two panels: area vs power, colored by FoM, and FoM vs spec error. The reference points are marked with red crosses.

After the run, the whole final population is re-simulated and the `N_CLOSEST` distinct designs are ranked by spec error, with ties broken by FoM. Each row shows the per-spec errors, the measured performance, the saturation margin and the sizing. The closest designs are circled on the final front.

## Results

1920 evaluations in 53 s (8 workers). The closest design:

| | gm/id load | gm/id tail | L in / load / tail [µm] | W in / load / tail [µm] | I tail [µA] | R_CM [kΩ] |
|---|---|---|---|---|---|---|
| sizing | 6.05 | 6.47 | 0.5 / 0.28 / 4.0 | 261.8 / 13.6 / 155.8 | 201.1 | 249.2 |

| metric | target | ngspice |
|---|---|---|
| Differential gain [dB] | 36 | 36.05 (+0.1 %) |
| GBW [MHz] | 500 | 418.1 (−16.4 %) |
| Output CM [V] | 2.0 | 2.28 (+13.8 %) |
| Power [µW] | 1200 | 1322 (+10.2 %) |
| Input-referred noise [µVrms] | ≤ 20 | 19.66 (within the limit) |
| **Spec error, RMS** | minimize | **10.62 %** |
| Transient gain at 0.66 MHz [dB] | = AC gain, 36.00 | 36.00 |
| Slew rate up / down [V/µs] | reference `I_TAIL/CL` = 402 | 269 / 269 |
| Saturation margin [V] | > 0 | 0.17 |
| Peak differential swing [V] | | 1.45 |
| SNR [dB] | | 58.3 |
| Area | minimize | 2014 µm² (44.9 µm × 44.9 µm) |
| Schreier FoM [dB] | maximize | 173.3 |

!!! info "Reading the trade-off"
    The output CM is set by the loads, `VOCM = VDD − Vsg3`. Even at the lowest load gm/id allowed (6), `Vsg3` stays around 1.0 V, so `VOCM ≈ 2.28 V` and the 2.0 V target can't be reached in this topology. The optimizer accepts that error and balances it against GBW and power. To move the CM, change the topology (for example, add a CMFB amplifier) or widen `GMID_LOAD_BOUNDS`.
