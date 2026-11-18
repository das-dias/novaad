# Discrete-time fully-differential amplifier

The [fully-differential amplifier](fd-amplifier.md) with **switched-capacitor common-mode sensing**: the resistors between the outputs are replaced by two capacitors `C_CM` and a transmission gate. The gate resets the load bias node `cmfb` to a reference `avcm` on every clock cycle. R-NSGA-II sizes it in the static reset phase. The picked design is then verified clocked at 50 MHz, with a periodic-steady-state Bode plot built from a swept sine.

[:material-notebook: `notebooks/05_dt_fd_amplifier_rnsga2.ipynb`](https://github.com/das-dias/novaad/blob/master/notebooks/05_dt_fd_amplifier_rnsga2.ipynb){ .md-button }

## Architecture

```text
                 VDD ─────────┬───────────────────┬──────────
                              │                   │
          cmfb ──────────┤ M3            M4 ├──────── cmfb
                              │                   │
              outn ───────────┼──┬─| |─-┬─-| |─┬──┼─────────── outp
                              │  │ C_CM │ C_CM │  │
                              │  │      cmfb ──┼──┼── TG (ph) ── avcm
              inp ───────┤ M1 │  CL           CL  │ M2 ├───── inn
                              └────────┬──────────┘
                                     tail
   IREF ↓                              │
        nb ──┬──────────────────┤ M5  (active tail current source)
       M6 ├──┘ (diode)                 │
            GND ──────────────────────GND
```

The amplifier is a `.subckt` macro with pins `avdd avss inp inn outp outn avcm ibias clk clkb`:

```python
def dut(d: dict, name: str = "fd_amp") -> str:
    nmos, pmos = spice.NMOS, spice.PMOS
    return f'''.subckt {name} avdd avss inp inn outp outn avcm ibias clk clkb
{fet("M6", "ibias", "ibias", "avss", "avss", nmos, d["w_tail"], d["l_tail"])}
{fet("M5", "tail", "ibias", "avss", "avss", nmos, d["w_tail"], d["l_tail"])}
{fet("M1", "outn", "inp", "tail", "avss", nmos, d["w_in"], d["l_in"])}
{fet("M2", "outp", "inn", "tail", "avss", nmos, d["w_in"], d["l_in"])}
{fet("M3", "outn", "cmfb", "avdd", "avdd", pmos, d["w_load"], d["l_load"])}
{fet("M4", "outp", "cmfb", "avdd", "avdd", pmos, d["w_load"], d["l_load"])}
C1 outp cmfb {d["c_cm"]}
C2 outn cmfb {d["c_cm"]}
* transmission gate resetting the common-mode node, with half-width dummies for charge-injection cancellation
{fet("MSPDL", "cmfb", "clk", "cmfb", "avdd", pmos, d["w_switch_p_dummy"], d["l_switch"])}
{fet("MSP", "cmfb", "clkb", "avcm", "avdd", pmos, d["w_switch_p"], d["l_switch"])}
{fet("MSPDR", "avcm", "clk", "avcm", "avdd", pmos, d["w_switch_p_dummy"], d["l_switch"])}
{fet("MSNDL", "cmfb", "clkb", "cmfb", "avss", nmos, d["w_switch_n_dummy"], d["l_switch"])}
{fet("MSN", "cmfb", "clk", "avcm", "avss", nmos, d["w_switch_n"], d["l_switch"])}
{fet("MSNDR", "avcm", "clkb", "avcm", "avss", nmos, d["w_switch_n_dummy"], d["l_switch"])}
.ends {name}'''
```

- **Reset phase `ph`:** the transmission gate (minimum length, 130 nm) pulls `cmfb` to `avcm`.
- **Amplify phase `phb`:** the gate opens, and `cmfb` floats and follows the output common mode through `C_CM`.
- **Clocks:** `ph`/`phb` are non-overlapping and anti-phase at `F_CLK = 50 MHz`, with a `T_NOV = 0.5 ns` gap.
- **Dummies:** half-width dummy switches on both sides of each switch cancel part of the charge injection.

## Testbench: static and clocked

`testbench(d, control, vd_source, clocked)` instantiates the macro as `XDUT`. Internal nodes and devices are addressed hierarchically, for example `v(xdut.tail)` and `spice.mos("xdut.xm1", spice.NMOS, "vdss")`, and `utils.dc_tables` flattens the instance.

- **Static reset phase**, used by the optimizer: `ph = VDD`, `phb = 0`.
    - The switch holds `cmfb`, so a DC operating point exists, and OP + AC + noise run as in the continuous-time example.
    - With `cmfb` pinned, the output common mode is open-loop: a few µA of mismatch between loads and tail rails it.
    - A high-gain B-source servo therefore sets `avcm` so the output CM sits where novaad sized it, as a replica-bias reference would. It only acts on the common mode, so differential gain and noise are unaffected. The solved level is returned as `r["avcm"]`.
- **Clocked**, used for verification: `ph`/`phb` are `pulse` sources, and `avcm` is a fixed source at the solved level.

```python
def reset_level(d: dict, clocked: bool) -> str:
    if clocked:   # the level the static servo found
        return f"VAVCM avcm 0 {d['avcm']}"
    vocm = d["vocm"]
    return f"BAVCM avcm 0 V={vocm} + 100 * ((v(outp) + v(outn)) / 2 - {vocm})"


def clocks(clocked: bool) -> str:
    if not clocked:   # static reset phase
        return f"VPH ph 0 {VDD}\nVPHB phb 0 0"
    period = 1 / F_CLK
    width = period / 2 - T_NOV - 2 * T_EDGE
    return (f"VPH ph 0 pulse(0 {VDD} 0 {T_EDGE} {T_EDGE} {width} {period})\n"
            f"VPHB phb 0 pulse(0 {VDD} {period / 2} {T_EDGE} {T_EDGE} {width} {period})")
```

The optimization problem, the objectives and the live Pareto front are the same as in the [fully-differential example](fd-amplifier.md#the-optimization-problem). The 8th variable is `C_CM` instead of `R_CM`. The area counts the switches and the MIM capacitors, `2·C_CM/CAP_DENSITY` with 1.5 fF/µm².

## Periodic steady state: swept-sine Bode

ngspice has no PAC analysis, and its `pss` only handles autonomous oscillators. The clocked small-signal transfer is therefore measured in the time domain. The result is the baseband (sideband 0) response that PSS + PAC would give:

1. **Coherent frequency:** each test frequency is snapped to `f = (m/N)·F_CLK` with `N ≤ PSS_MAX_CYCLES`, so the input sine and the clock share the period `T = N/F_CLK`.
2. **Settle, then record:** a clocked transient runs for an integer number of periods to settle (at least `10·τ` and 20 clock cycles). Then one more period is recorded and `linearize`d onto a uniform grid.
3. **DFT:** a DFT of `vod` and `vd` at `f` gives `H(f) = Vod(f)/Vd(f)`. Clock feedthrough and harmonics fall on other bins.

```python
def pss_point(f_target: float) -> tuple[float, complex]:
    ratio = Fraction(f_target / F_CLK).limit_denominator(PSS_MAX_CYCLES)
    f = float(ratio) * F_CLK                  # f/F_CLK = m/N: input and clock share the period T = N/F_CLK
    period = ratio.denominator / F_CLK
    settle = ceil(max(10 * tau, PSS_SETTLE_CYCLES / F_CLK) / period) * period
    tstep = 1 / (F_CLK * STEPS_PER_CLK)
    control = f"""tran {tstep} {settle + period} {settle} {tstep}
let vod = v(outp) - v(outn)
linearize vod v(vd)
wrdata pss.txt vod v(vd)"""
    wd = spice.run(testbench(d, control, f"dc 0 sin(0 {PSS_AMP} {f})", clocked=True))
    data = spice.read(wd / "pss.txt").iloc[:-1]   # exactly one period
    e = np.exp(-2j * pi * f * data["time"].to_numpy())
    return f, complex(data["vod"].to_numpy() @ e / (data["v(vd)"].to_numpy() @ e))
```

24 log-spaced points, from `F_CLK/500` to `2·F_CLK`, run in parallel. The plot overlays the static AC response and marks the Nyquist frequency `F_CLK/2`.

## Results

At 1.2 V, with the specs of the continuous-time example: 1920 evaluations in 103 s (8 workers). The closest design:

| | gm/id load | gm/id tail | L in / load / tail [µm] | W in / load / tail [µm] | I tail [µA] | C_CM [fF] |
|---|---|---|---|---|---|---|
| sizing | 7.29 | 10.72 | 0.5 / 0.5 / 2.0 | 51.9 / 18.4 / 78.7 | 185.3 | 50.5 |

| metric | target | ngspice |
|---|---|---|
| Differential gain [dB] | 26 | 24.49 (−5.8 %) |
| GBW [MHz] | 500 | 438.8 (−12.2 %) |
| Output CM [V] | 0.6 | 0.59 (−2.1 %) |
| Power [µW] | 400 | 436.8 (+9.2 %) |
| Input-referred noise [µVrms] | ≤ 60 | 42.35 (within the limit) |
| **Spec error, RMS** | minimize | **7.39 %** |
| Clocked DC gain, PSS [dB] | static AC: 24.49 | **24.42** |
| Clocked GBW, PSS [MHz] | static AC: 439.4 | **434.4** |
| CM reset level `avcm` [V] | servo-solved | 0.592 |
| Slew rate up / down [V/µs] | reference `I_TAIL/CL` = 371 | 291 / 261 |
| Saturation margin [V] | > 0 | 0.09 |
| Area | minimize | 453 µm² (21.3 µm × 21.3 µm) |
| Schreier FoM [dB] | maximize | 171.7 |

Clocked, the common-mode loop behaves as intended:
- `cmfb` stays within 0.585–0.596 V around `avcm = 0.592 V`;
- the output common mode holds at 0.575–0.582 V across both phases.

The periodic response matches the static AC within 0.1 dB in gain and about 1 % in GBW. Below Nyquist, the switched CM sensing doesn't disturb the differential path.
