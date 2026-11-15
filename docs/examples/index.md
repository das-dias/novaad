# Examples

The `notebooks/` directory contains four design examples in the open **gf180mcu** 180 nm process, using the 3.3 V devices `nfet_03v3` and `pfet_03v3`. Each one sizes the circuit with novaad and verifies it in ngspice. The last two also wrap novaad and ngspice in a pymoo multi-objective optimizer.

| | Example | Method | Notebook |
|---|---|---|---|
| 0 | [LUT extraction](../lut-extraction.md) | ciel + ngspice characterization | `01_pdk_ngspice_lut_extraction.ipynb` |
| 1 | [Common-source amplifier](cs-amplifier.md) | hand design with the gm/id method | `02_cs_amplifier_design.ipynb` |
| 2 | [Fully-differential amplifier](fd-amplifier.md) | R-NSGA-II, 8 variables, 4 objectives | `03_fd_amplifier_rnsga2.ipynb` |
| 3 | [Voltage buffer (OTA follower)](voltage-buffer.md) | R-NSGA-III, 7 variables, 4 objectives | `04_buffer_widlar_rnsga3.ipynb` |

## Running the notebooks

```bash
make dev                              # uv sync --all-groups: pymoo, inspice, ipykernel, nbconvert ...
uv run jupyter lab notebooks/         # or open them in VS Code
```

Run notebook **01 first**. It installs the PDK and writes `notebooks/luts/gf180_{nch,pch}.csv`, which the other notebooks read. To execute a notebook headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_cs_amplifier_design.ipynb
```

## How every example is organized

**Specs at the top.** The first code cell of every notebook holds the design specifications and knobs as ALL-CAPS globals: supply, load, gain/bandwidth/noise/power targets, gm/id choices, optimizer settings and transient test signals. Edit them and re-run the notebook.

**novaad sizes the circuit.** The transistors are sized with `Device.sizing(DeviceSizingSpecification(...), return_dcop=True)`. Bias voltages that depend on each other, such as a diode-connected load setting the drain voltage of the pair, are resolved with a short fixed-point iteration.

**ngspice verifies it.** Each testbench is a Python f-string netlist. Each transistor is split into `m` fingers no wider than the LUT's reference width:

```python
def fet(name, d, g, s, b, model, w, l):
    m = max(1, ceil(w / REF_WIDTH - 1e-9))   # REF_WIDTH-wide fingers, like the LUT
    return f"{name} {d} {g} {s} {b} {model} w={w / m:.6g} l={l:.6g} m={m}"
```

The verification sections are the same in every example:

1. **DC operating point table.** `utils.dc_tables` parses the netlist with [InSpice](https://github.com/InSpice/InSpice) to map each element's terminals to nets. It then tabulates currents, terminal voltages and the BSIM4 bias quantities of every MOSFET (`vgs`, `vds`, `vth`, `vdsat`, `gm`, `gds`, region), plus a node-voltage table.
2. **AC Bode plot** of magnitude and phase. The bandwidth is measured with `spice.bandwidth`, which gives `GBW = A0·f−3dB`.
3. **Transients:** a small sine (checked against the AC gain at the same frequency), a full-supply ramp, and a full-supply step. Input and output share one voltage axis. The slew rate is the 40–60 % slope from `spice.slew_rate`.
4. **Noise** density, integrated and referred to the input.
5. **Report table:** specs versus the novaad prediction versus ngspice.

## Shared helpers

| module | functions |
|---|---|
| `notebooks/spice.py` | `header`, `run`, `read`, `bandwidth`, `slew_rate` (see [LUT extraction](../lut-extraction.md#3-a-minimal-ngspice-driver)) |
| `notebooks/utils.py` | `parse` (InSpice), `dc_vectors`, `dc_tables`, `tran_plot` (Bokeh, shared voltage axis), `footprint` (area as `x µm × x µm`) |

!!! note "Results on these pages"
    The numbers quoted in the examples come from the last execution of each notebook with its default specs, at the typical corner and 27 °C. The optimizer runs are seeded (`SEED = 1`), so they are reproducible on the same LUTs.
