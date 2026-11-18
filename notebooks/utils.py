"""Parse SPICE testbenches with InSpice into DC operating-point tables, plus plotting helpers."""

import contextlib
import re
import io
from types import SimpleNamespace

from InSpice.Spice.Parser import SpiceSource
from InSpice.Spice.Parser.HighLevelParser import Element, Subcircuit

from spice import column

GROUND = {"0", "gnd", "vss"}
# PDK MOSFETs: subcircuits from the model library, each wrapping one OSDI (PSP103) device
MOS_MODELS = {"sg13_lv_nmos", "sg13_lv_pmos", "sg13_hv_nmos", "sg13_hv_pmos"}


def parse(netlist: str) -> list[SimpleNamespace]:
    """Circuit elements of an ngspice netlist, parsed by InSpice (dot commands and .control are skipped).

    Subcircuit instances defined in the netlist are flattened with ngspice's names: R1 in XDUT is
    `r.xdut.r1`, node tail is `xdut.tail`. PDK MOSFET instances become `M` elements.
    """
    # the .control block is ngspice scripting, not circuit: InSpice can't parse all of it (e.g. `$&var`, `^`)
    netlist = re.sub(r"^\.control\b.*?^\.endc\b", "", netlist, flags=re.S | re.M | re.I)
    # InSpice's PLY parser prints its grammar-table warnings on first use
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        source = SpiceSource(netlist)
    lines = list(source.obj_lines)
    subckts = {s.name.lower(): s for s in lines if isinstance(s, Subcircuit)}
    return list(_flatten([e for e in lines if isinstance(e, Element)], subckts))


def _flatten(elements, subckts, prefix: str = "", pins: dict | None = None):
    for e in elements:
        name = (e.letter + e.name).lower()
        nodes = [str(n) for n in e._nodes]
        if prefix:   # inside a subcircuit: pins map to the instance's nets, the rest get the instance path
            nodes = [pins.get(n, n if n.lower() in ("0", "gnd") else f"{prefix}{n}") for n in nodes]
        model = str(e.parameters[0]).lower() if e.letter.upper() == "X" and e.parameters else ""
        if model in subckts:
            sub = subckts[model]
            yield from _flatten([s for s in sub if isinstance(s, Element)], subckts, f"{prefix}{name}.",
                                dict(zip(map(str, sub._nodes), nodes)))
            continue
        mos = model in MOS_MODELS
        yield SimpleNamespace(
            label=(prefix + e.letter + e.name).upper() if prefix else e.letter.upper() + e.name,
            letter="M" if mos else e.letter.upper(), model=model if mos else "", parameters=e.parameters, _nodes=nodes,
            # ngspice name: X instances join the path with dots, other elements get their letter in front
            spice=f"{prefix}{name}" if mos or not prefix else f"{e.letter.lower()}.{prefix}{name}")


# ── DC operating-point tables ────────────────────────────────────────────────
TERMINALS = {"M": ("d", "g", "s", "b"), "E": ("+", "-", "c+", "c-")}
MOS_BIAS = ("vgs", "vds", "vsb", "vth", "vdss")   # PSP103 operating-point quantities [V]


def _mos_vec(e, param: str) -> str:
    return f"@n.{e.spice}.n{e.model}[{param}]"


def _current_vector(e) -> str | None:
    """ngspice vector holding the DC current of an element (needs `.option savecurrents`)."""
    if e.letter == "M":
        return _mos_vec(e, "ids")
    n = e.spice
    return {"V": f"i({n})", "E": f"i({n})", "I": f"@{n}[current]",
            "R": f"@{n}[i]", "C": f"@{n}[i]", "L": f"@{n}[i]"}.get(e.letter)


def dc_vectors(netlist: str) -> str:
    """Everything `dc_tables` needs, as a space-separated list for `wrdata` after an `op`."""
    elements = parse(netlist)
    nodes = sorted({n for e in elements for n in e._nodes} - GROUND)
    vecs = [f"v({n})" for n in nodes] + [v for e in elements if (v := _current_vector(e))]
    vecs += [_mos_vec(e, p) for e in elements if e.letter == "M" for p in (*MOS_BIAS, "gm", "gds")]
    return " ".join(dict.fromkeys(vecs))


def dc_tables(netlist: str, op) -> tuple:
    """(per-element table: terminals, node voltages, current, MOSFET bias; node-voltage table)."""
    from pandas import DataFrame

    def volt(n):
        return 0.0 if n.lower() in GROUND else op[f"v({n})"][0]

    elements = parse(netlist)
    rows = []
    for e in elements:
        terms = TERMINALS.get(e.letter, ("+", "-"))
        row = {"element": e.label, "type": e.model or e.letter}
        row["nets"] = " ".join(f"{t}={n}" for t, n in zip(terms, e._nodes))
        vec = _current_vector(e)
        row["I [uA]"] = op[column(vec)][0] * 1e6 if vec else float("nan")
        for t, n in zip(terms, e._nodes):
            row[f"V({t}) [V]"] = volt(n)
        if e.letter == "M":
            for p in MOS_BIAS:
                row[f"{p} [V]"] = op[column(_mos_vec(e, p))][0]
            row["gm [uS]"] = op[column(_mos_vec(e, "gm"))][0] * 1e6
            row["gds [uS]"] = op[column(_mos_vec(e, "gds"))][0] * 1e6
            row["region"] = "saturation" if abs(row["vds [V]"]) > abs(row["vdss [V]"]) else "triode"
        rows.append(row)
    nodes = sorted({n for e in elements for n in e._nodes} - GROUND)
    return (DataFrame(rows).set_index("element"),
            DataFrame({"V [V]": [volt(n) for n in nodes]}, index=nodes).rename_axis("node"))


def tran_plot(title: str, t, vin, vout, in_label="vin", out_label="vout", t_scale=1e6, t_unit="us"):
    """Bokeh plot of a transient: input and output on the same voltage axis."""
    from bokeh.plotting import figure

    p = figure(title=title, x_axis_label=f"t [{t_unit}]", y_axis_label="voltage [V]", width=900, height=260)
    p.line(t * t_scale, vin, line_width=2, legend_label=in_label)
    p.line(t * t_scale, vout, line_width=2, color="orange", legend_label=out_label)
    p.legend.location = "top_left"
    return p


def footprint(area_m2: float) -> str:
    """An area as the side of the equivalent square, e.g. `40.1 µm × 40.1 µm`."""
    side = area_m2 ** 0.5 * 1e6
    return f"{side:.1f} µm × {side:.1f} µm"
