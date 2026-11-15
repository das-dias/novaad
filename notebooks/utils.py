"""Parse SPICE testbenches with InSpice into DC operating-point tables, plus plotting helpers."""

import contextlib
import re
import io

from InSpice.Spice.Parser import Ast, SpiceSource
from InSpice.Spice.Parser.HighLevelParser import Element

GROUND = {"0", "gnd", "vss"}
ENG = [(1e9, "G"), (1e6, "Meg"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]


def parse(netlist: str) -> list[Element]:
    """Element lines of an ngspice netlist, parsed by InSpice (dot commands and .control are skipped)."""
    # the .control block is ngspice scripting, not circuit: InSpice can't parse all of it (e.g. `$&var`, `^`)
    netlist = re.sub(r"^\.control\b.*?^\.endc\b", "", netlist, flags=re.S | re.M | re.I)
    # InSpice's PLY parser prints its grammar-table warnings on first use
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        source = SpiceSource(netlist)
    return [line for line in source.obj_lines if isinstance(line, Element)]


# ── DC operating-point tables ────────────────────────────────────────────────
TERMINALS = {"M": ("d", "g", "s", "b"), "E": ("+", "-", "c+", "c-")}
MOS_PARAMS = ("id", "vgs", "vds", "vbs", "vth", "vdsat", "gm", "gds")


def _current_vector(e: Element) -> str | None:
    """ngspice vector holding the DC current of an element (needs `.option savecurrents`)."""
    name = (e.letter + e.name).lower()
    return {"V": f"i({name})", "E": f"i({name})", "I": f"@{name}[current]",
            "R": f"@{name}[i]", "C": f"@{name}[i]", "L": f"@{name}[i]", "M": f"@{name}[id]"}.get(e.letter.upper())


def dc_vectors(netlist: str) -> str:
    """Everything `dc_tables` needs, as a space-separated list for `wrdata` after an `op`."""
    elements = parse(netlist)
    nodes = sorted({str(n) for e in elements for n in e._nodes} - GROUND)
    vecs = [f"v({n})" for n in nodes] + [v for e in elements if (v := _current_vector(e))]
    vecs += [f"@{(e.letter + e.name).lower()}[{p}]" for e in elements if e.letter.upper() == "M" for p in MOS_PARAMS[1:]]
    return " ".join(dict.fromkeys(vecs))


def dc_tables(netlist: str, op) -> tuple:
    """(per-element table: terminals, node voltages, current, MOSFET bias; node-voltage table)."""
    from pandas import DataFrame

    def volt(n):
        return 0.0 if str(n).lower() in GROUND else op[f"v({n})"][0]

    def col(vec):  # spice.read renames "@m1[id]" -> "m1.id"
        return vec[1:].replace("[", ".").rstrip("]") if vec.startswith("@") else vec

    rows = []
    for e in parse(netlist):
        name, letter = e.letter.upper() + e.name, e.letter.upper()
        terms = TERMINALS.get(letter, ("+", "-"))
        row = {"element": name, "type": _fmt(e.parameters[0]) if letter == "M" else letter}
        row["nets"] = " ".join(f"{t}={n}" for t, n in zip(terms, e._nodes))
        vec = _current_vector(e)
        row["I [uA]"] = op[col(vec)][0] * 1e6 if vec else float("nan")
        for t, n in zip(terms, e._nodes):
            row[f"V({t}) [V]"] = volt(n)
        if letter == "M":
            m = name.lower()
            for p in MOS_PARAMS[1:6]:
                row[f"{p} [V]"] = op[f"{m}.{p}"][0]
            row["gm [uS]"], row["gds [uS]"] = op[f"{m}.gm"][0] * 1e6, op[f"{m}.gds"][0] * 1e6
            row["region"] = "saturation" if abs(row["vds [V]"]) > abs(row["vdsat [V]"]) else "triode"
        rows.append(row)
    nodes = sorted({str(n) for e in parse(netlist) for n in e._nodes} - GROUND)
    return (DataFrame(rows).set_index("element"),
            DataFrame({"V [V]": [volt(n) for n in nodes]}, index=nodes).rename_axis("node"))


def _fmt(p) -> str:
    """A parameter as written in the netlist: `7.6u`, `200k`, `nfet_03v3`, `sin(...)`."""
    if isinstance(p, Ast.Set):
        return _fmt(p.right)
    if isinstance(p, Ast.Number) and p.unit:
        return f"{p.value:g}{p.unit}"
    if isinstance(p, Ast.Number):
        p = p.value
    if isinstance(p, Ast.Function):
        return f"{p.name}(...)" if hasattr(p, "name") else "f(...)"
    s = str(p).strip("'") if not isinstance(p, (int, float)) else repr(p)
    try:
        x = float(s)
    except ValueError:
        return s
    scale, sym = next(((k, u) for k, u in ENG if abs(x) >= k), (1, "") if x == 0 else (1e-15, "f"))
    return f"{x / scale:.3g}{sym}"


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
