# plotting and user control
from pathlib import Path
from tempfile import gettempdir
from warnings import warn
from itertools import cycle
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Div, CrosshairTool, Span
from pandas import DataFrame, merge, concat
from numpy import log10, ceil
from novaad import Device, cgui

# (title, x, y, x_label, y_label, x_log, y_log)
GRAPHS = [
    ("Gm/Id vs Vgs", "vgs", "gmoverid", "Vgs [V]", "Gm/Id [1/V]", False, False),
    ("Gm/Id vs Jd", "jd", "gmoverid", "Jd [A/m]", "Gm/Id [1/V]", True, False),
    ("Ft vs Jd", "jd", "ft", "Jd [A/m]", "Ft [Hz]", True, True),
    ("Av vs Gm/Id", "gmoverid", "av_db", "Gm/Id [1/V]", "Av [dB]", False, False),
    ("Fom Av*Bw vs Jd", "jd", "ft*av", "Jd [A/m]", "Ft*Av [Hz]", True, True),
    ("Fom Noise*Bw vs Jd", "jd", "ft*gmoverid", "Jd [A/m]", "Ft*Gm/Id [Hz]", True, True),
    ("Cgg, vs Vgs", "vgs", "cgg_fF", "Vgs [V]", "Cgg [fF]", False, False),
    ("Cgs vs Vgs", "vgs", "cgs_fF", "Vgs [V]", "Cgs [fF]", False, False),
    ("Cgd vs Vgs", "vgs", "cgd_fF", "Vgs [V]", "Cgd [fF]", False, False),
    ("Ron vs Vgs", "vgs", "ron", "Vgs [V]", "Ron [Ω]", False, True),
]
LEGEND_GRAPH = "Fom Noise*Bw vs Jd"


def _table(df: DataFrame) -> DataTable:
    return DataTable(
        source=ColumnDataSource(df),
        columns=[TableColumn(field=c, title=c) for c in df.columns],
        index_position=None,
        sizing_mode="stretch_width",
        height=30 + 25 * len(df),
    )


class GuiApp:
    def __init__(self, config: dict, **kwargs):
        self.config = config

    def show_results_table(self, results: DataFrame, **kwargs):
        title = kwargs.get("title", "Device Results Table")
        dcop_df = DataFrame(
            data={
                "ID": results["id"],
                "Type": results["type"],
                "Vgs [V]": results["vgs"].apply(lambda x: f"{x:.2f}"),
                "Vds [V]": results["vds"].apply(lambda x: f"{x:.2f}"),
                "Vsb [V]": results["vsb"].apply(lambda x: f"{x:.2f}"),
                "Ids [uA]": results["ids"].apply(
                    lambda x: f"{x/1e-6:.4f}" if x is not None else "N/A"
                ),
            }
        )

        sizing_df = DataFrame(
            data={
                "ID": results["id"],
                "Type": results["type"],
                "Wch [um]": results["wch"].apply(lambda x: f"{x/1e-6:.4f}"),
                "Lch [nm]": results["lch"].apply(lambda x: f"{x/1e-9:.4f}"),
            }
        )

        electric_model_df = DataFrame(
            data={
                "ID": results["id"],
                "Type": results["type"],
                "Gm/Id [1/V]": results["gmoverid"].apply(
                    lambda x: f"{x:.4f}" if x is not None else "N/A"
                ),
                "Gm [uS]": results["gm"].apply(
                    lambda x: f"{x/1e-6:.4f}" if x is not None else "N/A"
                ),
                "Gds [uS]": results["gds"].apply(
                    lambda x: f"{x/1e-6:.4f}" if x is not None else "N/A"
                ),
                "Cgg [fF]": results["cgg"].apply(
                    lambda x: f"{x/1e-15:.4f}" if x is not None else "N/A"
                ),
                "Cgs [fF]": results["cgs"].apply(
                    lambda x: f"{x/1e-15:.4f}" if x is not None else "N/A"
                ),
                "Cgd [fF]": results["cgd"].apply(
                    lambda x: f"{x/1e-15:.4f}" if x is not None else "N/A"
                ),
                "Csb [fF]": results["csb"].apply(
                    lambda x: f"{x/1e-15:.4f}" if x is not None else "N/A"
                ),
                "Cdb [fF]": results["cdb"].apply(
                    lambda x: f"{x/1e-15:.4f}" if x is not None else "N/A"
                ),
                "Ron [Ω]": results["ron"].apply(
                    lambda x: f"{x:.2f}" if x is not None else "N/A"
                ),
                "Av [dB]": results["av"].apply(
                    lambda x: f"{20*log10(x):.4f}" if x is not None else "N/A"
                ),
                "Ft [GHz]": results["ft"].apply(
                    lambda x: f"{x/1e9:.4f}" if x is not None else "N/A"
                ),
                "FoM Av*Ft [GHz]": results["fom_bw"].apply(
                    lambda x: f"{x/1e9:.2f}" if x is not None else "N/A"
                ),
                "FoM (Gm/Id)*Ft [GHz/V]": results["fom_nbw"].apply(
                    lambda x: f"{x/1e9:.4f}" if x is not None else "N/A"
                ),
                "Vng,rms [uV]": results["vng_rms"].apply(
                    lambda x: f"{x/1e-6:.4f}" if x is not None else "N/A"
                ),
            }
        )

        output_file(Path(gettempdir()) / "novaad_results.html", title=title)
        show(
            column(
                Div(text=f"<h2>{title}</h2>"),
                Div(text="<h3>DC-OP</h3>"),
                _table(dcop_df),
                Div(text="<h3>Sizing</h3>"),
                _table(sizing_df),
                Div(text="<h3>Electric Model</h3>"),
                _table(electric_model_df),
                sizing_mode="stretch_width",
            )
        )

    def show_device_graphs(self, args, **kwargs):
        tol = kwargs.get("tol", 1e-2)
        lch_tol = kwargs.get("lch-tol", 1e-9)

        device_type = args["--type"]
        assert device_type in [
            "nch",
            "pch",
        ], "Invalid device type. Must be 'nch' or 'pch'."
        
        with cgui.status(f"Loading {device_type} LUT..."):
            self.device = Device(
                lut_path=self.config[device_type]["lut_path"],
                device_type=device_type,
                ref_width=float(self.config[device_type]["ref_width"]),
            )

        target_lch = (
            [float(l) for l in args["LCH_PLOT"]] if args["--lch-plot"] else ["all"]
        )
        if target_lch[0] == "all":
            target_lch = self.device.lut["lch"].unique().tolist()

        assert all(
            [l in self.device.lut["lch"].unique() for l in target_lch]
        ), f"Invalid channel length. \
      Interpolated values are not supported for Graph visualization. \
        Please use a valid channel length: {self.device.lut['lch'].unique()}"

        vds = float(args["--vds"]) if args["--vds"] else self.device.lut["vds"].mean()
        vsb = float(args["--vsb"]) if args["--vsb"] else self.device.lut["vsb"].min()

        if vds not in self.device.lut["vds"].unique():
            warn(f"Invalid Vds value: {vds}. Using nearest value.")
            vds = self.device.lut["vds"][
                (self.device.lut["vds"] - vds).abs().argsort()[0]
            ]
        if vsb not in self.device.lut["vsb"].unique():
            warn(f"Invalid Vsb value: {vsb}. Using nearest value.")
            vsb = self.device.lut["vsb"][
                (self.device.lut["vsb"] - vsb).abs().argsort()[0]
            ]

        frames = []

        for l in cgui.progress(target_lch, description="Extracting curves..."):
            lch = int(ceil(l / 1e-9))
            query = f"abs(vds-{vds})<={tol} & abs(vsb-{vsb})<={tol}"
            query = f"abs(lch-{l})<={lch_tol} & {query}"
            gm_id_vs_vgs = self.device.wave_vs_wave("gmoverid", "vgs", query=query)
            gm_id_vs_jd = self.device.wave_vs_wave("gmoverid", "jd", query=query)
            ft_vs_jd = self.device.wave_vs_wave("ft", "jd", query=query)
            av_vs_gm_id = self.device.wave_vs_wave("av", "gmoverid", query=query)
            fom_av_bw_vs_jd = self.device.wave_vs_wave("ft*av", "jd", query=query)
            fom_noise_bw_vs_jd = self.device.wave_vs_wave(
                "ft*gmoverid", "jd", query=query
            )
            cgg_vs_vgs = self.device.wave_vs_wave("cgg", "vgs", query=query)
            cgs_vs_vgs = self.device.wave_vs_wave("cgs", "vgs", query=query)
            cgd_vs_vgs = self.device.wave_vs_wave("cgd", "vgs", query=query)
            ron_vs_vgs = self.device.wave_vs_wave("ron", "vgs", query=query)

            aux_df = gm_id_vs_vgs
            aux_df = merge(aux_df, gm_id_vs_jd)
            aux_df = merge(aux_df, ft_vs_jd)
            aux_df = merge(aux_df, av_vs_gm_id)
            aux_df = merge(aux_df, fom_av_bw_vs_jd)
            aux_df = merge(aux_df, fom_noise_bw_vs_jd)
            aux_df = merge(aux_df, cgg_vs_vgs)
            aux_df = merge(aux_df, cgs_vs_vgs)
            aux_df = merge(aux_df, cgd_vs_vgs)
            aux_df = merge(aux_df, ron_vs_vgs)
            aux_df["lch"] = str(lch) + " nm"

            frames.append(aux_df)

        plot_df = concat(frames)
        plot_df["av_db"] = 20 * log10(plot_df["av"])
        plot_df["cgg_fF"] = plot_df["cgg"] / 1e-15
        plot_df["cgs_fF"] = plot_df["cgs"] / 1e-15
        plot_df["cgd_fF"] = plot_df["cgd"] / 1e-15

        colors = [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#0072B2",
            "#CC79A7",
        ]
        colors = (
            colors
            if kwargs.get("colors", None) is None
            else (
                kwargs.get("colors") if isinstance(kwargs.get("colors"), list) else None
            )
        )
        assert colors is not None, "Colors must be a list."
        colors = cycle(colors)
        color_map = {k: next(colors) for k in plot_df["lch"].unique()}

        x_ranges = {}  # graphs with the same x variable share zoom/pan
        vspans = {}  # ... and the vertical crosshair line
        figs = {}
        for title, x, y, x_label, y_label, x_log, y_log in GRAPHS:
            p = figure(
                title=title,
                x_axis_type="log" if x_log else "linear",
                y_axis_type="log" if y_log else "linear",
                x_axis_label=x_label,
                y_axis_label=y_label,
                output_backend="webgl",
                tooltips=[("Lch", "@lch"), (x_label, "$x"), (y_label, "$y")],
                width=600,
                height=240,
            )
            if x in x_ranges:
                p.x_range = x_ranges[x]
            x_ranges.setdefault(x, p.x_range)
            vspan = Span(dimension="height", line_dash="dashed")
            hspan = Span(dimension="width", line_dash="dashed")
            p.add_tools(CrosshairTool(overlay=(vspan, hspan)))
            for other in vspans.setdefault(x, []):
                vspan.js_link("location", other, "location")
                other.js_link("location", vspan, "location")
            vspans[x].append(vspan)
            figs[title] = p

        for lch in plot_df["lch"].unique():
            src = ColumnDataSource(plot_df[plot_df["lch"] == lch])
            renderers = {}
            for title, x, y, *_ in GRAPHS:
                kw = {"legend_label": lch} if title == LEGEND_GRAPH else {}
                renderers[title] = figs[title].line(
                    x, y, source=src, color=color_map[lch], **kw
                )
            # legend click hides this lch in every graph, like plotly's global legend
            for title, r in renderers.items():
                if title != LEGEND_GRAPH:
                    renderers[LEGEND_GRAPH].js_link("visible", r, "visible")
        figs[LEGEND_GRAPH].legend.click_policy = "hide"

        title = f"Gm/Id Method @ Vds={vds:.2f}V Vsb={vsb:.2f}V ({device_type.upper()})"
        output_file(Path(gettempdir()) / "novaad_graphs.html", title=title)
        show(
            column(
                Div(text=f"<h2>{title}</h2>"),
                gridplot(list(figs.values()), ncols=2),
            )
        )
