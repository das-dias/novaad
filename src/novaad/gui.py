# plotting and user control
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from warnings import warn
from pandas import DataFrame, merge, concat
from numpy import log10, ceil
from itertools import cycle
from novaad import Device


class GuiApp:
    def __init__(self, config: dict, **kwargs):
        self.config = config

    def show_results_table(self, results: DataFrame, **kwargs):
        title = kwargs.get("title", "Device Results Table")
        table_fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("DC-OP", "Sizing", "Electric Model"),
            vertical_spacing=0.03,
            specs=[
                [{"type": "table"}],
                [{"type": "table"}],
                [{"type": "table"}],
            ],
        )

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

        table_fig.add_trace(
            go.Table(
                header=dict(values=dcop_df.columns),
                cells=dict(values=[dcop_df[k].tolist() for k in dcop_df.columns]),
            ),
            row=1,
            col=1,
        )
        table_fig.add_trace(
            go.Table(
                header=dict(values=sizing_df.columns),
                cells=dict(values=[sizing_df[k].tolist() for k in sizing_df.columns]),
            ),
            row=2,
            col=1,
        )
        table_fig.add_trace(
            go.Table(
                header=dict(values=electric_model_df.columns),
                cells=dict(
                    values=[
                        electric_model_df[k].tolist() for k in electric_model_df.columns
                    ]
                ),
            ),
            row=3,
            col=1,
        )

        table_fig.update_layout(
            title=title,
            font=dict(family="Arial, sans-serif", size=14, color="black"),
        )

        table_fig.show()

    def show_device_graphs(self, args, **kwargs):
        tol = kwargs.get("tol", 1e-2)
        lch_tol = kwargs.get("lch-tol", 1e-9)
        verbose = kwargs.get("verbose", 0)

        device_type = args["--type"]
        assert device_type in [
            "nch",
            "pch",
        ], "Invalid device type. Must be 'nch' or 'pch'."
        
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

        layout = go.Layout(
            title=f"Gm/Id Method @ Vds={vds:.2f}V Vsb={vsb:.2f}V ({device_type.upper()})",
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            height=1200,
        )
        plot_df = DataFrame(
            columns=[
                "vgs",
                "gmoverid",
                "jd",
                "ft",
                "av",
                "ft*av",
                "ft*gmoverid",
                "cgg",
                "cgs",
                "cgd",
                "ron",
                "lch",
            ]
        )

        for l in target_lch:
            if verbose > 0:
                print(f"Processing lch={l}...")

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

            plot_df = concat([plot_df, aux_df])

            plot_df["jd_log"] = log10(plot_df["jd"])
            plot_df["av_db"] = 20 * log10(plot_df["av"])
            plot_df["ft_log"] = log10(plot_df["ft"])
            plot_df["ft*av_log"] = log10(plot_df["ft*av"])
            plot_df["ft*gmoverid_log"] = log10(plot_df["ft*gmoverid"])
            plot_df["cgg_fF"] = plot_df["cgg"] / 1e-15
            plot_df["cgs_fF"] = plot_df["cgs"] / 1e-15
            plot_df["cgd_fF"] = plot_df["cgd"] / 1e-15
            plot_df["ron_log"] = log10(plot_df["ron"])

        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=(
                "Gm/Id vs Vgs",
                "Gm/Id vs Jd",
                "Ft vs Jd",
                "Av vs Gm/Id",
                "Fom Av*Bw vs Jd",
                "Fom Noise*Bw vs Jd",
                "Cgg, vs Vgs",
                "Cgs vs Vgs",
                "Cgd vs Vgs",
                "Ron vs Vgs",
            ),
            vertical_spacing=0.1,
        )
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

        for lch in plot_df["lch"].unique():
            lch_df = plot_df[plot_df["lch"] == lch]
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["vgs"],
                    y=lch_df["gmoverid"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["jd_log"],
                    y=lch_df["gmoverid"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["jd_log"],
                    y=lch_df["ft_log"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["gmoverid"],
                    y=lch_df["av_db"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["jd_log"],
                    y=lch_df["ft*av_log"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["jd_log"],
                    y=lch_df["ft*gmoverid_log"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=True,
                ),
                row=3,
                col=2,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["vgs"],
                    y=lch_df["cgg_fF"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["vgs"],
                    y=lch_df["cgs_fF"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=4,
                col=2,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["vgs"],
                    y=lch_df["cgd_fF"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=5,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=lch_df["vgs"],
                    y=lch_df["ron_log"],
                    mode="lines",
                    marker=dict(color=color_map[lch]),
                    name=lch,
                    showlegend=False,
                ),
                row=5,
                col=2,
            )

        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "a": "ᵃ",
            "b": "ᵇ",
            "c": "ᶜ",
            "d": "ᵈ",
            "e": "ᵉ",
            "f": "ᶠ",
            "g": "ᵍ",
            "h": "ʰ",
            "i": "ᶦ",
            "j": "ʲ",
            "k": "ᵏ",
            "l": "ˡ",
            "m": "ᵐ",
            "n": "ⁿ",
            "o": "ᵒ",
            "p": "ᵖ",
            "q": "۹",
            "r": "ʳ",
            "s": "ˢ",
            "t": "ᵗ",
            "u": "ᵘ",
            "v": "ᵛ",
            "w": "ʷ",
            "x": "ˣ",
            "y": "ʸ",
            "z": "ᶻ",
            "A": "ᴬ",
            "B": "ᴮ",
            "C": "ᶜ",
            "D": "ᴰ",
            "E": "ᴱ",
            "F": "ᶠ",
            "G": "ᴳ",
            "H": "ᴴ",
            "I": "ᴵ",
            "J": "ᴶ",
            "K": "ᴷ",
            "L": "ᴸ",
            "M": "ᴹ",
            "N": "ᴺ",
            "O": "ᴼ",
            "P": "ᴾ",
            "Q": "Q",
            "R": "ᴿ",
            "S": "ˢ",
            "T": "ᵀ",
            "U": "ᵁ",
            "V": "ⱽ",
            "W": "ᵂ",
            "X": "ˣ",
            "Y": "ʸ",
            "Z": "ᶻ",
            "+": "⁺",
            "-": "⁻",
            "=": "⁼",
            "(": "⁽",
            ")": "⁾",
        }

        def int_to_super(n: int):
            return "".join([superscript_map[c] for c in str(n)])

        jd_tick_vals = [
            val for val in set([int(v) for v in plot_df["jd_log"].to_numpy()])
        ]
        jd_tick_text = [f"10{int_to_super(int(val))}" for val in jd_tick_vals]

        ft_tick_vals = [
            val for val in set([int(v) for v in plot_df["ft_log"].to_numpy()])
        ]
        ft_tick_text = [f"10{int_to_super(int(val))}" for val in ft_tick_vals]

        ftav_tick_vals = [
            val for val in set([int(v) for v in plot_df["ft*av_log"].to_numpy()])
        ]
        ftav_tick_text = [f"10{int_to_super(int(val))}" for val in ftav_tick_vals]

        ftgmoverid_tick_vals = [
            val for val in set([int(v) for v in plot_df["ft*gmoverid_log"].to_numpy()])
        ]
        ftgmoverid_tick_text = [
            f"10{int_to_super(int(val))}" for val in ftgmoverid_tick_vals
        ]

        ron_tick_vals = [
            val for val in set([int(v) for v in plot_df["ron_log"].to_numpy()])
        ]
        ron_tick_text = [f"10{int_to_super(int(val))}" for val in ron_tick_vals]

        fig.update_xaxes(title_text="Vgs [V]", row=1, col=1)
        fig.update_yaxes(title_text="Gm/Id [1/V]", row=1, col=1)

        fig.update_xaxes(
            title_text="Jd [A/m]",
            tickvals=jd_tick_vals,
            ticktext=jd_tick_text,
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Gm/Id [1/V]", row=1, col=2)

        fig.update_xaxes(
            title_text="Jd [A/m]",
            tickvals=jd_tick_vals,
            ticktext=jd_tick_text,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Ft [Hz]",
            tickvals=ft_tick_vals,
            ticktext=ft_tick_text,
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Gm/Id [1/V]", row=2, col=2)
        fig.update_yaxes(title_text="Av [dB]", row=2, col=2)

        fig.update_xaxes(
            title_text="Jd [A/m]",
            tickvals=jd_tick_vals,
            ticktext=jd_tick_text,
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title_text="Ft*Av [Hz]",
            tickvals=ftav_tick_vals,
            ticktext=ftav_tick_text,
            row=3,
            col=1,
        )

        fig.update_xaxes(
            title_text="Jd [A/m]",
            tickvals=jd_tick_vals,
            ticktext=jd_tick_text,
            row=3,
            col=2,
        )
        fig.update_yaxes(
            title_text="Ft*Gm/Id [Hz]",
            tickvals=ftgmoverid_tick_vals,
            ticktext=ftgmoverid_tick_text,
            row=3,
            col=2,
        )

        fig.update_xaxes(title_text="Vgs [V]", row=4, col=1)
        fig.update_yaxes(title_text="Cgg [fF]", row=4, col=1)

        fig.update_xaxes(title_text="Vgs [V]", row=4, col=2)
        fig.update_yaxes(title_text="Cgs [fF]", row=4, col=2)

        fig.update_xaxes(title_text="Vgs [V]", row=5, col=1)
        fig.update_yaxes(title_text="Cgd [fF]", row=5, col=1)

        fig.update_xaxes(title_text="Vgs [V]", row=5, col=2)
        fig.update_yaxes(
            title_text="Ron [Ω]",
            tickvals=ron_tick_vals,
            ticktext=ron_tick_text,
            row=5,
            col=2,
        )

        fig.update_layout(layout)
        fig.show()
