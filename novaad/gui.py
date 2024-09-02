# plotting and user control
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from warnings import warn
from pandas import DataFrame, merge, concat
from numpy import log10, ceil, zeros
from itertools import cycle
from novaad import Device, SizingSpecification, DcOp, ElectricModel, Sizing


class GuiApp:
  def __init__(self, device:Device, **kwargs):
    self.device = device
    
  def run_device_sizing(self, args, dcop: DcOp, sizing: Sizing, electric_model: ElectricModel, **kwargs):
    tol = kwargs.get('tol', 1e-2)
    lch_tol = kwargs.get('lch-tol', 1e-9)
    verbose = kwargs.get('verbose', 0)
    
    # Update plots
    table_fig = make_subplots(
      rows=3, cols=1, 
      subplot_titles=(
        "DC-OP",
        "Sizing",
        "Electric Model"
      ),
      vertical_spacing=0.03,
      specs=[[{"type": "table"}],
             [{"type": "table"}],
             [{"type": "table"}],
            ]
    )
    
    dcop_df = dcop.to_df()
    sizing_df = sizing.to_df()
    electric_model_df = electric_model.to_df()
    
    # format info for output to SI units
    dcop_df['ids'] = dcop_df['ids'].apply(lambda x: f"{x/1e-6:.4}")
    dcop_df['vgs'] = dcop_df['vgs'].apply(lambda x: f"{x:.2}")
    dcop_df['vds'] = dcop_df['vds'].apply(lambda x: f"{x:.2}")
    dcop_df['vsb'] = dcop_df['vsb'].apply(lambda x: f"{x:.2}")
    
    dcop_df = dcop_df.rename(columns={
      'vgs': 'Vgs [V]',
      'vds': 'Vds [V]',
      'vsb': 'Vsb [V]',
      'ids': 'Ids [uA]',
    })
    
    sizing_df['wch'] = sizing_df['wch'].apply(lambda x: f"{x/1e-6:.4}")
    sizing_df['lch'] = sizing_df['lch'].apply(lambda x: f"{x/1e-9:.4}")
    
    sizing_df = sizing_df.rename(columns={
      'wch': 'Wch [um]',
      'lch': 'Lch [nm]',
    })
    
    electric_model_df['gm'] = electric_model_df['gm'].apply(lambda x: f"{x/1e-3:.4}") \
      if 'gm' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['gds'] = electric_model_df['gds'].apply(lambda x: f"{x/1e-6:.4}") \
      if 'gds' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['ft'] = electric_model_df['ft'].apply(lambda x: f"{x/1e9:.4}") \
      if 'ft' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['av'] = electric_model_df['av'].apply(lambda x: f"{20*log10(x):.4}") \
      if 'av' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['jd'] = electric_model_df['jd'].apply(lambda x: f"{x:.2e}") \
      if 'jd' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['cgs'] = electric_model_df['cgs'].apply(lambda x: f"{x/1e-15:.4}") \
      if 'cgs' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['cgd'] = electric_model_df['cgd'].apply(lambda x: f"{x/1e-15:.4}") \
      if 'cgd' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['cgb'] = electric_model_df['cgb'].apply(lambda x: f"{x/1e-15:.4}") \
      if 'cgb' in electric_model_df.columns else zeros(len(electric_model_df))
    electric_model_df['cgg'] = electric_model_df['cgg'].apply(lambda x: f"{x/1e-15:.4}") \
      if 'cgg' in electric_model_df.columns else zeros(len(electric_model_df))
    
    electric_model_df = electric_model_df.rename(columns={
      'gm': 'Gm [mS]',
      'gds': 'Gds [uS]',
      'ft': 'Ft [GHz]',
      'av': 'Av [dB]',
      'jd': 'Jd [F/m]',
      'cgs': 'Cgs [fF]',
      'cgd': 'Cgd [fF]',
      'cgb': 'Cgb [fF]',
      'cgg': 'Cgg [fF]',
    })
    
    table_fig.add_trace(
      go.Table(
        header=dict(values=dcop_df.columns),
        cells=dict(values=[dcop_df[k].tolist() for k in dcop_df.columns])
      ), row=1, col=1
    )
    table_fig.add_trace(
      go.Table(
        header=dict(values=sizing_df.columns),
        cells=dict(values=[sizing_df[k].tolist() for k in sizing_df.columns])
      ), row=2, col=1
    )
    table_fig.add_trace(
      go.Table(
        header=dict(values=electric_model_df.columns),
        cells=dict(values=[electric_model_df[k].tolist() for k in electric_model_df.columns])
      ), row=3, col=1
    )
    table_fig.show()
    
    # Update graph plots
    
    target_lch = [float(l) for l in args['<lch-plot>']] if args['--lch-plot'] else ['all']
    if target_lch[0] == 'all':
      target_lch = self.device.lut['lch'].unique().tolist()
      
    assert all([l in self.device.lut['lch'].unique() for l in target_lch]), f"Invalid channel length. \
      Interpolated values are not supported for Graph visualization. \
        Please use a valid channel length: {self.device.lut['lch'].unique()}"
    
    sizing_spec = SizingSpecification(
      vgs=dcop.vgs,
      vds=dcop.vds,
      vsb=dcop.vsb,
    )
    
    vds_vsb_pairs = set([
      (vds, vsb) for vds in sizing_spec.to_df()['vds'] for vsb in sizing_spec.to_df()['vsb']])
    
    for vds, vsb in vds_vsb_pairs:
      if vds not in self.device.lut['vds'].unique():
        warn(f"Invalid Vds value: {vds}. Using nearest value.")
        vds = self.device.lut['vds'][(self.device.lut['vds']-vds).abs().argsort()[0]]
      if vsb not in self.device.lut['vsb'].unique():
        warn(f"Invalid Vsb value: {vsb}. Using nearest value.")
        vsb = self.device.lut['vsb'][(self.device.lut['vsb']-vsb).abs().argsort()[0]]
      
      fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=(
          "Gm/Id vs Vgs",
          "Gm/Id vs Jd",
          "Ft vs Jd",
          "Av vs Gm/Id",
          "Fom Av*Bw vs Jd",
          "Fom Noise*Bw vs Jd"
        )
      )
      
      layout = go.Layout(
        title=f"Gm/Id Method @ Vds={vds:.2f}V Vsb={vsb:.2f}V",
        font=dict(family="Arial, sans-serif", size=14, color="black")
      )    
      plot_df = DataFrame(
        columns=['vgs', 'gmoverid', 'jd', 
                'ft', 'av', 'ft*av', 'ft*gmoverid', 'lch']
      )
      
      for l in target_lch:
        if verbose > 0:
          print(f"Processing lch={l}...")
          
        lch = int(ceil(l/1e-9))
        query = f"abs(vds-{vds})<={tol} & abs(vsb-{vsb})<={tol}"
        query = f"abs(lch-{l})<={lch_tol} & {query}"
        gm_id_vs_vgs = self.device.wave_vs_wave("gmoverid", "vgs", query=query)
        gm_id_vs_jd = self.device.wave_vs_wave("gmoverid", "jd", query=query)
        ft_vs_jd = self.device.wave_vs_wave("ft", "jd", query=query)
        av_vs_gm_id = self.device.wave_vs_wave("av", "gmoverid", query=query)
        fom_av_bw_vs_jd = self.device.wave_vs_wave("ft*av", "jd", query=query)
        fom_noise_bw_vs_jd = self.device.wave_vs_wave("ft*gmoverid", "jd", query=query)
        
        aux_df = gm_id_vs_vgs
        aux_df = merge(aux_df, gm_id_vs_jd, how='outer')
        aux_df = merge(aux_df, ft_vs_jd, how='outer')
        aux_df = merge(aux_df, av_vs_gm_id, how='outer')
        aux_df = merge(aux_df, fom_av_bw_vs_jd, how='outer')
        aux_df = merge(aux_df, fom_noise_bw_vs_jd, how='outer')
        aux_df['lch'] = str(lch) + ' nm'
        plot_df = concat([plot_df,aux_df])
        
      plot_df['jd_log'] = log10(plot_df['jd'])
      plot_df['av_db'] = 20*log10(plot_df['av'])
      plot_df['ft_log'] = log10(plot_df['ft'])
      plot_df['ft*av_log'] = log10(plot_df['ft*av'])
      plot_df['ft*gmoverid_log'] = log10(plot_df['ft*gmoverid'])
      fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=(
          "Gm/Id vs Vgs",
          "Gm/Id vs Jd",
          "Ft vs Jd",
          "Av vs Gm/Id",
          "FoM Gain-Bandwidth vs Jd",
          "FoM Noise-Bandwidth vs Jd"
        )
      )
      colors = [
        '#000000',
        '#E69F00',
        '#56B4E9',
        '#009E73',
        '#F0E442',
        '#0072B2',
        '#0072B2',
        '#CC79A7',
      ] 
      colors = colors if kwargs.get('colors', None) is None \
        else kwargs.get('colors') \
            if isinstance(kwargs.get('colors'), list) else None
      assert colors is not None, "Colors must be a list."
      colors = cycle(colors)
      color_map = {
        k: next(colors) for k in plot_df['lch'].unique()
      }
      
      for lch in plot_df['lch'].unique():
        lch_df = plot_df[plot_df['lch'] == lch]
        fig.add_trace(go.Scattergl(
          x=lch_df['vgs'], y=lch_df['gmoverid'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(
          x=lch_df['jd_log'], y=lch_df['gmoverid'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scattergl(
          x=lch_df['jd_log'], y=lch_df['ft_log'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(
          x=lch_df['gmoverid'], y=lch_df['av_db'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=False), row=2, col=2)
        fig.add_trace(go.Scattergl(
          x=lch_df['jd_log'], y=lch_df['ft*av_log'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(
          x=lch_df['jd_log'], y=lch_df['ft*gmoverid_log'], mode='lines', marker=dict(color=color_map[lch]), name=lch, showlegend=True), row=3, col=2)
      
      fig.update_xaxes(title_text="Vgs [V]", row=1, col=1)
      fig.update_yaxes(title_text="Gm/Id [1/V]", row=1, col=1)
      
      superscript_map = {
      "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
      "7": "⁷", "8": "⁸", "9": "⁹", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ",
      "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ", "k": "ᵏ",
      "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "q": "۹", "r": "ʳ",
      "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
      "z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ",
      "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
      "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "Q": "Q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
      "U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ", "+": "⁺",
      "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾"}
      
      def int_to_super(n:int):
        return "".join([superscript_map[c] for c in str(n)])
      
      jd_tick_vals = [val for val in set([int(v) for v in plot_df['jd_log'].to_numpy()])]
      jd_tick_text = [f"10{int_to_super(int(val))}" for val in jd_tick_vals]
      
      ft_tick_vals = [val for val in set([int(v) for v in plot_df['ft_log'].to_numpy()])]
      ft_tick_text = [f"10{int_to_super(int(val))}" for val in ft_tick_vals]
      
      ftav_tick_vals = [val for val in set([int(v) for v in plot_df['ft*av_log'].to_numpy()])]
      ftav_tick_text = [f"10{int_to_super(int(val))}" for val in ftav_tick_vals]
      
      ftgmoverid_tick_vals = [val for val in set([int(v) for v in plot_df['ft*gmoverid_log'].to_numpy()])]
      ftgmoverid_tick_text = [f"10{int_to_super(int(val))}" for val in ftgmoverid_tick_vals]
      
      fig.update_xaxes(title_text="Jd [A/m]", 
                      tickvals=jd_tick_vals,
                      ticktext=jd_tick_text , row=1, col=2)
      fig.update_yaxes(title_text="Gm/Id [1/V]", row=1, col=2)
      
      fig.update_xaxes(title_text="Jd [A/m]", 
                      tickvals=jd_tick_vals,
                      ticktext=jd_tick_text, row=2, col=1)
      fig.update_yaxes(title_text="Ft [Hz]", 
                      tickvals=ft_tick_vals,
                      ticktext=ft_tick_text, row=2, col=1)
      
      fig.update_xaxes(title_text="Gm/Id [1/V]", row=2, col=2)
      fig.update_yaxes(title_text="Av [dB]", row=2, col=2)
      
      fig.update_xaxes(title_text="Jd [A/m]", 
                      tickvals=jd_tick_vals,
                      ticktext=jd_tick_text, row=3, col=1)
      fig.update_yaxes(title_text="Ft*Av [Hz]", 
                      tickvals=ftav_tick_vals,
                      ticktext=ftav_tick_text, row=3, col=1)
      
      fig.update_xaxes(title_text="Jd [A/m]", 
                      tickvals=jd_tick_vals,
                      ticktext=jd_tick_text, row=3, col=2)
      fig.update_yaxes(title_text="Ft*Gm/Id [Hz]", 
                      tickvals=ftgmoverid_tick_vals,
                      ticktext=ftgmoverid_tick_text, row=3, col=2)
      
      fig.update_layout(layout)
      fig.show()
      
  def run_moscap_sizing(self, args, **kwargs):
    raise NotImplementedError("MOSCAP sizing not implemented.")
  
  def run_switch_sizing(self, args, **kwargs):
    raise NotImplementedError("Switch sizing not implemented.")
    
# Run the app
if __name__ == '__main__':
  
  cfg = {
    'nch': {
      'lut-path': '/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv',
      'varmap': {
        'vgs': 'vgs_n',
        'vds': 'vds_n',
        'vsb': 'vsb_n',
        'lch': 'length_wave',
        'wch': 'wch_n',
        'gmoverid': 'gmoverid_n',
        'gm': 'gm_n',
        'ids': 'id_n',
        'ft': 'ft_n',
        'av': 'av_n',
        'jd': 'jd_n',
        'cgs': 'cgs_n',
        'cgd': 'cgd_n',
        'cgb': 'cgb_n',
        'cgg': 'cgg_n',
        'qg': 'qg_n',
        'qd': 'qd_n',
        'qs': 'qs_n',
      },
    },
    'ref-width': 10e-6,
  }
  app = GuiApp(cfg)
  args = {
    '--gui': True,
    '--vgs': '0.8',
    '--vds': '0.3',
    '--vsb': '0.0',
    '--lch': '180e-9',
    '--ids': None,
    '--gm': '1e-3',
    '--wch': None,
    '--gmid': '10',
    '--nch': True,
    '--pch': False,
    '--verbose': 1
  }
  
  args2 = {
    '--gui': True,
    '--vgs': ['0.8', '0.9'],
    '--vds': ['0.3', '0.4'],
    '--vsb': ['0.0', '0.0'],
    '--lch': ['180e-9', '1e-6'],
    '--ids': None,
    '--gm': ['1e-3', '1e-3'],
    '--wch': None,
    '--gmid': ['10', '10'],
    '--nch': True,
    '--pch': False,
    '--verbose': 1
  }
  
  args2 = {
    '--gui': True,
    '--vgs': ['0.8', '0.9'],
    '--vds': ['0.3', '0.4'],
    '--vsb': ['0.0', '0.0'],
    '--lch': ['180e-9', '1e-6'],
    '--ids': None,
    '--gm': ['1e-3', '1e-3'],
    '--wch': None,
    '--gmid': ['10', '10'],
    '--nch': True,
    '--pch': False,
    '--lch-plot': 'all',
    '--verbose': 1
  }
  
  app.run(args2, verbose=1, tol=1e-2)