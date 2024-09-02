"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    
    novaad --nch --vgs=0.8 --vds=0.5 --vsb=0.0 --lch=180e-9 --gmid=10 --gm=1e-3

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    
    novaad --pch --vgs=0.8 --vds=0.5 --vsb=0.0 --ids=500e-6 --lch=180e-9 --wch=18e-6

Usage:
  novaad (--nch | --pch) [--gui] [ --lch-plot <lch-plot> ... ] [--vgs <vgs> ... --vds <vds> ... --vsb <vsb> ... --lch <lch> ... ( --wch <wch> ... |  --gmid <gmid> ... (--ids <ids> ... | --gm <gm> ...) | --ron <ron> ... | --cgg <cgg> ...)] [--verbose]
  novaad COMMAND_FILE
  novaad (-h | --help)
  novaad --version
  novaad --setup-config <cfg-path>

Options:
  -h --help       Show this screen.
  --version       Show version.
  --vgs           Gate-Source Voltage.
  --vds           Drain-Source Voltage.
  --vsb           Substrate Voltage.
  --lch           Channel Length.
  --gmid          Transconductance Efficiency.
  --ids           Drain Current.
  --wch           Channel Width.
  --cgg           Gate-Source Capacitance.
  --ron           On-Resistance.
  COMMAND_FILE    Input Command File.
  --gui           Launch GUI.
  --lch-plot      Channel lengths to include in plot [default: all].
  --verbose       Verbose Output.
  --setup-config  Set Configuration File Path.
"""

from docopt import docopt, DocoptExit
from warnings import warn
from pathlib import Path
from pprint import pprint

from yaml import safe_load, safe_dump
from numpy import log10, zeros
from novaad import Device, SizingSpecification, DcOp, Sizing, GuiApp

import pdb

global __REF_CWD__
__REF_CWD__ = str(Path(__file__).resolve().cwd())

global __DEFAULT_CFG_PATH__
__DEFAULT_CFG_PATH__ = str(Path(__REF_CWD__+'/cfg.yml').resolve())
global CFG_PATH
CFG_PATH = __DEFAULT_CFG_PATH__
  
def device_sizing(args, cfg) -> bool:
  verbose = 0
  if args['--verbose']:
    verbose = 1
  device_type = None
  if args['--nch']:
    device_type = 'nch'
  elif args['--pch']:
    device_type = 'pch'
  else: raise ValueError("Device type not found in arguments.")
  device = None
  device_cfg = cfg.get(device_type, None)
  assert device_cfg is not None, "Device ('nch' | 'pch') not found in configuration."
  lut_path = device_cfg.get('lut-path', None)
  assert lut_path is not None, "Device 'lut-path' not found in configuration."
  lut_path = Path(lut_path).resolve()
  assert lut_path is not None, "Device 'lut-path' was not resolved."
  lut_varmap = device_cfg.get('varmap', None)
  if lut_varmap is not None:
    lut_varmap = {v: k for k, v in lut_varmap.items()}
  
  bsim4_params_path = Path(device_cfg.get('bsim4-params-path', None)).resolve()
  bsim4_params_varmap = device_cfg.get('bsim4-params-varmap', None)
  if bsim4_params_varmap is not None:
    bsim4_params_varmap = {k: v for k, v in bsim4_params_varmap.items()}
  reference_width = cfg.get('ref-width', None)
  if reference_width is None:
    warn("Reference width not found in configuration.")
    warn("Using default reference width of 10 um.")
    reference_width = 10e-6
  reference_width = float(reference_width)
  device = Device(
    lut_path, 
    lut_varmap=lut_varmap, 
    bsim4params_path=bsim4_params_path, 
    bsim4params_varmap=bsim4_params_varmap, 
    ref_width=reference_width,
    device_type=device_type
  )
  
  sizing_spec = SizingSpecification(
    vgs=[float(v) for v in args['<vgs>']] if args['--vgs'] else [device.lut['vgs'].mean().astype(float)],
    vds=[float(v) for v in args['<vds>']] if args['--vds'] else [device.lut['vds'].mean().astype(float)],
    vsb=[float(v) for v in args['<vsb>']] if args['--vsb'] else [device.lut['vsb'].min().astype(float)],
    lch=[float(v) for v in args['<lch>']] if args['--lch'] else [device.lut['lch'].min().astype(float)],
    gmoverid=[float(v) for v in args['<gmid>']] if args['--gmid'] else [device.lut['gmoverid'].mean().astype(float)],
    ids = [float(v) for v in args['<ids>']] if args['--ids'] else None,
    gm = [float(v) for v in args['<gm>']] if args['--gm'] else None
  )
  if not args['--gm'] and not args['--ids']:
    warn ("No 'gm' or 'ids' specified. Using mean 'gm' as default.")
    sizing_spec.gm = [device.lut['gm'].mean().astype(float)]
  dcop, sizing = device.sizing(sizing_spec, return_dcop=True)
  electric_model = device.electric_model(dcop, sizing)
  
  # format info for output to SI units
  
  dcop_df = dcop.to_df()
  sizing_df = sizing.to_df()
  
  electric_model_df = electric_model.to_df()
  
  
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
  
  print()
  print("Summary:")
  print('DC-OP:')
  print(dcop_df)
  print()
  print('Sizing:')
  print(sizing_df)
  print()
  print('Electric Model:')  
  print(electric_model_df)

  if args['--gui']:
    gui = GuiApp(device)
    gui.run_device_sizing(args, dcop, sizing, electric_model, verbose=1 if args['--verbose'] else 0, tol=1e-2)
  
def moscap_sizing(args, cfg):
  raise NotImplementedError("MOSCAP sizing not implemented.")

def switch_sizing(args, cfg):
  raise NotImplementedError("Switch sizing not implemented.")

def electrical_params_from_sizing_dcop(args, cfg):
  raise NotImplementedError("Electrical parameters from sizing and DC-OP not implemented.")


def app(args, cfg) -> bool:
  
  if args['--verbose']:
    print('Arguments:')
    pprint(args)
    
    print('Configuration:')
    pprint(cfg)
    
  if args['--gmid'] or ( not args['--ron'] and not args['--cgg']):
    return device_sizing(args, cfg)
  if args['--ron']:
    return switch_sizing(args, cfg)
  if args['--cgg']:
    return moscap_sizing(args, cfg)
  
  return False

def config(args):
  cfg = None
  with open(__DEFAULT_CFG_PATH__, 'r') as f:
    cfg = safe_load(f)
  if cfg is None:
    raise ValueError("Configuration file not found.")
  if cfg.get('cfg-path', '') != '':
    CFG_PATH = str(Path(cfg.get('cfg-path', None)).resolve())
    with open(CFG_PATH, 'r') as f:
      cfg = safe_load(f)
  if args["--setup-config"]:
    CFG_PATH = Path(args["<cfg-path>"]).resolve()
    new_cfg = None
    with open(CFG_PATH, 'r') as f:
      new_cfg = safe_load(f)
    if new_cfg is not None:
      with open(__DEFAULT_CFG_PATH__, 'w') as f:
        cfg['cfg-path'] = str(CFG_PATH)
        f.write(safe_dump(cfg, default_flow_style=False, ident=2, width=80)) 
        cfg = new_cfg
  return cfg

def main():  
  args = docopt(__doc__, version='novaad 0.1')
  cfg = config(args)
  if not args["COMMAND_FILE"]:
    app(args, cfg)
  else:
    fp = Path(args["COMMAND_FILE"])
    assert fp.exists(), "Command file does not exist."
    assert fp.is_file(), "Command file must be a file."
    ext = fp.suffixes[0] if len(fp.suffixes) > 0 else ''
    assert ext in ['.txt', ''], "Command file must be a text file."
    with open(args["COMMAND_FILE"], 'r') as f:
      for line in f.readlines():
        line = line if len(line) > 0 else '-h'
        argv = line.split()
        try:
          args = docopt(__doc__, argv=argv, version='novaad 0.1')
          app(args, cfg)
        except DocoptExit:
          continue
        except Exception as e:
          print(f"Input:\t{line}\tOutput: {e}")
          
if __name__ == '__main__':
  main()